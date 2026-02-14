"""
Document indexer for semantic search using Sentence-BERT and FAISS.

Encodes documents into dense vectors and stores them in a FAISS index
for fast approximate nearest-neighbor retrieval. Supports saving,
loading, and incremental updates.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


class DocumentIndexer:
    """
    Semantic document indexer backed by FAISS.

    Converts text documents into embedding vectors using a
    SentenceTransformer model and indexes them in a FAISS vector
    store for sub-millisecond nearest-neighbor search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        nlist: int = 100,
    ):
        """
        Initialize the document indexer.

        Args:
            model_name: SentenceTransformer model for encoding text.
            index_type: FAISS index type - 'flat' (exact) or 'ivf'
                       (approximate, faster for large collections).
            nlist: Number of Voronoi cells for IVF index. Only used
                  when index_type='ivf'.
        """
        self.model_name = model_name
        self.index_type = index_type
        self.nlist = nlist

        self._model = None
        self.index: Optional[faiss.Index] = None
        self.dimension: Optional[int] = None

        # Document storage: maps internal index position to document
        self.documents: List[Dict] = []  # [{"id": ..., "text": ...}, ...]
        self._id_to_position: Dict[str, int] = {}

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts: List of document texts to encode.
            batch_size: Encoding batch size.
            show_progress: Whether to display a progress bar.

        Returns:
            Numpy array of shape (n_docs, embedding_dim) with L2-normalized
            embeddings.
        """
        model = self._get_model()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalize for cosine similarity via IP
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from precomputed embeddings.

        Args:
            embeddings: Numpy array of shape (n, dim) with float32 vectors.

        Returns:
            Trained FAISS index ready for search.
        """
        n, dim = embeddings.shape
        self.dimension = dim

        if self.index_type == "flat":
            # Exact search using inner product (cosine sim with normalized vectors)
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            # Approximate search for large collections
            quantizer = faiss.IndexFlatIP(dim)
            actual_nlist = min(self.nlist, max(1, n // 10))
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVF index requires training
            self.index.train(embeddings)
            # Set number of probes for search (higher = more accurate but slower)
            self.index.nprobe = min(10, actual_nlist)

        self.index.add(embeddings)

        print(
            f"Built {self.index_type} index with {self.index.ntotal} vectors "
            f"of dimension {dim}"
        )

        return self.index

    def index_documents(
        self,
        documents: List[Dict],
        text_key: str = "text",
        id_key: str = "id",
    ) -> int:
        """
        Index a batch of documents (encode + add to FAISS index).

        Each document should be a dict with at least a text field and
        an optional ID field.

        Args:
            documents: List of document dicts.
            text_key: Key containing the text to encode.
            id_key: Key containing the document ID.

        Returns:
            Number of documents indexed.
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.encode_documents(texts)

        if self.index is None:
            self.build_index(embeddings)
        else:
            self.index.add(embeddings)

        # Store document metadata
        for doc in documents:
            position = len(self.documents)
            doc_id = doc.get(id_key, f"doc_{position}")
            self._id_to_position[doc_id] = position
            self.documents.append(doc)

        print(f"Indexed {len(documents)} documents (total: {len(self.documents)})")
        return len(documents)

    def add_documents(
        self,
        new_docs: List[Dict],
        text_key: str = "text",
        id_key: str = "id",
    ) -> int:
        """
        Incrementally add new documents to an existing index.

        Args:
            new_docs: List of new document dicts to add.
            text_key: Key containing the text to encode.
            id_key: Key containing the document ID.

        Returns:
            Number of new documents added.
        """
        if self.index is None:
            # No existing index: create a new one
            return self.index_documents(new_docs, text_key, id_key)

        texts = [doc[text_key] for doc in new_docs]
        embeddings = self.encode_documents(texts)
        self.index.add(embeddings)

        for doc in new_docs:
            position = len(self.documents)
            doc_id = doc.get(id_key, f"doc_{position}")
            self._id_to_position[doc_id] = position
            self.documents.append(doc)

        print(f"Added {len(new_docs)} documents (total: {len(self.documents)})")
        return len(new_docs)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search the index for nearest neighbors of a query vector.

        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,).
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (position, score) tuples sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_documents() first.")

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Clamp top_k to available documents
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((int(idx), float(score)))

        return results

    def save_index(self, directory: str) -> None:
        """
        Save the FAISS index and document metadata to disk.

        Args:
            directory: Directory path to save the index files.
        """
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(
                self.index, os.path.join(directory, "faiss.index")
            )

        # Save document metadata
        metadata = {
            "documents": self.documents,
            "id_to_position": self._id_to_position,
            "model_name": self.model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
        }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Index saved to {directory}")

    def load_index(self, directory: str) -> None:
        """
        Load a saved FAISS index and document metadata from disk.

        Args:
            directory: Directory path containing the saved index files.
        """
        # Load FAISS index
        index_path = os.path.join(directory, "faiss.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

        # Load document metadata
        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.documents = metadata["documents"]
            self._id_to_position = metadata["id_to_position"]
            self.model_name = metadata.get("model_name", self.model_name)
            self.index_type = metadata.get("index_type", self.index_type)
            self.dimension = metadata.get("dimension")

        print(
            f"Index loaded from {directory}: "
            f"{len(self.documents)} documents, "
            f"{self.index.ntotal if self.index else 0} vectors"
        )

    def get_stats(self) -> Dict:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index size, model info, and document count.
        """
        return {
            "num_documents": len(self.documents),
            "num_vectors": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.dimension,
            "model_name": self.model_name,
            "index_type": self.index_type,
        }
