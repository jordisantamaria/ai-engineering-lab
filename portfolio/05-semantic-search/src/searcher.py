"""
Semantic search engine with hybrid search and cross-encoder reranking.

Combines dense vector retrieval (FAISS) with sparse keyword matching
(BM25) for robust search. Optionally reranks top results with a
cross-encoder for maximum precision at the top.
"""

import math
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from indexer import DocumentIndexer


class BM25:
    """
    Simple BM25 implementation for keyword-based document scoring.

    Used as the sparse retrieval component in hybrid search to complement
    the dense semantic retrieval from FAISS.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 parameters.

        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_term_freqs: List[Counter] = []
        self.doc_count: int = 0
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        """
        Build the BM25 index from a list of document texts.

        Args:
            documents: List of raw document strings.
        """
        self.doc_count = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []

        # Document frequency for IDF computation
        df: Counter = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            # Count document frequency (each term counted once per doc)
            df.update(set(tokens))

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 1
        )

        # Compute IDF for each term
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.doc_count - freq + 0.5) / (freq + 0.5) + 1
            )

    def score(self, query: str) -> List[float]:
        """
        Score all documents against a query.

        Args:
            query: Search query string.

        Returns:
            List of BM25 scores, one per document.
        """
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            tf = self.doc_term_freqs[i]
            doc_len = self.doc_lengths[i]

            for token in query_tokens:
                if token not in self.idf:
                    continue

                term_freq = tf.get(token, 0)
                idf = self.idf[token]

                # BM25 scoring formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()


class SemanticSearcher:
    """
    Semantic search engine with hybrid retrieval and reranking.

    Combines:
    1. Dense retrieval via FAISS (semantic similarity)
    2. Sparse retrieval via BM25 (keyword matching)
    3. Optional cross-encoder reranking for top results
    """

    def __init__(
        self,
        indexer: DocumentIndexer,
        use_bm25: bool = True,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        """
        Initialize the semantic searcher.

        Args:
            indexer: Fitted DocumentIndexer with documents and FAISS index.
            use_bm25: Whether to include BM25 keyword matching.
            use_reranker: Whether to apply cross-encoder reranking.
            reranker_model: Model name for the cross-encoder reranker.
            semantic_weight: Weight for semantic scores in hybrid mode.
            bm25_weight: Weight for BM25 scores in hybrid mode.
        """
        self.indexer = indexer
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self._reranker = None
        self._bm25: Optional[BM25] = None

        # Build BM25 index if enabled
        if use_bm25 and indexer.documents:
            self._build_bm25()

    def _build_bm25(self) -> None:
        """Build the BM25 index from the indexer's documents."""
        texts = [doc.get("text", "") for doc in self.indexer.documents]
        self._bm25 = BM25()
        self._bm25.fit(texts)

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Perform semantic search for a query.

        Uses dense retrieval (FAISS) only. For hybrid search that
        includes keyword matching, use hybrid_search().

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with document info and scores,
            sorted by relevance descending.
        """
        start_time = time.time()

        # Encode the query
        query_embedding = self.indexer.encode_documents(
            [query], show_progress=False
        )

        # Search FAISS index
        results = self.indexer.search(query_embedding, top_k=top_k)

        # Build result list
        search_results = []
        for rank, (position, score) in enumerate(results, 1):
            if position < len(self.indexer.documents):
                doc = self.indexer.documents[position]
                search_results.append(
                    {
                        "document_id": doc.get("id", f"doc_{position}"),
                        "text": doc.get("text", ""),
                        "score": round(score, 4),
                        "rank": rank,
                        "source": "semantic",
                    }
                )

        search_time = (time.time() - start_time) * 1000

        # Attach timing to the first result (or return empty)
        for r in search_results:
            r["search_time_ms"] = round(search_time, 2)

        return search_results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Retrieves candidates from both FAISS (semantic) and BM25
        (keyword), normalizes their scores, and produces a weighted
        combination for the final ranking.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            semantic_weight: Override default semantic weight.
            bm25_weight: Override default BM25 weight.

        Returns:
            List of result dicts sorted by combined score.
        """
        if self._bm25 is None or not self.use_bm25:
            return self.search(query, top_k=top_k)

        start_time = time.time()

        sw = semantic_weight if semantic_weight is not None else self.semantic_weight
        bw = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Normalize weights
        total = sw + bw
        sw /= total
        bw /= total

        # Get semantic results (fetch extra for merging)
        fetch_k = min(top_k * 3, len(self.indexer.documents))
        query_embedding = self.indexer.encode_documents(
            [query], show_progress=False
        )
        semantic_results = self.indexer.search(query_embedding, top_k=fetch_k)

        # Get BM25 scores for all documents
        bm25_scores = self._bm25.score(query)

        # Normalize semantic scores to [0, 1]
        sem_scores_dict: Dict[int, float] = {}
        max_sem = max((s for _, s in semantic_results), default=1.0)
        min_sem = min((s for _, s in semantic_results), default=0.0)
        range_sem = max_sem - min_sem if max_sem != min_sem else 1.0

        for pos, score in semantic_results:
            sem_scores_dict[pos] = (score - min_sem) / range_sem

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
        range_bm25 = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0

        bm25_norm = {
            i: (s - min_bm25) / range_bm25
            for i, s in enumerate(bm25_scores)
            if s > 0
        }

        # Merge all candidate positions
        all_positions = set(sem_scores_dict.keys()) | set(bm25_norm.keys())

        # Compute combined scores
        combined = {}
        for pos in all_positions:
            s_sem = sem_scores_dict.get(pos, 0.0)
            s_bm25 = bm25_norm.get(pos, 0.0)
            combined[pos] = sw * s_sem + bw * s_bm25

        # Sort by combined score
        sorted_positions = sorted(
            combined.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        search_time = (time.time() - start_time) * 1000

        results = []
        for rank, (pos, score) in enumerate(sorted_positions, 1):
            if pos < len(self.indexer.documents):
                doc = self.indexer.documents[pos]
                results.append(
                    {
                        "document_id": doc.get("id", f"doc_{pos}"),
                        "text": doc.get("text", ""),
                        "score": round(score, 4),
                        "rank": rank,
                        "source": "hybrid",
                        "search_time_ms": round(search_time, 2),
                    }
                )

        return results

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank search results using a cross-encoder model.

        Cross-encoders are more accurate than bi-encoders but slower,
        so they are applied only to the top candidates from the initial
        retrieval stage.

        Args:
            query: Original search query.
            results: List of search result dicts from search() or
                    hybrid_search().
            top_k: Number of reranked results to return. Defaults to
                  the length of the input results.

        Returns:
            Reranked list of result dicts with updated scores.
        """
        if not results:
            return results

        reranker = self._get_reranker()

        # Prepare query-document pairs for the cross-encoder
        pairs = [(query, r["text"]) for r in results]
        scores = reranker.predict(pairs)

        # Attach reranker scores
        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 4)
            result["original_score"] = result["score"]
            result["score"] = round(float(score), 4)

        # Sort by reranker score
        results.sort(key=lambda r: r["score"], reverse=True)

        # Update ranks
        for i, result in enumerate(results, 1):
            result["rank"] = i
            result["source"] = "reranked"

        if top_k:
            results = results[:top_k]

        return results
