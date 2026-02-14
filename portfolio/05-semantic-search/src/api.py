"""
FastAPI application for semantic search.

Provides endpoints to index documents, perform semantic and hybrid
searches, and retrieve index statistics.
"""

import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from indexer import DocumentIndexer
from searcher import SemanticSearcher


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class DocumentInput(BaseModel):
    """Schema for a single document to index."""
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")


class IndexRequest(BaseModel):
    """Schema for the /index endpoint request."""
    documents: List[DocumentInput] = Field(
        ..., description="List of documents to index"
    )


class IndexResponse(BaseModel):
    """Schema for the /index endpoint response."""
    indexed_count: int
    total_documents: int
    message: str


class SearchRequest(BaseModel):
    """Schema for the /search endpoint request."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")
    hybrid: bool = Field(
        True, description="Use hybrid search (semantic + keyword)"
    )
    rerank: bool = Field(
        False, description="Apply cross-encoder reranking to results"
    )


class SearchResult(BaseModel):
    """Schema for a single search result."""
    document_id: str
    text: str
    score: float
    rank: int
    source: str


class SearchResponse(BaseModel):
    """Schema for the /search endpoint response."""
    results: List[SearchResult]
    query: str
    total_results: int
    search_time_ms: float


class StatsResponse(BaseModel):
    """Schema for the /stats endpoint response."""
    num_documents: int
    num_vectors: int
    embedding_dimension: Optional[int]
    model_name: str
    index_type: str


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""
    status: str
    index_ready: bool
    num_documents: int


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic Search API",
    description=(
        "Semantic document search using Sentence-BERT embeddings and FAISS. "
        "Supports hybrid search (semantic + BM25) and cross-encoder reranking."
    ),
    version="1.0.0",
)

# Initialize the indexer and searcher
_indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2", index_type="flat")
_searcher: Optional[SemanticSearcher] = None


def _get_searcher() -> SemanticSearcher:
    """Get or create the SemanticSearcher instance."""
    global _searcher
    if _searcher is None or len(_indexer.documents) != len(
        getattr(_searcher, "_last_doc_count", -1) or []
    ):
        _searcher = SemanticSearcher(
            indexer=_indexer,
            use_bm25=True,
            use_reranker=False,
        )
        _searcher._last_doc_count = len(_indexer.documents)
    return _searcher


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the current health status of the service."""
    return HealthResponse(
        status="healthy",
        index_ready=_indexer.index is not None,
        num_documents=len(_indexer.documents),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return statistics about the current index."""
    stats = _indexer.get_stats()
    return StatsResponse(**stats)


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Index a batch of documents for semantic search.

    Encodes document texts into embeddings and adds them to the
    FAISS vector index. Documents can be added incrementally.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    try:
        docs = [{"id": d.id, "text": d.text} for d in request.documents]
        count = _indexer.add_documents(docs)

        # Rebuild the searcher to include new documents in BM25
        global _searcher
        _searcher = None  # Force rebuild on next search

        return IndexResponse(
            indexed_count=count,
            total_documents=len(_indexer.documents),
            message=f"Successfully indexed {count} documents.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(exc)}",
        )


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search indexed documents using semantic similarity.

    Supports three modes:
    - Semantic only: dense vector retrieval via FAISS
    - Hybrid: semantic + BM25 keyword matching (default)
    - Reranked: hybrid + cross-encoder reranking for top precision
    """
    if _indexer.index is None or len(_indexer.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Use /index first.",
        )

    try:
        start_time = time.time()
        searcher = _get_searcher()

        if request.hybrid:
            results = searcher.hybrid_search(
                query=request.query, top_k=request.top_k
            )
        else:
            results = searcher.search(
                query=request.query, top_k=request.top_k
            )

        # Optional reranking
        if request.rerank and results:
            results = searcher.rerank(
                query=request.query,
                results=results,
                top_k=request.top_k,
            )

        search_time = (time.time() - start_time) * 1000

        search_results = [
            SearchResult(
                document_id=r["document_id"],
                text=r["text"],
                score=r["score"],
                rank=r["rank"],
                source=r.get("source", "semantic"),
            )
            for r in results
        ]

        return SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            search_time_ms=round(search_time, 2),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
