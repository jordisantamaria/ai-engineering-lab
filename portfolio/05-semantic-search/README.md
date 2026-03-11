# Semantic Search for Enterprise Documentation

## Business Problem

Keyword search in internal documentation has critical limitations in enterprise environments:

- **Misaligned vocabulary**: users search "how to request time off" but the document says "statutory absence request". Keyword search doesn't find the result.
- **Synonyms and context**: searching "cancel order" doesn't find documents about "purchase order annulment" or "product returns".
- **Information overload**: employees spend an average of 1.8 hours/day searching for internal information (McKinsey). In large companies, this cost is enormous.
- **Lost knowledge**: 80% of corporate knowledge is in unstructured documents (PDFs, wikis, emails, manuals). If it can't be found, it doesn't exist.

## Proposed Solution

Semantic search system that understands the **meaning** of queries, not just exact words. It uses language embeddings to represent documents and queries in a vector space where proximity indicates semantic similarity.

### Architecture

```
Corporate documents
        |
        v
  Sentence-BERT (encoding)
        |
        v
  Embeddings (384/768-dim vectors)
        |
        v
  FAISS Index (fast vector search)
        |
        v
  [Optional] Cross-Encoder Reranking
        |
        v
  Results ranked by semantic relevance

  ---

  User query
        |
        v
  Sentence-BERT (encoding)
        |
        v
  Query embedding
        |
        v
  FAISS nearest neighbor search
        |
        v
  Top-K documents + scores
```

### Key Components

1. **Sentence-BERT**: embedding model that converts text into dense vectors capturing semantic meaning.
2. **FAISS (Facebook AI Similarity Search)**: high-performance vector search library. Supports billions of vectors with response times <10ms.
3. **Cross-Encoder Reranking**: optional step that reranks initial results with a more precise model to improve top-5 quality.
4. **Hybrid search**: combines semantic search with keyword search (BM25) for the best of both worlds.

## Expected Results

| Metric | Keyword Search | Semantic Search |
|---------|---------------|-------------------|
| MRR (Mean Reciprocal Rank) | 0.35 | >0.65 |
| Recall@10 | 0.45 | >0.80 |
| Search time | ~50ms | <100ms |
| User satisfaction | 55% | >85% |

## Technologies

- **sentence-transformers**: state-of-the-art semantic embeddings
- **FAISS**: high-performance vector search (Meta AI)
- **FastAPI**: REST API for integration
- **numpy / pandas**: data and vector manipulation
- **Pydantic**: data validation

## How to Run

### 1. Installation

```bash
cd portfolio/05-semantic-search
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8004
```

### 3. Index Documents

```bash
curl -X POST "http://localhost:8004/index" \
    -H "Content-Type: application/json" \
    -d '{
        "documents": [
            {"id": "doc1", "text": "Procedure for requesting annual leave"},
            {"id": "doc2", "text": "Return policy for defective products"},
            {"id": "doc3", "text": "Onboarding guide for new employees"}
        ]
    }'
```

### 4. Search

```bash
curl -X POST "http://localhost:8004/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "how to request days off", "top_k": 5}'
```

Expected response:
```json
{
    "results": [
        {
            "document_id": "doc1",
            "text": "Procedure for requesting annual leave",
            "score": 0.8923,
            "rank": 1
        }
    ],
    "query": "how to request days off",
    "total_results": 1,
    "search_time_ms": 12.5
}
```

### 5. Docker

```bash
docker build -t semantic-search .
docker run -p 8004:8004 semantic-search
```

## How to Present It: Client Pitch

### Value Proposition

> "Your employees find the information they need 3x faster. Our semantic search understands what the user *means*, not just what they type. It works with documents in multiple languages and requires no manual labeling."

### Estimated ROI

**Scenario**: company with 1,000 employees who spend 30 minutes/day searching for internal information.

| Item | Before (keywords) | After (semantic) |
|----------|-------------------|---------------------|
| Search time/day/employee | 30 min | 10 min |
| Total hours lost/month | 10,000 h | 3,333 h |
| Lost productivity cost (30 EUR/h) | 300,000 EUR/month | 100,000 EUR/month |
| Successful searches (1st attempt) | 40% | >80% |

**Estimated savings: ~200,000 EUR/month** in recovered productivity. Additionally, reduction in errors from using outdated or incorrect information.

### Key Points for the Presentation

1. **Live demo**: index some of the client's documents and demonstrate semantic searches that fail with keywords.
2. **Multi-language**: the model supports searches in Spanish, English, Catalan, and other languages simultaneously.
3. **Incrementality**: documents can be added to the index without rebuilding it from scratch.
4. **Integration**: REST API that connects to any wiki, intranet, or document management system.
5. **Privacy**: everything runs on-premise, documents never leave the client's infrastructure.

### Frequently Asked Client Questions

- **"Does it work with PDFs and scanned documents?"** - Yes, it combines with OCR for scanned documents and with text extractors for native PDFs.
- **"How many documents can it handle?"** - FAISS scales to millions of documents with response times <100ms. For very large collections, an IVF (Inverted File) index is used.
- **"Does it need a GPU?"** - Not for serving (CPU is sufficient for search). GPU is recommended for initial indexing if there are many documents.
- **"Can it integrate with our SharePoint/Confluence?"** - Yes, via connectors that extract content and index it periodically.
