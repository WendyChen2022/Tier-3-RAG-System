# Tier-3 Agentic RAG System

Production-grade Retrieval-Augmented Generation system featuring hybrid search,
BGE-Reranker, self-correcting agent loop, semantic caching, and RAGAS evaluation.

## Architecture

```
[PDF Upload]
     │
     ▼
[Docling Parser] ──► [HybridChunker]
                            │
               ┌────────────┴────────────┐
          [ChromaDB]                 [BM25 Index]
       (all-MiniLM-L6-v2)           (keyword)
               └────────────┬────────────┘
                    [RRF Fusion — top 20]
                            │
                   [BGE-Reranker — top 5]   ← NEW
                            │
               [SelfCorrectionAgent Loop]
               ├── Grade relevance (Claude)
               ├── Rewrite query if score < 0.7
               └── Generate grounded answer
                            │
                   [FastAPI /query]
                            │
               [Redis Semantic Cache]
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-haiku-4-5 (Anthropic) |
| Embeddings | all-MiniLM-L6-v2 (local, free) |
| Reranker | BAAI/bge-reranker-base (local, free) |
| PDF Parsing | Docling |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 (LangChain Community) |
| Fusion | Reciprocal Rank Fusion |
| Semantic Cache | Redis |
| API | FastAPI + Uvicorn |
| Evaluation | RAGAS (10-sample dataset) |
| Logging | Loguru (structured JSON) |

## Quick Start

### 1. Configure
```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY — that's the only key you need
```

### 2. Start infrastructure
```bash
docker compose up chromadb redis -d
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API
```bash
uvicorn app.main:app --reload
```

### 5. Ingest a document
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@your_document.pdf"
```

### 6. Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about X?"}'
```

Response includes observability metrics:
```json
{
  "answer": "...",
  "iterations": 1,
  "converged": true,
  "latency_ms": 342.1,
  "reranker_scores": [0.921, 0.874, 0.812, 0.743, 0.698],
  "source_documents": [...]
}
```

### 7. Run RAGAS evaluation
```bash
python -c "from app.eval.ragas_eval import run_evaluation; run_evaluation()"
```

## Key Design Decisions

**BGE-Reranker (two-stage retrieval)**: Initial hybrid retrieval fetches 20
candidates; BGE-Reranker cross-encoder re-scores and returns top 5. This
eliminates the "Converged: No" problem by ensuring the LLM receives only
genuinely relevant context.

**Local embeddings + reranker**: all-MiniLM-L6-v2 and bge-reranker-base both
run locally — no embedding API costs. The sentence-transformers models are
already downloaded by HybridChunker at first run.

**Single API key**: Only `ANTHROPIC_API_KEY` is required. LLM (Claude) handles
grading, rewriting, and answering. Everything else runs locally.

**Observability**: Every query response includes `latency_ms` and
`reranker_scores`. Every retrieval writes a structured JSON log entry to
`logs/app.log` with per-chunk scores for post-hoc analysis.

**RAGAS evaluation**: `data/eval_dataset.json` contains 10 samples covering
all major RAG components. Run evaluation to get Faithfulness, Answer Relevancy,
Context Precision, and Context Recall scores.
