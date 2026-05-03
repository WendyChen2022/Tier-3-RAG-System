"""
Hybrid BM25 + dense vector retriever with RRF fusion and BGE-Reranker.

Pipeline
--------
1. Dense retrieval  : ChromaDB vector search (top_k=20)
2. Sparse retrieval : BM25 keyword search    (top_k=20)
3. RRF fusion       : merge & deduplicate both lists
4. BGE-Reranker     : cross-encoder re-scores fused candidates
5. Return top_n     : best 5 chunks go to the LLM

This two-stage approach solves "Converged: No" by ensuring the LLM
receives genuinely relevant chunks rather than approximate ANN matches.

Observability
-------------
Every retrieval logs a structured JSON record containing:
  - query, latency_ms
  - per-chunk reranker scores
  - top chunk preview
These records appear in logs/app.log for post-hoc analysis.
"""
from __future__ import annotations

import time
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from loguru import logger
from pydantic import Field

from app.core.config import get_settings

settings = get_settings()


# ── BGE-Reranker (loaded once, shared across requests) ────────────────────────

@lru_cache(maxsize=1)
def _load_reranker():
    """Load BAAI/bge-reranker-base once and cache it in memory."""
    from sentence_transformers import CrossEncoder
    logger.info(f"Loading reranker: {settings.reranker_model}")
    model = CrossEncoder(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=512,
    )
    logger.info("Reranker loaded ✓")
    return model


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in results:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.metadata.get("source", "") + doc.page_content[:64]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
            doc_map[doc_id] = doc

    return sorted(
        [(doc_map[doc_id], score) for doc_id, score in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    BM25 + dense fusion → BGE-Reranker pipeline.

    top_k  : candidates fetched from each retriever (default 20)
    top_n  : chunks returned to the LLM after reranking (default 5)
    """
    vector_store: Chroma
    bm25_retriever: BM25Retriever
    top_k: int = Field(default_factory=lambda: settings.retriever_top_k)
    top_n: int = Field(default_factory=lambda: settings.reranker_top_n)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        t0 = time.perf_counter()

        # ── Stage 1: hybrid retrieval ─────────────────────────────────────────
        dense_results  = self.vector_store.similarity_search(query, k=self.top_k)
        sparse_results = self.bm25_retriever.get_relevant_documents(query)

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        candidates = [doc for doc, _ in fused[: self.top_k]]

        # ── Stage 2: BGE-Reranker ─────────────────────────────────────────────
        reranker = _load_reranker()
        pairs = [(query, doc.page_content) for doc in candidates]
        reranker_scores: list[float] = reranker.predict(pairs).tolist()

        # Attach reranker score to each doc's metadata for observability
        scored = sorted(
            zip(candidates, reranker_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_docs = []
        for doc, score in scored[: self.top_n]:
            doc.metadata["reranker_score"] = round(score, 4)
            top_docs.append(doc)

        # ── Observability log ─────────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "retrieval_metrics",
            query=query[:80],
            latency_ms=latency_ms,
            candidates_fetched=len(candidates),
            chunks_returned=len(top_docs),
            reranker_scores=[round(s, 4) for _, s in scored[: self.top_n]],
            top_chunk_preview=top_docs[0].page_content[:120] if top_docs else "",
        )

        return top_docs
