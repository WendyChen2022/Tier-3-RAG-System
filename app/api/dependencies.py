"""FastAPI dependency injectors — Claude LLM, local embeddings, BGE-Reranker."""
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers import BM25Retriever

from app.agents.self_correction import SelfCorrectionAgent
from app.core.config import get_settings
from app.engine.vector_store import build_vector_store
from app.engine.hybrid_retriever import HybridRetriever
from app.engine.ingestor import DocumentIngestor

settings = get_settings()


@lru_cache(maxsize=1)
def _build_agent() -> SelfCorrectionAgent:
    # ── LLM: Anthropic Claude ──────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )

    # ── Vector store (local HuggingFace embeddings) ────────────────────────────
    vector_store = build_vector_store()

    # BM25 retriever initialised with an empty corpus at startup;
    # it is rebuilt with real documents after the first ingestion.
    bm25_retriever = BM25Retriever.from_texts(
        ["placeholder"],
        metadatas=[{"source": "init"}],
    )
    bm25_retriever.k = settings.retriever_top_k

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        top_k=settings.retriever_top_k,
        top_n=settings.reranker_top_n,
    )

    return SelfCorrectionAgent(llm=llm, retriever=retriever)


def get_agent() -> SelfCorrectionAgent:
    return _build_agent()


def get_ingestor() -> DocumentIngestor:
    return DocumentIngestor(vector_store=build_vector_store())
