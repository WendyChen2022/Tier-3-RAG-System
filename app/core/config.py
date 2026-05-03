from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Tier-3 Agentic RAG System"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # LLM — Anthropic Claude
    anthropic_api_key: SecretStr = Field(..., validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Embeddings — local sentence-transformers (no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "rag_documents"

    # Redis (semantic cache)
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    semantic_cache_score_threshold: float = 0.95

    # Hybrid Search
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # RAG Pipeline
    retriever_top_k: int = 20       # fetch 20 candidates for reranker
    reranker_top_n: int = 5         # reranker keeps best 5 for LLM

    chunk_size: int = 512
    chunk_overlap: int = 64

    # BGE Reranker
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"    # change to "cuda" if GPU available

    # Docling parsing
    use_ocr: bool = True
    generate_page_images: bool = False

    # Chunking
    use_hybrid_chunker: bool = True
    hybrid_chunker_max_tokens: int = 200
    hybrid_chunker_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_chunker_merge_peers: bool = True

    # Self-correction Agent
    max_correction_iterations: int = 3
    correction_relevance_threshold: float = 0.7

    # RAGAS Evaluation
    ragas_eval_dataset_path: str = "data/eval_dataset.json"
    ragas_metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
