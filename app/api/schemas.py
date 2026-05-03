from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2048)


class QueryResponse(BaseModel):
    answer: str
    iterations: int
    converged: bool
    source_documents: list[dict]
    # Observability fields
    latency_ms: float = 0.0
    reranker_scores: list[float] = Field(default_factory=list)


class IngestResponse(BaseModel):
    filename: str
    chunk_count: int
