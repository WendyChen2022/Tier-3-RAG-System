"""FastAPI route definitions — includes observability metrics in response."""
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from loguru import logger

from app.agents.self_correction import SelfCorrectionAgent
from app.api.schemas import IngestResponse, QueryRequest, QueryResponse
from app.api.dependencies import get_agent, get_ingestor

router = APIRouter(prefix="/api/v1")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    agent: SelfCorrectionAgent = Depends(get_agent),
) -> QueryResponse:
    t0 = time.perf_counter()
    logger.info(f"Query received: {request.question[:80]}")

    state = await agent.run(request.question)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Extract reranker scores from retrieved doc metadata for the response
    reranker_scores = [
        d.metadata.get("reranker_score", 0.0)
        for d in state.documents
    ]

    logger.info(
        "query_complete",
        latency_ms=latency_ms,
        iterations=state.iterations,
        converged=state.converged,
        reranker_scores=reranker_scores,
    )

    return QueryResponse(
        answer=state.answer,
        iterations=state.iterations,
        converged=state.converged,
        source_documents=[d.metadata for d in state.documents],
        latency_ms=latency_ms,
        reranker_scores=reranker_scores,
    )


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile,
    ingestor=Depends(get_ingestor),
) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    chunk_count = await ingestor.ingest_bytes(contents, filename=file.filename)
    logger.info(f"Ingested {file.filename}: {chunk_count} chunks")
    return IngestResponse(filename=file.filename, chunk_count=chunk_count)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
