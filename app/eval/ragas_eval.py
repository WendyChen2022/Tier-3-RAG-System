"""RAGAS evaluation harness — uses Claude (Anthropic) instead of OpenAI."""
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.core.config import get_settings

settings = get_settings()

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


def _build_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.anthropic_api_key.get_secret_value(),
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.reranker_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def load_eval_dataset(path: str | None = None) -> Dataset:
    dataset_path = Path(path or settings.ragas_eval_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")
    with dataset_path.open() as f:
        data: list[dict[str, Any]] = json.load(f)
    return Dataset.from_list(data)


def run_evaluation(dataset: Dataset | None = None) -> dict[str, float]:
    ds = dataset or load_eval_dataset()
    metrics = [METRIC_MAP[m] for m in settings.ragas_metrics if m in METRIC_MAP]

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    logger.info(f"Running RAGAS evaluation on {len(ds)} samples with {len(metrics)} metrics using Claude")
    result = evaluate(ds, metrics=metrics)

    scores = {k: float(v) for k, v in result.items()}
    logger.info(f"RAGAS scores: {scores}")
    return scores