"""Self-correcting RAG agent using iterative retrieval and answer grading."""
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader. Given a question and a retrieved context, "
            "output a JSON object with a single key 'score' (float 0-1) reflecting "
            "how well the context supports answering the question. Output ONLY JSON.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter. Rewrite the question to improve retrieval. "
            "Output only the rewritten question, no explanation.",
        ),
        ("human", "Original question: {question}"),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context is insufficient, say so explicitly.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class CorrectionState:
    question: str
    documents: list[Document] = field(default_factory=list)
    answer: str = ""
    iterations: int = 0
    converged: bool = False


class SelfCorrectionAgent:
    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever) -> None:
        self._llm = llm
        self._retriever = retriever
        self._grader = _GRADE_PROMPT | llm | StrOutputParser()
        self._rewriter = _REWRITE_PROMPT | llm | StrOutputParser()
        self._answerer = _ANSWER_PROMPT | llm | StrOutputParser()

    async def run(self, question: str) -> CorrectionState:
        state = CorrectionState(question=question)

        while state.iterations < settings.max_correction_iterations:
            state.iterations += 1
            logger.info(f"Correction loop iteration {state.iterations}")

            state.documents = await self._retriever.ainvoke(state.question)
            context = "\n\n".join(d.page_content for d in state.documents)

            grade_raw = await self._grader.ainvoke(
                {"question": state.question, "context": context}
            )
            score = self._parse_grade(grade_raw)
            logger.debug(f"Relevance score: {score:.3f}")

            if score >= settings.correction_relevance_threshold:
                state.answer = await self._answerer.ainvoke(
                    {"question": state.question, "context": context}
                )
                state.converged = True
                break

            logger.info("Relevance below threshold — rewriting query")
            state.question = await self._rewriter.ainvoke({"question": state.question})

        if not state.converged:
            logger.warning("Max iterations reached without convergence")
            context = "\n\n".join(d.page_content for d in state.documents)
            state.answer = await self._answerer.ainvoke(
                {"question": question, "context": context}
            )

        return state

    @staticmethod
    def _parse_grade(raw: str) -> float:
        import json
        try:
            return float(json.loads(raw).get("score", 0.0))
        except Exception:
            return 0.0
