"""Unit tests for the self-correction agent."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document

from app.agents.self_correction import SelfCorrectionAgent, CorrectionState


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(
        return_value=[Document(page_content="Relevant context.", metadata={"source": "test.pdf"})]
    )
    return retriever


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


async def test_correction_state_defaults():
    state = CorrectionState(question="What is RAG?")
    assert state.iterations == 0
    assert state.converged is False
    assert state.documents == []
