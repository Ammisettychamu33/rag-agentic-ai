import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import app
from src.graph import (
    AgentState,
    format_context_for_prompt,
    retrieve_node,
    generate_node,
    create_rag_graph,
    SYSTEM_PROMPT
)

client = TestClient(app)

# ============================================================================
# Mock eBook Context Chunks (Derived from Ebook-Agentic-AI.pdf)
# ============================================================================
EBOOK_CHUNKS = {
    "agentic_ai_def": [
        Document(
            page_content="Agentic AI represents a paradigm shift where AI systems possess autonomy, goal-oriented behavior, and decision-making capabilities without constant human intervention.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 5}
        ),
        Document(
            page_content="Unlike reactive AI models, Agentic AI proactively assesses environments, plans multi-step actions, and executes complex workflows.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 6}
        )
    ],
    "traditional_vs_agentic": [
        Document(
            page_content="Traditional automation systems follow rigid, deterministic rule-based scripts. In contrast, AI agents adapt dynamically to changing contexts, reason through ambiguity, and handle unexpected states.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12}
        )
    ],
    "core_components": [
        Document(
            page_content="The core components of an Agentic Architecture include: 1) Perception & Environment Sensing, 2) Planning & Reasoning Engine, 3) Memory Module (short-term & long-term), and 4) Tool Use & Action Execution.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 18}
        )
    ],
    "memory_role": [
        Document(
            page_content="Memory plays a foundational role in agentic workflows by providing conversational continuity, preserving task state, recalling past tool executions, and enabling self-reflection and learning over time.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 24}
        )
    ],
    "empty_or_irrelevant": []
}

EXPECTED_REFUSAL_MESSAGE = "I cannot answer this question based on the provided Agentic AI eBook as the information is not present in the text."


# ============================================================================
# 1. Unit Tests for Prompt Formatting & Context Isolation
# ============================================================================

def test_format_context_with_metadata():
    """
    Verify that format_context_for_prompt preserves source and page citations.
    """
    docs = EBOOK_CHUNKS["agentic_ai_def"]
    formatted = format_context_for_prompt(docs)

    assert "[Chunk #1 | Source: Ebook-Agentic-AI.pdf | Page: 5]" in formatted
    assert "[Chunk #2 | Source: Ebook-Agentic-AI.pdf | Page: 6]" in formatted
    assert "paradigm shift" in formatted


def test_format_context_empty():
    """
    Verify that empty document lists produce a clear fallback text.
    """
    formatted = format_context_for_prompt([])
    assert formatted == "No relevant context found."


def test_system_prompt_strictly_forbids_hallucination():
    """
    Verify that the system prompt explicitly contains the required strict refusal string and rules.
    """
    assert "STRICTLY and SOLELY based on the provided eBook context" in SYSTEM_PROMPT
    assert EXPECTED_REFUSAL_MESSAGE in SYSTEM_PROMPT


# ============================================================================
# 2. Benchmark Query Tests (Mocked LLM & Vector Store)
# ============================================================================

@pytest.mark.parametrize(
    "query,context_key,expected_answer_snippet,expected_score",
    [
        (
            "What is Agentic AI according to the eBook?",
            "agentic_ai_def",
            "Agentic AI represents a paradigm shift where AI systems possess autonomy and goal-oriented behavior.",
            0.8850
        ),
        (
            "How do AI agents differ from traditional automation systems?",
            "traditional_vs_agentic",
            "Traditional automation follows deterministic rule-based scripts, whereas AI agents adapt dynamically.",
            0.8620
        ),
        (
            "What are core components of Agentic Architecture?",
            "core_components",
            "The core components include Perception, Planning & Reasoning, Memory, and Tool Execution.",
            0.9100
        ),
        (
            "What role does memory play in Agentic AI workflows?",
            "memory_role",
            "Memory provides conversational continuity, preserves task state, and enables self-reflection.",
            0.8740
        )
    ]
)
def test_in_domain_benchmark_queries(query, context_key, expected_answer_snippet, expected_score):
    """
    Verify that in-domain benchmark queries (Queries 1-4) return grounded answers,
    retain source/page metadata, and produce confidence scores between 0.0 and 1.0.
    """
    chunks = EBOOK_CHUNKS[context_key]

    mock_graph_output = {
        "question": query,
        "context": chunks,
        "answer": expected_answer_snippet,
        "score": expected_score
    }

    with patch("app.rag_app.invoke", return_value=mock_graph_output):
        response = client.post("/chat", json={"question": query})
        assert response.status_code == 200

        data = response.json()
        assert data["question"] == query
        assert data["answer"] == expected_answer_snippet
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert len(data["retrieved_chunks"]) == len(chunks)

        for chunk in data["retrieved_chunks"]:
            assert chunk["source"] == "Ebook-Agentic-AI.pdf"
            assert isinstance(chunk["page"], int)
            assert len(chunk["content"]) > 0


def test_out_of_domain_query_fifa_world_cup_refusal():
    """
    Benchmark Query 5: "Who won 2022 FIFA World Cup?"
    Verify that out-of-domain questions with no relevant eBook context are strictly refused.
    """
    query = "Who won 2022 FIFA World Cup?"

    mock_graph_output = {
        "question": query,
        "context": [],
        "answer": EXPECTED_REFUSAL_MESSAGE,
        "score": 0.0
    }

    with patch("app.rag_app.invoke", return_value=mock_graph_output):
        response = client.post("/chat", json={"question": query})
        assert response.status_code == 200

        data = response.json()
        assert data["question"] == query
        assert data["answer"] == EXPECTED_REFUSAL_MESSAGE
        assert data["confidence_score"] == 0.0
        assert data["retrieved_chunks"] == []
        assert "Argentina" not in data["answer"]
        assert "FIFA" not in data["answer"] or EXPECTED_REFUSAL_MESSAGE in data["answer"]


# ============================================================================
# 3. Direct Node-Level Grounding Tests (Without Live APIs)
# ============================================================================

def test_generate_node_handles_empty_context_directly():
    """
    Verify generate_node immediately returns refusal when context list is empty,
    without attempting any LLM calls.
    """
    state: AgentState = {
        "question": "Who won 2022 FIFA World Cup?",
        "context": [],
        "answer": "",
        "score": 0.0
    }

    result = generate_node(state)
    assert result["answer"] == EXPECTED_REFUSAL_MESSAGE
    assert result["score"] == 0.0


def test_generate_node_with_mocked_llm_chain():
    """
    Verify generate_node properly passes formatted context to the LLM chain and returns state.
    """
    state: AgentState = {
        "question": "What is Agentic AI?",
        "context": EBOOK_CHUNKS["agentic_ai_def"],
        "answer": "",
        "score": 0.85
    }

    mock_llm_response = AIMessage(content="Agentic AI represents autonomous goal-directed systems (Page 5).")

    with patch("src.graph.ChatOpenAI") as mock_chat_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response
        mock_chat_cls.return_value = mock_instance

        with patch("src.graph.PROMPT_TEMPLATE") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_llm_response
            mock_prompt.__or__.return_value = mock_chain

            result = generate_node(state)
            assert result["answer"] == "Agentic AI represents autonomous goal-directed systems (Page 5)."
            assert result["score"] == 0.85
