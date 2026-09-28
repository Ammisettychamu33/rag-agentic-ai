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
from app import app

client = TestClient(app)

# ============================================================================
# 1. Health Check Endpoint Tests
# ============================================================================

def test_health_endpoint():
    """
    Verify GET /health returns 200 OK and expected JSON schema.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-agentic-ai-api"
    assert "version" in data


def test_root_endpoint():
    """
    Verify GET / returns 200 OK and service metadata.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "endpoints" in data


# ============================================================================
# 2. Chat Endpoint Validation Tests
# ============================================================================

def test_chat_empty_question_rejected():
    """
    Verify POST /chat rejects empty string with 422 Unprocessable Entity.
    """
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_whitespace_question_rejected():
    """
    Verify POST /chat rejects whitespace-only string with 422.
    """
    response = client.post("/chat", json={"question": "    "})
    assert response.status_code == 422


def test_chat_missing_question_field_rejected():
    """
    Verify POST /chat rejects missing payload fields with 422.
    """
    response = client.post("/chat", json={})
    assert response.status_code == 422


# ============================================================================
# 3. Chat Endpoint Logic & Response Structure Tests (Mocked Graph)
# ============================================================================

def test_chat_successful_response_structure():
    """
    Verify POST /chat with valid payload returns grounded answer, confidence score,
    and retrieved chunk metadata (using a mocked LangGraph workflow without live API calls).
    """
    mock_chunks = [
        Document(
            page_content="Agentic AI refers to autonomous systems capable of goal-directed behavior.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 3}
        ),
        Document(
            page_content="Key components include perception, reasoning, tool use, and reflection.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 4}
        )
    ]

    mock_graph_output = {
        "question": "What is Agentic AI according to the eBook?",
        "context": mock_chunks,
        "answer": "According to the eBook, Agentic AI refers to autonomous systems with goal-directed behavior.",
        "score": 0.8950
    }

    # Patch rag_app.invoke so no live OpenAI/Pinecone calls are triggered
    with patch("app.rag_app.invoke", return_value=mock_graph_output) as mock_invoke:
        payload = {"question": "What is Agentic AI according to the eBook?"}
        response = client.post("/chat", json=payload)

        # 1. Assert status code
        assert response.status_code == 200

        # 2. Assert response schema
        data = response.json()
        assert data["question"] == "What is Agentic AI according to the eBook?"
        assert "Agentic AI refers to autonomous systems" in data["answer"]
        assert data["confidence_score"] == 0.8950
        assert len(data["retrieved_chunks"]) == 2

        # 3. Assert chunk structure
        chunk1 = data["retrieved_chunks"][0]
        assert chunk1["source"] == "Ebook-Agentic-AI.pdf"
        assert chunk1["page"] == 3
        assert "Agentic AI refers to autonomous" in chunk1["content"]

        # 4. Verify graph was invoked with correct initial state
        mock_invoke.assert_called_once_with({
            "question": "What is Agentic AI according to the eBook?",
            "context": [],
            "answer": "",
            "score": 0.0
        })


def test_openapi_schema_generated():
    """
    Verify that FastAPI generates the OpenAPI 3.x schema correctly.
    """
    openapi = app.openapi()
    assert openapi["info"]["title"] == "Agentic AI RAG API"
    assert "/chat" in openapi["paths"]
    assert "/health" in openapi["paths"]
