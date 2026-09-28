import os
import sys
from typing import List, Optional, Any, Union
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# Import the pre-compiled LangGraph RAG workflow
from src.graph import rag_app, AgentState

# ============================================================================
# 1. Pydantic Request & Response Schemas
# ============================================================================

class ChatRequest(BaseModel):
    """
    Request model for the /chat endpoint.
    """
    question: str = Field(
        ...,
        min_length=1,
        description="The user's query strictly regarding the Agentic AI eBook.",
        examples=["What is Agentic AI according to the eBook?"]
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return trimmed


class RetrievedChunk(BaseModel):
    """
    Represents an individual text chunk retrieved from the vector store.
    """
    content: str = Field(..., description="The textual content of the chunk.")
    source: str = Field(..., description="The source document name.")
    page: Union[int, str] = Field(..., description="The eBook page number.")


class ChatResponse(BaseModel):
    """
    Standardized response model for /chat.
    """
    question: str = Field(..., description="The original user query.")
    answer: str = Field(..., description="The grounded answer generated strictly from the eBook.")
    confidence_score: float = Field(..., description="Relevance / confidence score between 0.0 and 1.0.")
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="List of context chunks used to ground the answer."
    )


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str = "healthy"
    service: str = "rag-agentic-ai-api"
    version: str = "1.0.0"


# ============================================================================
# 2. FastAPI Application Initialization
# ============================================================================

app = FastAPI(
    title="Agentic AI RAG API",
    description="Grounded Question-Answering API for the Agentic AI eBook using LangGraph, Pinecone, and OpenAI.",
    version="1.0.0"
)

# Enable CORS for frontend clients (e.g., Streamlit, React, or browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 3. API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
def root_endpoint():
    """
    Root endpoint providing basic API information.
    """
    return {
        "service": "Agentic AI RAG Chatbot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
def health_check() -> HealthResponse:
    """
    Health check endpoint to verify that the service is running.
    """
    return HealthResponse(
        status="healthy",
        service="rag-agentic-ai-api",
        version="1.0.0"
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG Chat"]
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main RAG Chat endpoint:
    1. Validates the question.
    2. Invokes the LangGraph RAG workflow (retrieve -> generate).
    3. Returns the grounded answer, confidence score, and retrieved context chunks.
    """
    try:
        # Prepare initial state for the LangGraph workflow
        initial_state = {
            "question": request.question,
            "context": [],
            "answer": "",
            "score": 0.0
        }

        # Invoke the LangGraph pipeline
        result = rag_app.invoke(initial_state)

        # Extract output fields from the final state
        answer = result.get("answer", "")
        score = float(result.get("score", 0.0))
        raw_docs = result.get("context", [])

        # Format retrieved chunks for the response
        formatted_chunks: List[RetrievedChunk] = []
        for doc in raw_docs:
            formatted_chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=Path(doc.metadata.get("source", "Ebook-Agentic-AI.pdf")).name,
                    page=doc.metadata.get("page", 1)
                )
            )

        return ChatResponse(
            question=request.question,
            answer=answer,
            confidence_score=score,
            retrieved_chunks=formatted_chunks
        )

    except Exception as e:
        # Secure error handling: do NOT leak credentials or raw traces in API responses
        error_type = type(e).__name__
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the RAG request: {error_type}"
        )


if __name__ == "__main__":
    import uvicorn
    # Allow running directly with: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
