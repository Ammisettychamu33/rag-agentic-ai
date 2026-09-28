import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# ============================================================================
# Helper Functions
# ============================================================================

def check_backend_health(url: str) -> bool:
    """
    Checks whether the FastAPI backend is running and healthy.
    """
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


def query_rag_api(question: str, url: str) -> dict:
    """
    Sends a query to the FastAPI /chat endpoint.
    """
    endpoint = f"{url}/chat"
    payload = {"question": question}
    response = requests.post(endpoint, json=payload, timeout=60)
    return response.json()


# ============================================================================
# Sidebar Configuration
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_endpoint = st.text_input("FastAPI Backend URL", value=API_URL)
    
    # Check backend health
    is_healthy = check_backend_health(api_endpoint)
    if is_healthy:
        st.success("🟢 Backend API: Online")
    else:
        st.warning("🟡 Backend API: Offline (Run `uvicorn app:app --reload`)")

    st.markdown("---")
    st.subheader("📚 Source Material")
    st.info("**Document**: Agentic AI eBook (`Ebook-Agentic-AI.pdf`)\n\n**Pages**: 60 Pages (119 Chunks)")

    st.markdown("---")
    st.subheader("🎯 Benchmark Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are core components of Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won 2022 FIFA World Cup? (Refusal Test)"
    ]

    selected_sample = st.selectbox("Load a sample question:", ["-- Select a benchmark query --"] + sample_queries)


# ============================================================================
# Main Application UI
# ============================================================================
st.title("🤖 Agentic AI RAG Assistant")
st.markdown(
    "Ask questions about **Agentic AI architecture, autonomous systems, and enterprise design patterns**. "
    "Answers are **strictly grounded** in the provided Agentic AI eBook."
)

st.markdown("---")

# Pre-populate question input if sample was selected
default_question = ""
if selected_sample and selected_sample != "-- Select a benchmark query --":
    # Strip the "(Refusal Test)" suffix if present
    default_question = selected_sample.replace(" (Refusal Test)", "")

user_question = st.text_area(
    "Enter your question:",
    value=default_question,
    placeholder="e.g., What are the core components of Agentic Architecture?",
    height=100
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)

# ============================================================================
# Query Execution & Response Rendering
# ============================================================================
if ask_button:
    trimmed_query = user_question.strip()

    if not trimmed_query:
        st.error("⚠️ Please enter a valid question before asking.")
    else:
        with st.spinner("🔍 Retrieving context from Pinecone and generating grounded response..."):
            try:
                # 1. Attempt call to FastAPI backend
                if is_healthy:
                    response_data = query_rag_api(trimmed_query, api_endpoint)
                    
                    if "detail" in response_data:
                        st.error(f"❌ Backend Error: {response_data['detail']}")
                    else:
                        answer = response_data.get("answer", "")
                        score = response_data.get("confidence_score", 0.0)
                        chunks = response_data.get("retrieved_chunks", [])

                        # Display Answer
                        st.markdown("### 📝 Answer")
                        st.write(answer)

                        # Display Confidence Score
                        st.markdown("---")
                        st.metric(label="Relevance / Confidence Score", value=f"{score:.4f}")

                        # Display Retrieved Chunks
                        if chunks:
                            st.markdown("### 📑 Retrieved Context Chunks")
                            for idx, chunk in enumerate(chunks, 1):
                                source = chunk.get("source", "Unknown")
                                page = chunk.get("page", "N/A")
                                content = chunk.get("content", "")

                                with st.expander(f"Chunk #{idx} — Source: `{source}` | Page: `{page}`"):
                                    st.write(content)
                        else:
                            st.info("No context chunks were retrieved for this query.")

                else:
                    # Backend offline message
                    st.warning(
                        "⚠️ **FastAPI Backend is currently offline.**\n\n"
                        "To use the full interactive chatbot, start the backend in a separate terminal:\n"
                        "```bash\n"
                        "uvicorn app:app --reload --port 8000\n"
                        "```\n"
                        "*Note: Live OpenAI and Pinecone API calls require active API credits in your `.env` configuration.*"
                    )

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the FastAPI backend. Please make sure `uvicorn app:app --reload` is running.")
            except Exception as e:
                st.error(f"❌ An error occurred: {type(e).__name__}")
