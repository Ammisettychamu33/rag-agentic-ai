import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

# Synchronize Streamlit Cloud secrets into os.environ if running on Streamlit Cloud
if hasattr(st, "secrets"):
    for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "RAG_API_URL"]:
        try:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass

# Import the pre-compiled LangGraph RAG workflow for direct in-process execution
from src.graph import rag_app

# Page Configuration
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

# ============================================================================
# Helper Functions
# ============================================================================

def check_backend_health(url: str) -> bool:
    """
    Checks whether an external FastAPI backend is running and healthy.
    """
    try:
        import requests
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


def query_rag_api(question: str, url: str) -> dict:
    """
    Sends a query to the external FastAPI /chat endpoint.
    """
    import requests
    endpoint = f"{url}/chat"
    payload = {"question": question}
    response = requests.post(endpoint, json=payload, timeout=60)
    return response.json()


def query_rag_direct(question: str) -> dict:
    """
    Direct in-process execution using the LangGraph RAG workflow.
    Used for standalone deployments on Streamlit Community Cloud.
    """
    initial_state = {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0
    }

    result = rag_app.invoke(initial_state)

    answer = result.get("answer", "")
    score = float(result.get("score", 0.0))
    raw_docs = result.get("context", [])

    formatted_chunks = []
    for doc in raw_docs:
        formatted_chunks.append({
            "content": doc.page_content,
            "source": Path(doc.metadata.get("source", "Ebook-Agentic-AI.pdf")).name,
            "page": doc.metadata.get("page", 1)
        })

    return {
        "question": question,
        "answer": answer,
        "confidence_score": score,
        "retrieved_chunks": formatted_chunks
    }


# ============================================================================
# Sidebar Configuration
# ============================================================================
with st.sidebar:
    st.header("⚙️ Execution Mode")
    
    execution_mode = st.radio(
        "Select Pipeline Mode:",
        ["Direct LangGraph Engine (Recommended for Streamlit Cloud)", "FastAPI REST API"],
        index=0
    )

    api_endpoint = DEFAULT_API_URL
    if execution_mode == "FastAPI REST API":
        api_endpoint = st.text_input("FastAPI Backend URL", value=DEFAULT_API_URL)
        is_healthy = check_backend_health(api_endpoint)
        if is_healthy:
            st.success("🟢 Backend API: Online")
        else:
            st.warning("🟡 Backend API: Offline")
    else:
        st.success("🟢 In-Process LangGraph Engine Active")

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
                # Execute query based on selected mode
                if execution_mode == "FastAPI REST API":
                    if check_backend_health(api_endpoint):
                        response_data = query_rag_api(trimmed_query, api_endpoint)
                    else:
                        st.error("❌ FastAPI backend is unreachable. Start `uvicorn app:app --reload` or switch to Direct LangGraph Engine mode.")
                        response_data = None
                else:
                    # Direct in-process LangGraph execution
                    response_data = query_rag_direct(trimmed_query)

                if response_data:
                    if "detail" in response_data:
                        st.error(f"❌ Processing Error: {response_data['detail']}")
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

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "credit_balance_exhausted" in error_str:
                    st.error(
                        "⚠️ **OpenAI API Quota Limit (429)**: The OpenAI account has no available credits. "
                        "Please add credits at [OpenAI Billing](https://platform.openai.com/settings/organization/billing) or update your key in Streamlit Secrets."
                    )
                elif "PINECONE" in error_str.upper():
                    st.error("⚠️ **Pinecone Connection Error**: Please verify your Pinecone API key and index name in Streamlit Secrets.")
                else:
                    st.error(f"❌ An error occurred during query execution: {type(e).__name__}")
