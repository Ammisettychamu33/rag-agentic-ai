import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Synchronize Streamlit Cloud secrets into os.environ if running in Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "RAG_API_URL"]:
            try:
                if key in st.secrets and not os.getenv(key):
                    val = str(st.secrets[key]).strip()
                    if val and not val.startswith("your_"):
                        os.environ[key] = val
            except Exception:
                pass
except Exception:
    pass

# Project Paths (Cross-platform compatible)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"

# Ingestion Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding & Vector Database Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
SIMILARITY_METRIC = "cosine"
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# LLM Generation Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
