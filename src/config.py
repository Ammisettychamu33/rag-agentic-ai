import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
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
