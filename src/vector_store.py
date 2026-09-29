import os
import sys
import time
import hashlib
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env if present
load_dotenv()

# Synchronize Streamlit Cloud secrets into os.environ if available
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]:
            if key in st.secrets and not os.getenv(key):
                val = str(st.secrets[key]).strip()
                if val and not val.startswith("your_"):
                    os.environ[key] = val
except Exception:
    pass

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Constants based on assignment specifications
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
SIMILARITY_METRIC = "cosine"
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")


def validate_api_keys() -> bool:
    """
    Validates that necessary API keys are present in the environment.
    Does not print or expose secret values.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()

    is_openai_valid = bool(openai_key and not openai_key.startswith("your_"))
    is_pinecone_valid = bool(pinecone_key and not pinecone_key.startswith("your_"))

    return is_openai_valid and is_pinecone_valid


def get_embeddings() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI embeddings client.
    Uses 'text-embedding-3-small' with 1536 dimensions.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key or openai_key.startswith("your_"):
        raise ValueError(
            "Missing or invalid OPENAI_API_KEY. Please configure OPENAI_API_KEY in your `.env` file (local) or Streamlit Secrets (Cloud)."
        )

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=openai_key
    )


def init_pinecone_index(
    index_name: str = DEFAULT_INDEX_NAME,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = SIMILARITY_METRIC
) -> Pinecone:
    """
    Initializes the Pinecone client and creates the index if it does not already exist.

    Args:
        index_name (str): Name of the Pinecone index (default: agentic-ai-index).
        dimension (int): Vector dimension (1536 for text-embedding-3-small).
        metric (str): Similarity metric ('cosine').

    Returns:
        Pinecone: The authenticated Pinecone client instance.
    """
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_key or pinecone_key.startswith("your_"):
        raise ValueError(
            "Missing or invalid PINECONE_API_KEY. Please configure PINECONE_API_KEY in your `.env` file (local) or Streamlit Secrets (Cloud)."
        )

    pc = Pinecone(api_key=pinecone_key)

    # List existing indexes
    index_names = [idx.name for idx in pc.list_indexes()]

    if index_name not in index_names:
        print(f"[INFO] Creating Pinecone index '{index_name}' (dimension={dimension}, metric={metric})...", flush=True)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(index_name).status.ready:
            time.sleep(1)
        print(f"[SUCCESS] Pinecone index '{index_name}' created successfully and is ready.", flush=True)
    else:
        desc = pc.describe_index(index_name)
        print(f"[INFO] Pinecone index '{index_name}' exists and is ready.", flush=True)
        print(f"[INFO] Index details: Dimension={desc.dimension}, Metric={desc.metric}, Status={desc.status['state']}", flush=True)

    return pc


def generate_deterministic_chunk_id(chunk: Document, index_idx: int) -> str:
    """
    Generates a deterministic vector ID for each chunk based on source, page, index, and content hash.
    This prevents creating duplicate vectors if the indexing pipeline is run multiple times.
    """
    source_name = Path(chunk.metadata.get("source", "doc")).stem
    page_num = chunk.metadata.get("page", 0)
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{source_name}_p{page_num}_c{index_idx}_{content_hash}"


def index_documents(
    chunks: List[Document],
    index_name: str = DEFAULT_INDEX_NAME
) -> PineconeVectorStore:
    """
    Embeds document chunks using OpenAI and upserts them into the Pinecone vector index.

    Args:
        chunks (List[Document]): List of chunked Document objects.
        index_name (str): Target Pinecone index name.

    Returns:
        PineconeVectorStore: The LangChain vector store connected to the Pinecone index.
    """
    # 1. Initialize Pinecone index
    init_pinecone_index(index_name=index_name)

    # 2. Initialize Embeddings
    embeddings = get_embeddings()

    # 3. Generate deterministic IDs for safe, idempotency-preserving indexing
    chunk_ids = [generate_deterministic_chunk_id(chunk, idx) for idx, chunk in enumerate(chunks)]

    print(f"[INFO] Embedding and indexing {len(chunks)} chunk(s) into Pinecone index '{index_name}'...", flush=True)

    # 4. Upsert documents into Pinecone using LangChain PineconeVectorStore
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        ids=chunk_ids
    )

    print(f"[SUCCESS] Successfully indexed {len(chunks)} chunk(s) into Pinecone.", flush=True)
    return vector_store


def get_vector_store(index_name: str = DEFAULT_INDEX_NAME) -> PineconeVectorStore:
    """
    Returns an existing PineconeVectorStore instance for querying/retrieval.
    """
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )


def get_retriever(
    k: int = 4,
    index_name: str = DEFAULT_INDEX_NAME
):
    """
    Returns a LangChain retriever configured to retrieve top-k similar chunks.
    """
    vector_store = get_vector_store(index_name=index_name)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


if __name__ == "__main__":
    # Force UTF-8 stdout encoding
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================", flush=True)
    print("  Step 3: OpenAI Embeddings & Pinecone Indexing   ", flush=True)
    print("==================================================", flush=True)

    # 1. Verify Credentials
    if not validate_api_keys():
        print("\n[ACTION REQUIRED] API keys are missing or not configured in `.env`.", flush=True)
        print("Please ensure your `.env` file contains valid API keys.\n", flush=True)
        print("==================================================", flush=True)
        print("[STATUS] Paused before making API calls.", flush=True)
        print("==================================================", flush=True)
        sys.exit(0)

    # 2. Locate and load the PDF chunks using src.ingestion
    base_dir = Path(__file__).resolve().parent.parent
    data_folder = base_dir / "data"

    from src.ingestion import load_and_chunk_pdf, find_pdf_in_data_dir

    pdf_path = find_pdf_in_data_dir(str(data_folder))
    if not pdf_path:
        print(f"[ERROR] No PDF file found in: {data_folder}", flush=True)
        sys.exit(1)

    print(f"[INFO] Ingesting PDF from: {pdf_path}", flush=True)
    chunks = load_and_chunk_pdf(pdf_path=pdf_path, chunk_size=1000, chunk_overlap=200)

    # 3. Index the chunks into Pinecone
    try:
        vector_store = index_documents(chunks=chunks, index_name=DEFAULT_INDEX_NAME)

        # 4. Verify Index Details & Vector Count
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_client = pc.Index(DEFAULT_INDEX_NAME)
        time.sleep(3)  # pause for index statistics to settle
        stats = index_client.describe_index_stats()

        print("\n---------------- Pinecone Index Stats ----------------", flush=True)
        print(f"Index Name:        {DEFAULT_INDEX_NAME}", flush=True)
        print(f"Dimension:         {EMBEDDING_DIMENSION}", flush=True)
        print(f"Similarity Metric: {SIMILARITY_METRIC}", flush=True)
        print(f"Total Vectors:     {stats.total_vector_count}", flush=True)

        # 5. Perform a Test Similarity Search
        test_query = "What is Agentic AI and how does it work?"
        print(f"\n---------------- Testing Similarity Search ----------------", flush=True)
        print(f"Query: \"{test_query}\"", flush=True)

        # Query with score
        results_with_scores = vector_store.similarity_search_with_relevance_scores(test_query, k=2)

        for rank, (doc, score) in enumerate(results_with_scores, 1):
            source_file = Path(doc.metadata.get("source", "Unknown")).name
            page_num = doc.metadata.get("page", "Unknown")
            print(f"\n--- Result #{rank} | Relevance Score: {score:.4f} | Page: {page_num} | Source: {source_file} ---", flush=True)
            snippet = doc.page_content[:300].strip().replace("\n", " ")
            print(f"Content Preview:\n\"{snippet}...\"", flush=True)

        print("\n==================================================", flush=True)
        print("[SUCCESS] Step 3: Embeddings & Vector Store Verified!", flush=True)
        print("==================================================", flush=True)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during vector store operation: {e}", flush=True)
