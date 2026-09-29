# Agentic AI RAG Assistant

A production-grade, strictly-grounded Retrieval-Augmented Generation (RAG) AI chatbot built with **Python**, **LangGraph**, **Pinecone**, **OpenAI**, **FastAPI**, and **Streamlit**. The system is engineered to answer complex domain queries strictly using content from the **Agentic AI Executive Guide eBook**, enforcing factual grounding and preventing hallucinations.

---

## 1. Project Overview

The **Agentic AI RAG Assistant** provides enterprise-grade question answering over proprietary technical literature. It parses, chunks, and indexes the *Agentic AI eBook*, embeds the content into high-dimensional vector representations, and orchestrates an autonomous retrieval-generation pipeline using a stateful LangGraph workflow.

Key capabilities:
- **Strict Grounding**: Answers are strictly constrained to retrieved context chunks from the eBook.
- **Hallucination Prevention**: Explicitly refuses out-of-domain queries (e.g., general trivia, unrelated events).
- **Confidence Scoring**: Computes mathematical cosine similarity metrics for every retrieval step.
- **Transparent Citations**: Every response returns the exact eBook source pages and text snippets used.
- **Multi-Interface Access**: Accessible via a REST API (FastAPI) and an interactive web interface (Streamlit).

---

## 2. Problem Statement

Standard Large Language Models (LLMs) suffer from:
1. **Knowledge Cutoffs & Hallucinations**: Fabricating answers or extrapolating beyond verified company facts.
2. **Context Dilution**: Inability to process entire hundred-page technical documents in raw prompts without high latency, cost, and the *"lost-in-the-middle"* phenomenon.
3. **Lack of Auditability**: Difficulty in verifying where a model obtained a particular factual claim.

This project solves these challenges by implementing an end-to-end RAG architecture with deterministic chunk tracking, Pinecone vector search, and a stateful LangGraph decision workflow.

---

## 3. RAG Architecture

```
                                  +-----------------------+
                                  | Ebook-Agentic-AI.pdf  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |     PyPDFLoader       |
                                  +-----------+-----------+
                                              |
                                              v
                        +-------------------------------------------+
                        |   RecursiveCharacterTextSplitter          |
                        |   (chunk_size=1000, chunk_overlap=200)    |
                        +---------------------+---------------------+
                                              |
                                              v
                                  +-----------------------+
                                  | OpenAI Embeddings     |
                                  | text-embedding-3-small|
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Pinecone Vector Store |
                                  | (agentic-ai-index)    |
                                  +-----------+-----------+
                                              |
==============================================|==============================================
                         LANGGRAPH WORKFLOW PIPELINE
==============================================|==============================================
                                              v
+------------------+         +-------------------------------+         +---------------------+         +----------------+
|      START       |  --->   |         retrieve_node         |  --->   |    generate_node    |  --->   |      END       |
| (Receive Query)  |         | (Pinecone Top-K Vector Search)|         | (Grounded LLM Chain)|         | (Emit Output)  |
+------------------+         +-------------------------------+         +---------------------+         +----------------+
                                                                                  |
                                              +-----------------------------------+
                                              |
                                              v
                             +---------------------------------+
                             |     Delivery Interfaces         |
                             |  - FastAPI (POST /chat)         |
                             |  - Streamlit Dashboard          |
                             +---------------------------------+
```

---

## 4. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ (tested on Python 3.13) | Core runtime environment |
| **Orchestration** | LangGraph (`>=0.1.0`) | Stateful workflow & state-machine execution |
| **LLM Framework** | LangChain (`>=0.2.0`, `langchain-core`, `langchain-openai`) | Document loaders, prompt templates, and model wrappers |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional semantic dense vectors |
| **Vector Database**| Pinecone (`pinecone>=3.0.0`, `langchain-pinecone`) | Cloud-native serverless vector indexing & cosine search |
| **PDF Ingestion** | PyPDF (`pypdf>=4.0.0`, `fonttools`) | Robust PDF extraction and page metadata preservation |
| **Backend API** | FastAPI (`fastapi>=0.110.0`, `uvicorn>=0.28.0`) | High-performance asynchronous REST API |
| **Data Validation**| Pydantic v2 (`pydantic>=2.7.0`) | Schema enforcement for requests & responses |
| **Frontend UI** | Streamlit (`streamlit>=1.32.0`) | Interactive web application for evaluation & querying |
| **Testing** | Pytest (`pytest>=8.0.0`) | Automated unit, schema, and grounding test suites |

---

## 5. Project Structure

```text
rag-agentic-ai/
├── data/
│   ├── .gitkeep
│   └── Ebook-Agentic-AI.pdf        # Agentic AI eBook source material (60 pages)
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Centralized configuration and constants
│   ├── ingestion.py                # PDF loader and text chunker implementation
│   ├── vector_store.py             # Pinecone vector store and embedding manager
│   └── graph.py                    # LangGraph state machine and grounded RAG workflow
├── tests/
│   ├── test_api.py                 # FastAPI route, schema, and validation tests
│   └── test_grounding.py           # Grounding verification and benchmark query tests
├── app.py                          # FastAPI application backend entrypoint
├── streamlit_app.py                # Streamlit frontend dashboard
├── requirements.txt                # Production and development dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules (prevents secret leaks)
└── README.md                       # Complete project documentation
```

---

## 6. Setup Instructions

### Prerequisites
- Python 3.10 or higher (Python 3.13 recommended)
- Git installed on your system
- An active OpenAI API Key & Pinecone API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rag-agentic-ai.git
   cd rag-agentic-ai
   ```

2. **Create and activate a virtual environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. Environment Variables

Copy `.env.example` to create your local `.env` configuration:

```bash
cp .env.example .env
```

Populate the `.env` file with your credentials:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-index
```

> [!NOTE]
> `.env` is explicitly declared in `.gitignore` to guarantee API keys are never committed to version control.

---

## 8. PDF Ingestion Process

PDF loading and text segmentation are implemented in [`src/ingestion.py`](src/ingestion.py):
1. **Loading**: `PyPDFLoader` iterates through all 60 pages of `data/Ebook-Agentic-AI.pdf`, extracting the text while attaching metadata: `{"source": "Ebook-Agentic-AI.pdf", "page": <page_num>}`.
2. **Chunking**: `RecursiveCharacterTextSplitter` segments the document using character hierarchy `["\n\n", "\n", " ", ""]` to preserve sentence semantics:
   - **`chunk_size = 1000`**: Balances semantic richness with token efficiency (approx. 150–250 words per chunk).
   - **`chunk_overlap = 200`**: Retains 200 characters across chunk boundaries to eliminate context loss.
3. **Output**: Produces exactly **119 chunks** from the 60-page eBook.

To test ingestion independently:
```bash
python src/ingestion.py
```

---

## 9. Embedding & Pinecone Vector Store Process

Implemented in [`src/vector_store.py`](src/vector_store.py):
- **Model**: OpenAI `text-embedding-3-small` (1536 output dimensions).
- **Index Provisioning**: Automatically provisions or connects to Pinecone Serverless index `agentic-ai-index` (`cloud="aws"`, `region="us-east-1"`, `metric="cosine"`).
- **Deterministic IDs**: Generates stable chunk IDs (`source_p{page}_c{index}_{hash}`) to prevent duplicate vectors upon re-indexing.
- **Payload**: Stores raw text chunk content alongside `source` and `page` metadata for downstream citation.

To execute indexing:
```bash
python src/vector_store.py
```

---

## 10. LangGraph Workflow

The RAG workflow in [`src/graph.py`](src/graph.py) uses LangGraph's `StateGraph`:

```python
class AgentState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    score: float
```

### Graph Execution Cycle:
1. **`START -> retrieve_node`**:
   - Queries Pinecone for top-$k$ ($k=4$) matching chunks using cosine similarity.
   - Computes the average cosine relevance score.
2. **`retrieve_node -> generate_node`**:
   - Combines and annotates chunks with page citations.
   - If no relevant context is found, it returns the standard refusal message.
   - Otherwise, invokes `ChatOpenAI(model="gpt-4o-mini", temperature=0.0)` with strict grounding system instructions.
3. **`generate_node -> END`**:
   - Emits the final `AgentState` containing the grounded answer, confidence score, and context references.

---

## 11. FastAPI API Documentation

Start the backend API server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger API docs are available at: `http://localhost:8000/docs`

### Endpoints

#### 1. `GET /health`
Returns system health status.
```json
{
  "status": "healthy",
  "service": "rag-agentic-ai-api",
  "version": "1.0.0"
}
```

#### 2. `POST /chat`
Submits a query to the LangGraph pipeline.

**Request Body**:
```json
{
  "question": "What is Agentic AI according to the eBook?"
}
```

**Response Body (200 OK)**:
```json
{
  "question": "What is Agentic AI according to the eBook?",
  "answer": "According to the eBook, Agentic AI represents a paradigm shift where AI systems possess autonomy, goal-oriented behavior, and decision-making capabilities without constant human intervention (Page 5).",
  "confidence_score": 0.8850,
  "retrieved_chunks": [
    {
      "content": "Agentic AI represents a paradigm shift where AI systems possess autonomy...",
      "source": "Ebook-Agentic-AI.pdf",
      "page": 5
    }
  ]
}
```

---

## 12. Streamlit UI Usage & Cloud Deployment

### Local Execution
Launch the frontend dashboard locally:
```bash
streamlit run streamlit_app.py
```

### Streamlit Community Cloud Deployment
1. Log in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
2. Click **New app** and select the repository: `Ammisettychamu33/rag-agentic-ai`.
3. Set **Main file path**: `streamlit_app.py`.
4. Expand **Advanced settings $\rightarrow$ Secrets** and paste your API keys in TOML format:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   PINECONE_API_KEY = "your_pinecone_api_key_here"
   PINECONE_INDEX_NAME = "agentic-ai-index"
   ```
5. Click **Deploy!** The application seamlessly executes the LangGraph RAG workflow in-process.

### Key Features:
- **Dual Pipeline Mode**: Supports both Direct In-Process LangGraph execution (recommended for Streamlit Cloud) and external FastAPI REST API client mode.
- **Benchmark Selector**: Quick-load predefined benchmark questions.
- **Response Cards**: Displays grounded answers, confidence score metrics, and expandable context chunks with page numbers.

---

## 13. Grounding & Hallucination Prevention

To ensure answers are strictly factual and audit-compliant:
1. **Strict System Prompt**: The LLM is explicitly instructed that it has no prior training knowledge and must refuse questions outside the eBook text.
2. **Deterministic Fallback**: If retrieved chunks are empty, the pipeline immediately returns:
   > *"I cannot answer this question based on the provided Agentic AI eBook as the information is not present in the text."*
3. **Confidence Scoring**: Cosine similarity scores quantify the semantic relevance of retrieved context.

---

## 14. Benchmark Queries & Evaluation

| # | Benchmark Query | Expected Grounded Behavior |
| :--- | :--- | :--- |
| **1** | *What is Agentic AI according to the eBook?* | Cites autonomy, goal-oriented execution, and decision-making (Pages 5-6). |
| **2** | *How do AI agents differ from traditional automation systems?* | Contrasts deterministic rule-based scripts with adaptive context reasoning (Page 12). |
| **3** | *What are core components of Agentic Architecture?* | Cites Perception, Planning & Reasoning, Memory, and Tool Execution (Page 18). |
| **4** | *What role does memory play in Agentic AI workflows?* | Explains conversational continuity, state preservation, and reflection (Page 24). |
| **5** | *Who won 2022 FIFA World Cup?* | **Strict Refusal**: Rejects the query since sports trivia is not in the eBook. |

---

## 15. Testing

The project includes unit, schema, and grounding tests using `pytest`:

Run all tests:
```bash
pytest tests/ -v
```

### Test Coverage (17 Passing Tests):
- **`tests/test_api.py`** (7 Tests):
  - Health check endpoint (`GET /health`)
  - Root endpoint (`GET /`)
  - Empty and whitespace question validation (`422 Unprocessable Entity`)
  - Missing payload validation
  - End-to-end response schema structure
  - OpenAPI 3.x schema generation
- **`tests/test_grounding.py`** (10 Tests):
  - Context citation formatting
  - System prompt grounding rules
  - In-domain benchmark query tests (Queries 1–4)
  - Out-of-domain FIFA World Cup refusal test (Query 5)
  - Direct node execution and error isolation

---

## 16. How to Run the Complete System

1. **Start the FastAPI Backend**:
   ```bash
   uvicorn app:app --port 8000 --reload
   ```

2. **In a second terminal, launch the Streamlit Frontend**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open `http://localhost:8501` in your browser.

---

## 17. Limitations & Transparency

- **API Quota Dependency**: Live vector generation and LLM completions depend on an active OpenAI billing balance. When credits are unavailable, the offline test suite and mock validations provide full functional verification.
- **Single Document Scope**: Optimized specifically for the provided 60-page Agentic AI eBook.

---

## 18. Future Improvements

- **Hybrid Search**: Combine dense vector search with sparse BM25 keyword matching for exact technical terms.
- **Reranking**: Implement a Cohere/Cross-Encoder reranker on the top-10 chunks before LLM generation.
- **Multi-Document Scaling**: Add namespace filtering in Pinecone to support querying across multiple technical whitepapers.
- **Streaming Responses**: Enable SSE (Server-Sent Events) in FastAPI for token-by-token streaming in the Streamlit UI.

---

## 19. License

Developed for the Appening AI Engineering Assignment.
