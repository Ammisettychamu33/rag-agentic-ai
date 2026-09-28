import os
import sys
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# 1. State Definition
# ============================================================================
class AgentState(TypedDict):
    """
    Represents the state passed through the LangGraph RAG workflow.

    Attributes:
        question: The user's input query.
        context: List of retrieved Document chunks from the vector database.
        answer: The generated answer produced by the LLM.
        score: The retrieval/grounding confidence score (between 0.0 and 1.0).
    """
    question: str
    context: List[Document]
    answer: str
    score: float


# ============================================================================
# 2. Prompt Engineering for Strict Grounding
# ============================================================================
SYSTEM_PROMPT = """You are an expert AI assistant specialized in Agentic AI.
Your task is to answer the user's question STRICTLY and SOLELY based on the provided eBook context.

Guidelines:
1. Base your entire answer ONLY on the information given in the Context below.
2. If the context does NOT contain enough information to answer the question, you MUST respond:
   "I cannot answer this question based on the provided Agentic AI eBook as the information is not present in the text."
3. Do NOT make assumptions, extrapolate, or use external knowledge not in the context.
4. Keep your answer clear, accurate, and concise.
5. When relevant, mention the section or page numbers provided in the context metadata.

Context:
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


# ============================================================================
# 3. Node Implementations
# ============================================================================

def format_context_for_prompt(documents: List[Document]) -> str:
    """
    Formats a list of retrieved Document objects into a clean, annotated text block.
    """
    if not documents:
        return "No relevant context found."

    formatted_blocks = []
    for idx, doc in enumerate(documents, 1):
        source = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted_blocks.append(f"[Chunk #{idx} | Source: {source} | Page: {page}]\n{content}")

    return "\n\n".join(formatted_blocks)


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve Node:
    Takes the user's question from the state, queries the Pinecone vector store,
    and returns the top-k matching document chunks along with average similarity score.
    """
    from src.vector_store import get_vector_store

    question = state.get("question", "").strip()
    if not question:
        return {"context": [], "score": 0.0}

    # Connect to Pinecone vector store
    vector_store = get_vector_store()

    # Retrieve top 4 relevant chunks with similarity/relevance scores
    results_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=4)

    documents = []
    scores = []

    for doc, score in results_with_scores:
        documents.append(doc)
        scores.append(score)

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    return {
        "context": documents,
        "score": round(avg_score, 4)
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate Node:
    Takes the retrieved context and question, formats the grounded prompt,
    and calls the ChatOpenAI model to produce an answer strictly from the eBook.
    """
    question = state.get("question", "")
    context_docs = state.get("context", [])
    retrieval_score = state.get("score", 0.0)

    # If no context was retrieved at all, return explicit fallback
    if not context_docs:
        return {
            "answer": "I cannot answer this question based on the provided Agentic AI eBook as the information is not present in the text.",
            "score": 0.0
        }

    # Format context chunks
    formatted_context = format_context_for_prompt(context_docs)

    # Initialize OpenAI Chat Model (gpt-4o-mini is cost-effective and accurate)
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=openai_key
    )

    # Build the grounded chain
    chain = PROMPT_TEMPLATE | llm

    # Execute generation
    response = chain.invoke({
        "context": formatted_context,
        "question": question
    })

    return {
        "answer": response.content,
        "score": retrieval_score
    }


# ============================================================================
# 4. LangGraph Workflow Graph Construction
# ============================================================================

def create_rag_graph():
    """
    Constructs and compiles the StateGraph workflow for the RAG pipeline.

    Workflow topology:
      START -> retrieve -> generate -> END
    """
    # 1. Initialize StateGraph with AgentState
    workflow = StateGraph(AgentState)

    # 2. Add Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 3. Add Edges (linear execution)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # 4. Compile workflow
    app = workflow.compile()
    return app


# Pre-compiled application instance for reuse
rag_app = create_rag_graph()


# ============================================================================
# 5. Local Static Validation (No Live API Calls)
# ============================================================================
if __name__ == "__main__":
    # Force UTF-8 stdout encoding
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================")
    print("      Step 4: LangGraph Workflow Verification     ")
    print("==================================================")

    # 1. Verify compilation and node registration
    print("[INFO] Checking StateGraph compilation...")
    app = create_rag_graph()
    
    # 2. Inspect graph structure
    nodes = list(app.get_graph().nodes.keys())
    edges = app.get_graph().edges

    print(f"[SUCCESS] StateGraph compiled successfully!")
    print(f"[INFO] Registered Nodes: {nodes}")
    print(f"[INFO] Registered Edges: {edges}")

    # 3. Verify AgentState schema
    state_keys = list(AgentState.__annotations__.keys())
    print(f"[INFO] AgentState Schema Attributes: {state_keys}")

    # 4. Verify System Prompt Template
    print("\n---------------- System Prompt Configuration ----------------")
    print(f"System Message Preview:\n{SYSTEM_PROMPT[:320]}...\n")

    print("==================================================")
    print("[SUCCESS] LangGraph RAG Workflow statically validated!")
    print("[NOTE] No live API calls were made during validation.")
    print("==================================================")
