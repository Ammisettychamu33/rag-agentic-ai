import os
import sys
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Loads a PDF document, extracts its text, and splits it into semantic chunks.

    Args:
        pdf_path (str): The file path to the PDF document.
        chunk_size (int): Maximum number of characters in each chunk (default: 1000).
        chunk_overlap (int): Number of overlapping characters between consecutive chunks (default: 200).

    Returns:
        List[Document]: A list of LangChain Document objects containing chunked text and metadata.
    """
    # 1. Validate that the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    print(f"[INFO] Loading PDF from: {pdf_path}")

    # 2. Extract text page-by-page using PyPDFLoader
    # PyPDFLoader parses the PDF and attaches metadata like {'source': pdf_path, 'page': page_num}
    loader = PyPDFLoader(file_path=pdf_path)
    raw_pages = loader.load()
    print(f"[SUCCESS] Successfully loaded {len(raw_pages)} page(s) from the PDF.")

    # 3. Configure the text splitter
    # RecursiveCharacterTextSplitter attempts to split on natural boundaries:
    # ["\n\n", "\n", " ", ""] to keep paragraphs and sentences intact.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # 4. Split the loaded document pages into smaller chunks
    chunks = text_splitter.split_documents(raw_pages)
    print(f"[SUCCESS] Created {len(chunks)} text chunk(s) from {len(raw_pages)} pages.")

    return chunks


def find_pdf_in_data_dir(data_dir: str = "data") -> str:
    """
    Helper function to locate the first PDF file inside the data directory.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return ""
    
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        return ""
    
    return str(pdf_files[0])


if __name__ == "__main__":
    # Force UTF-8 output encoding if supported by stdout
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Define paths relative to the project root
    base_dir = Path(__file__).resolve().parent.parent
    data_folder = base_dir / "data"

    print("==================================================")
    print("      Step 2: PDF Ingestion & Text Chunking       ")
    print("==================================================")

    # Check for PDF files in the data/ directory
    pdf_path = find_pdf_in_data_dir(str(data_folder))

    if not pdf_path:
        print(f"\n[ACTION REQUIRED] No PDF file found in: {data_folder}")
        print("Please copy/paste your Agentic AI eBook PDF into the 'data/' folder and rerun this script.\n")
    else:
        # Load and chunk the PDF
        try:
            chunks = load_and_chunk_pdf(
                pdf_path=pdf_path,
                chunk_size=1000,
                chunk_overlap=200
            )

            print("\n---------------- Sample Chunks Preview ----------------")
            # Display previews of the first 2 chunks
            for idx, chunk in enumerate(chunks[:2]):
                page_num = chunk.metadata.get("page", "Unknown")
                source_file = Path(chunk.metadata.get("source", "")).name
                print(f"\n--- [Chunk #{idx + 1} | Source: {source_file} | Page: {page_num} | Length: {len(chunk.page_content)} chars] ---")
                preview_text = chunk.page_content[:300].strip().replace("\n", " ")
                print(f"Content Preview:\n\"{preview_text}...\"")
            
            print("\n==================================================")
            print("[SUCCESS] Ingestion test completed successfully!")
            print("==================================================")

        except Exception as e:
            print(f"[ERROR] Error during ingestion: {e}")
