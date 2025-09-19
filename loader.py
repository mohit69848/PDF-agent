from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os

def get_dynamic_chunk_params(file_size_mb: float, text_len: int):
    """
    Dynamically compute chunk size and overlap.
    - Larger PDFs → smaller chunks
    - Small PDFs → larger chunks
    """

    # Scale chunk size: 1200 max (small docs) → 300 min (huge docs)
    chunk_size = int(1200 - (file_size_mb * 15) - (text_len / 80000))
    chunk_size = max(300, min(1200, chunk_size))  # clamp between 300–1200

    # Overlap = 10–20% of chunk size
    chunk_overlap = max(50, int(chunk_size * 0.15))

    return chunk_size, chunk_overlap


def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF file and split it into chunks dynamically.
    Uses text length + file size to decide chunking.
    Falls back to OCR for scanned PDFs.
    """
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    # Try normal text extraction
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        total_text = "".join([p.page_content for p in pages]).strip()
        if len(total_text) < 50:
            raise ValueError("Too little text, switching to OCR")
    except Exception:
        loader = UnstructuredPDFLoader(path, strategy="ocr_only")
        pages = loader.load()
        total_text = "".join([p.page_content for p in pages]).strip()

    # Clean pages and add page numbers
    cleaned_docs = []
    for i, page in enumerate(pages):
        text = page.page_content.strip()
        meta = dict(page.metadata)
        meta["page_number"] = i + 1
        if len(text) > 30:  # ignore empty pages
            cleaned_docs.append(Document(page_content=text, metadata=meta))

    # Get chunk params dynamically
    chunk_size, chunk_overlap = get_dynamic_chunk_params(file_size_mb, len(total_text))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(cleaned_docs)
