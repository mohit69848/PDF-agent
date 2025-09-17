# loader.py
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os

def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF file and split it into chunks dynamically based on file size.
    Handles scanned PDFs via OCR if normal extraction fails.
    """
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    # Adaptive chunk size
    if file_size_mb <= 5:
        chunk_size, chunk_overlap = 1000, 150
    elif file_size_mb <= 20:
        chunk_size, chunk_overlap = 800, 100
    elif file_size_mb <= 50:
        chunk_size, chunk_overlap = 600, 80
    else:
        chunk_size, chunk_overlap = 400, 50

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

    # Clean pages and add page numbers
    cleaned_docs = []
    for i, page in enumerate(pages):
        text = page.page_content.strip()
        meta = dict(page.metadata)
        meta["page_number"] = i + 1
        if len(text) > 30:  # ignore empty pages
            cleaned_docs.append(Document(page_content=text, metadata=meta))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(cleaned_docs)
