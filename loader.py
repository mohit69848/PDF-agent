# loader.py
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os
import math

def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF file and split it into chunks dynamically based on file size.
    Handles scanned PDFs via OCR if normal extraction fails.
    """

    # Get file size in MB
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    # Dynamically determine chunk size & overlap
    # Larger files → smaller chunks for better retrieval; smaller files → larger chunks
    # Using logarithmic scaling to smoothly adapt
    base_chunk = 1000
    chunk_size = max(300, int(base_chunk / math.log1p(file_size_mb)))
    chunk_overlap = max(30, int(chunk_size * 0.15))  # 15% overlap

    # Load PDF text
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        total_text = "".join([p.page_content for p in pages]).strip()
        if len(total_text) < 50:  # Possibly scanned PDF
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
