from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os

def load_pdf(path: str) -> List[Document]:
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

    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        total_text = "".join([p.page_content for p in pages]).strip()
        if len(total_text) < 50:
            raise ValueError("Too little text, switching to OCR")
    except Exception:
        loader = UnstructuredPDFLoader(path, strategy="ocr_only")
        pages = loader.load()

    cleaned_docs = []
    for i, page in enumerate(pages):
        text = page.page_content.strip()
        meta = dict(page.metadata)
        meta["page_number"] = i + 1
        if len(text) > 30:
            cleaned_docs.append(Document(page_content=text, metadata=meta))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(cleaned_docs)
