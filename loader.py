from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List


def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF, fallback to OCR if text extraction fails or content is too small.
    Splits the text into chunks and adds metadata including page numbers.
    """
    try:
        # Try loading with PyMuPDFLoader first
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        total_text = "".join([p.page_content for p in pages]).strip()
        if len(total_text) < 50:
            raise ValueError("Too little text, switching to OCR")
    except Exception:
        # Fallback to OCR with UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(path, strategy="ocr_only")
        pages = loader.load()

    cleaned_docs = []
    for i, page in enumerate(pages):
        text = page.page_content.strip()
        meta = dict(page.metadata)
        meta["page_number"] = i + 1  # Ensure page number is present
        if len(text) > 30:  # Only keep pages with significant content
            cleaned_docs.append(Document(page_content=text, metadata=meta))

    # Use RecursiveCharacterTextSplitter to split long pages into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    # Split documents and return
    return splitter.split_documents(cleaned_docs)
