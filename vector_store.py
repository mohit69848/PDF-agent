# vector_store.py
from langchain.vectorstores import PGVector
from langchain.docstore.document import Document
from embeddings import get_embedding
from config import DATABASE_URL
from typing import List, Callable
from sqlalchemy import create_engine, text
import os

embedding_model = get_embedding()

def sanitize_metadata(meta: dict) -> dict:
    """Ensure all metadata values are JSON-serializable."""
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)
    return clean_meta

class VectorStore:
    def __init__(self):
        self.vectordb = None

    def build(self, docs: List[Document], source_file: str = None, progress_callback: Callable = None):
        """Build vector store with PGVector. Drops old table if dimension mismatch."""
        if not docs:
            raise ValueError("No documents to ingest!")

        enriched_docs = []
        total_docs = len(docs)
        for i, d in enumerate(docs):
            meta = sanitize_metadata(d.metadata.copy())
            if source_file:
                meta["file_name"] = os.path.basename(source_file)
            if "page" in meta and "page_number" not in meta:
                meta["page_number"] = meta["page"]
            enriched_docs.append(Document(page_content=d.page_content, metadata=meta))
            if progress_callback:
                progress_callback(i + 1, total_docs)

        # Connect to DB
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Drop table if exists (to avoid dimension mismatch)
            conn.execute(text("DROP TABLE IF EXISTS pdf_docs"))
            conn.commit()

        # Initialize PGVector and insert documents
        self.vectordb = PGVector.from_documents(
            documents=enriched_docs,
            embedding=get_embedding(),
            collection_name="pdf_docs",
            connection_string=DATABASE_URL
        )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search PGVector for top_k similar documents."""
        if not self.vectordb:
            self.vectordb = PGVector(
                embedding_function=get_embedding(),
                collection_name="pdf_docs",
                connection_string=DATABASE_URL
            )
        return self.vectordb.similarity_search(query, k=top_k)
