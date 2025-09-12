from langchain.vectorstores import PGVector
from langchain.docstore.document import Document
from embeddings import get_embedding
from config import DATABASE_URL
from typing import List, Callable
import os
from sqlalchemy import create_engine, text

embedding_model = get_embedding()
CURRENT_DIM = embedding_model.embedding_size if hasattr(embedding_model, "embedding_size") else 384

def sanitize_metadata(meta: dict) -> dict:
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)
    return clean_meta

class VectorStore:
    def __init__(self, persist_dir: str = None):
        self.vectordb = None

    def build(self, docs: List[Document], source_file: str = None, progress_callback: Callable = None):
        if not docs:
            raise ValueError("No documents to ingest into vector store!")

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

        if not enriched_docs:
            raise ValueError("All documents were filtered out. Nothing to ingest!")

        # Check existing vector table dimension
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Check if pgvector table exists
            res = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_name='pdf_docs'")
            ).fetchone()

            if res:
                # Get the dimension of first vector stored
                try:
                    dim_res = conn.execute(
                        text("SELECT vector FROM pdf_docs LIMIT 1")
                    ).fetchone()
                    if dim_res:
                        existing_dim = len(dim_res[0])
                        if existing_dim != CURRENT_DIM:
                            # Drop table if dimension mismatch
                            conn.execute(text("DROP TABLE IF EXISTS pdf_docs"))
                            conn.commit()
                except Exception:
                    # If anything fails, drop table
                    conn.execute(text("DROP TABLE IF EXISTS pdf_docs"))
                    conn.commit()

        # Initialize PGVector and insert documents
        self.vectordb = PGVector.from_documents(
            documents=enriched_docs,
            embedding=embedding_model,
            collection_name="pdf_docs",
            connection_string=DATABASE_URL
        )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.vectordb:
            self.vectordb = PGVector(
                embedding_function=embedding_model,
                collection_name="pdf_docs",
                connection_string=DATABASE_URL
            )
        return self.vectordb.similarity_search(query, k=top_k)
