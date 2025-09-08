from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from embeddings import get_embedding
from config import CHROMA_DB_PATH
from typing import List, Callable
import os

def sanitize_metadata(meta: dict) -> dict:
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)
    return clean_meta

class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_DB_PATH):
        self.persist_dir = persist_dir
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

        self.vectordb = Chroma.from_documents(
            documents=enriched_docs,
            embedding=get_embedding(),
            persist_directory=self.persist_dir,
        )
        self.vectordb.persist()

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.vectordb:
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=get_embedding(),
            )
        return self.vectordb.similarity_search(query, k=top_k)
