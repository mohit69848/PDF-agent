from loader import load_pdf
from vector_store import VectorStore
from retriever import build_qa_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from typing import List, Dict, Callable
from reranker import rerank_with_llm
from config import LLM_MODEL, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import re

class PDFQAAgent:
    def __init__(self, persist_dir: str = None):
        self.vector_store = VectorStore()
        self.qa_chain = None

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF to ingest.")
         # Build vector store in PostgreSQL
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)
        # Build retriever and QA chain
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)
        return len(docs)

    def answer(self, question: str, top_k: int = 5) -> Dict:
        if not self.vector_store.vectordb:
            raise ValueError("Vector store is empty. Please ingest a PDF first.")

        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question)

 # Remove duplicates
        seen = set()
        unique_candidates = []
        for d in candidates:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_candidates.append(d)

         # Try exact match first
        exact_answer = None
        question_lower = question.lower()
        for doc in unique_candidates:
            content_lower = doc.page_content.lower()
            if question_lower in content_lower:
                exact_answer = doc.page_content.strip()
                break

        if not exact_answer:
            reranked_docs = rerank_with_llm(question, unique_candidates, top_k=top_k)
            exact_answer = "\n\n".join([f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs])
        else:
            reranked_docs = unique_candidates
# Generate final summary with LLM
        if not reranked_docs:
            summary = exact_answer
        else:
            context = "\n\n".join([f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs])
            llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
            prompt = f"""
You are an AI assistant. The user asked: "{question}"

Relevant text chunks:
{context}

Instructions:
1. Summarize concisely and structure the answer.
2. Use clear bullet points or sections such as:
   - Definition / Concept
   - Advantages / Benefits
   - Applications / Examples
   - Notes / References
3. Avoid repeating information.
4. Only include content from the provided text.
"""
            result = llm.invoke([HumanMessage(content=prompt)])
            summary = result.content if result else exact_answer

        return {
            "answer": summary,
            "sources": reranked_docs
        }
