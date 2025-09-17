from loader import load_pdf
from vector_store import VectorStore
from retriever import build_qa_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from typing import List, Callable, Dict
from reranker import rerank_with_llm
from config import LLM_MODEL, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import re

class PDFQAAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.qa_chain = None
        self.question_map: Dict[int, str] = {}

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        """Ingest PDF, split, map questions, and build vector store."""
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF to ingest.")

        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)

        return len(docs)

    def _map_questions(self, text: str):
        """Map numbered questions in PDF dynamically by parsing text patterns."""
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_exact_section(self, documents: List[Document], query: str) -> Document | None:
        """
        Find the exact section that contains the query.
        Looks for paragraphs or blocks containing the query text exactly.
        """
        query_lower = query.strip().lower()

        for doc in documents:
            text = doc.page_content
            # Split into paragraphs by double newline or lines separated by punctuation
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                if query_lower in para.lower():
                    return Document(page_content=para.strip(), metadata=doc.metadata)
        return None

    def answer(self, user_input: str, top_k: int = 5):
        """Answer the question using exact match or fallback to LLM reranking."""
        question_text = user_input.strip()

        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store is empty. Please ingest a PDF first.")

        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # First attempt exact match section search
        exact_doc = self.find_exact_section(candidates, question_text)
        if exact_doc:
            return {"answer": exact_doc.page_content.strip(), "sources": [exact_doc]}

        # Fallback to LLM rerank if exact not found
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join([f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs])
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

        prompt = f"""
You are an AI assistant. The user asked: "{question_text}"

Relevant text chunks:
{context}

Instructions:
1. Summarize concisely and structure the answer.
2. Use clear bullet points or sections.
3. Avoid repeating information.
4. Only include content from the provided text.
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
