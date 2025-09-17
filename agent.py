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
        self.question_map: Dict[int, str] = {}  # Maps question numbers to text

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        # Load PDF
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF to ingest.")

        # Map numbered questions
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Build vector store
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        # Build retriever and QA chain
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)

        return len(docs)

    def _map_questions(self, text: str):
        """Extract numbered questions like 1. Question text"""
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_reference_section(self, documents: List[Document]) -> Document | None:
        """Detect reference sections dynamically based on patterns"""
        for doc in documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                is_upper = clean_line.isupper() and len(clean_line) > 4
                is_end = i >= len(lines) - 5
                next_lines = lines[i+1:i+4] if i+4 <= len(lines) else lines[i+1:]
                looks_like_citation = any(
                    re.match(r'^\s*[\d\-\*\.\)]', nl.strip()) or "http" in nl for nl in next_lines
                )
                punct_count = sum(1 for c in clean_line if c in ".,;:[]()")
                is_punctuated = punct_count >= 2

                if is_upper and (looks_like_citation or is_end or is_punctuated):
                    return doc
        return None

    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        # Check for numeric question format like "5 question"
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store is empty. Please ingest a PDF first.")

        # Retrieve relevant documents
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Check for reference section first
        reference_doc = self.find_reference_section(candidates)
        if reference_doc:
            return {
                "answer": reference_doc.page_content.strip(),
                "sources": [reference_doc]
            }

        # Remove duplicates
        seen = set()
        unique_candidates = []
        for d in candidates:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_candidates.append(d)

        # Try exact match
        exact_answer = None
        q_lower = question_text.lower()
        for doc in unique_candidates:
            if q_lower in doc.page_content.lower():
                exact_answer = doc.page_content.strip()
                break

        if exact_answer:
            return {
                "answer": exact_answer,
                "sources": unique_candidates
            }

        # If no exact match, rerank
        reranked_docs = rerank_with_llm(question_text, unique_candidates, top_k=top_k)
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
2. Use clear bullet points or sections such as:
   - Definition / Concept
   - Advantages / Benefits
   - Applications / Examples
   - Notes / References
3. Avoid repeating information.
4. Only include content from the provided text.
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {
            "answer": summary,
            "sources": reranked_docs
        }
