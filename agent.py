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
        self.documents: List[Document] = []
        self.question_map: Dict[int, str] = {}

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        # Load PDF pages
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF.")

        self.documents = docs

        # Map numbered questions dynamically
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Build vector store for fallback
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        # Build QA chain
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)

        return len(docs)

    def _map_questions(self, text: str):
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_section(self, query: str) -> Document | None:
        """
        Dynamically find a section in PDF matching any part of the user query.
        """
        query_lower = query.strip().lower()
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Heading detection: uppercase or capitalized line
                is_heading = (
                    clean_line.isupper()
                    or re.match(r'^[A-Z][A-Za-z\s\-:]{2,50}$', clean_line)
                )

                # Match heading or any keyword in user query
                if is_heading and query_lower in clean_line.lower():
                    section_lines = [clean_line]

                    # Collect until next heading
                    for next_line in lines[i + 1:]:
                        next_clean = next_line.strip()
                        if not next_clean:
                            continue

                        is_next_heading = (
                            next_clean.isupper()
                            and len(next_clean.split()) <= 6
                            and len(next_clean) > 3
                        )

                        if is_next_heading:
                            break

                        section_lines.append(next_clean)

                    return Document(
                        page_content="\n".join(section_lines),
                        metadata=doc.metadata,
                    )

        # Fallback: search for any line containing keywords from query
        for doc in self.documents:
            for line in doc.page_content.splitlines():
                if any(word in line.lower() for word in query_lower.split()):
                    return Document(page_content=line.strip(), metadata=doc.metadata)

        return None

    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        # Handle numeric questions dynamically
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        # Dynamic section matching (no hardcoding)
        section_doc = self.find_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # Retrieve candidates via vector search
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Force include keyword hits
        keyword_hits = [
            d for d in self.documents if any(word in d.page_content.lower() for word in question_text.lower().split())
        ]
        candidates.extend(keyword_hits)

        # Exact text match in candidates
        for doc in candidates:
            if any(word in doc.page_content.lower() for word in question_text.lower().split()):
                return {"answer": doc.page_content.strip(), "sources": [doc]}

        # Rerank with LLM
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join(
            [f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs]
        )
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

        prompt = f"""
You are answering strictly from the document text.

User question: "{question_text}"

Relevant extracted text:
{context}

Instructions:
- Return text VERBATIM from the document (preserve formatting, line breaks, bullets).
- If OCR text, prefix with: "Extracted from image:"
- Do NOT summarize unless text is too long.
- If no relevant text, reply only: "⚠️ No relevant section found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
