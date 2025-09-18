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
        """
        Load PDF, build vector store, map numeric questions dynamically.
        """
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF.")

        self.documents = docs

        # Map numbered questions dynamically
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Build vector store for fallback retrieval
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        # Build QA chain
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)

        return len(docs)

    def _map_questions(self, text: str):
        """
        Dynamically map numbered questions from PDF content.
        """
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_section(self, query: str) -> Document | None:
        """
        Robust dynamic section detection for headings (e.g., DISCLAIMER, References)
        - Handles multi-line headings
        - Ignores irrelevant preamble
        - Stops at next heading
        """
        query_clean = query.strip().lower()
        
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            section_started = False
            section_lines = []

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Detect heading (robust for multi-line, colon, dash, uppercase)
                is_heading = (
                    line.isupper() or  # All uppercase headings
                    re.match(r'^[A-Z][A-Za-z\s\-\:]{2,100}$', line)  # e.g., "Disclaimer:", "References - List"
                )

                # Start of section
                if not section_started and is_heading and query_clean in line.lower():
                    section_started = True
                    section_lines.append(line)

                    # Handle multi-line headings
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        # Stop if next line looks like a new heading
                        if next_line.isupper() and len(next_line.split()) <= 6 and len(next_line) > 3:
                            break
                        # Likely part of multi-line heading
                        if re.match(r'^[A-Z][A-Za-z\s\-\:]{2,100}$', next_line):
                            section_lines.append(next_line)
                            j += 1
                        else:
                            break
                    i = j
                    continue

                # Collect section content until next heading
                if section_started:
                    if is_heading and len(line.split()) <= 6 and len(line) > 3:
                        break
                    section_lines.append(line)

                i += 1

            if section_lines:
                return Document(
                    page_content="\n".join(section_lines),
                    metadata=doc.metadata
                )

        return None

    def answer(self, user_input: str, top_k: int = 5):
        """
        Answer PDF questions:
        - Maps numeric questions
        - Checks exact section match
        - Retrieves similar content from vector store
        - Reranks with LLM if needed
        """
        question_text = user_input.strip().lower()

        # Map "tell me about X" -> "X" dynamically
        tell_match = re.match(r'(tell me about|what is|explain)\s+(.*)', question_text)
        if tell_match:
            question_text = tell_match.group(2)

        # Handle numeric questions like "5 question"
        numeric_match = re.match(r'(\d+)\s*question', question_text)
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        # Attempt exact section match
        section_doc = self.find_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # Retrieve candidates (similarity search)
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Force include keyword matches across all docs
        keyword_hits = [
            d for d in self.documents
            if question_text.lower() in d.page_content.lower()
        ]
        candidates.extend(keyword_hits)

        # Try exact text match in candidates
        for doc in candidates:
            if question_text.lower() in doc.page_content.lower():
                return {"answer": doc.page_content.strip(), "sources": [doc]}

        # Rerank candidates with LLM
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join(
            [f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs]
        )
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

        # Stricter prompt
        prompt = f"""
You are answering strictly from the document text.

User question: "{question_text}"

Relevant extracted text:
{context}

Instructions:
- If the extracted text contains the answer, return it VERBATIM (preserve formatting, line breaks, bullet points).
- If it’s from OCR, prefix with: "Extracted from image:"
- Never summarize or paraphrase unless the text is too long to fit.
- If no relevant text exists, reply only: "⚠️ No relevant section found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
