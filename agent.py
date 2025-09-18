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
        Dynamically find a section in the PDF matching any part of the user query.
        Collects full text under the heading until next heading or large gap.
        """
        query_lower = query.strip().lower()
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                # Detect heading: uppercase or capitalized line, 2-50 chars
                is_heading = (
                    line.isupper() or re.match(r'^[A-Z][A-Za-z\s\-:]{2,50}$', line)
                )

                # Heading matches the query
                if is_heading and query_lower in line.lower():
                    section_lines = []
                    i += 1

                    # Collect all lines until next heading or large empty lines
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if not next_line:
                            i += 1
                            continue

                        next_is_heading = (
                            next_line.isupper() and 2 <= len(next_line.split()) <= 6
                        )

                        if next_is_heading:
                            break

                        section_lines.append(next_line)
                        i += 1

                    # If no lines collected, include heading itself
                    if not section_lines:
                        section_lines.append(line)

                    return Document(
                        page_content="\n".join(section_lines).strip(),
                        metadata=doc.metadata,
                    )

                i += 1

        # Fallback: search for any line containing keywords from query
        for doc in self.documents:
            for line in doc.page_content.splitlines():
                if all(word in line.lower() for word in query_lower.split()):
                    return Document(page_content=line.strip(), metadata=doc.metadata)

        return None

    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        # Handle numeric questions like "5 question"
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        # First, attempt dynamic section match
        section_doc = self.find_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # Retrieve candidates (similarity search)
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Include keyword matches across all docs
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

        # Dynamic prompt with stricter instructions
        prompt = f"""
You are answering strictly from the document text.

User question: "{question_text}"

Relevant extracted text:
{context}

Instructions:
- Return the answer VERBATIM (preserve formatting, line breaks, bullet points).
- Prefix with "Extracted from image:" if text comes from OCR.
- Never summarize or paraphrase unless it exceeds max length.
- If no relevant text exists, reply only: "⚠️ No relevant section found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
