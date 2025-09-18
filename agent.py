from loader import load_pdf
from vector_store import VectorStore
from retriever import build_qa_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from typing import List, Callable, Dict, Tuple
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
        self.sections: List[Tuple[str, str, Dict]] = []  # (heading, content, metadata)

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        # Load PDF pages
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF.")
        self.documents = docs

        # Map numbered questions dynamically
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Detect sections dynamically
        self._detect_sections(docs)

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

    def _detect_sections(self, docs: List[Document]):
        """
        Dynamically detect headings and their content in the document.
        Stores them in self.sections as tuples: (heading, content, metadata)
        """
        heading_pattern = re.compile(
            r'^(?P<heading>[A-Z][A-Z0-9\s\-\.:]{2,100})$', re.MULTILINE
        )

        for doc in docs:
            lines = doc.page_content.splitlines()
            current_heading = None
            buffer = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                if heading_pattern.match(stripped):
                    # Save previous section
                    if current_heading and buffer:
                        content = "\n".join(buffer).strip()
                        self.sections.append((current_heading, content, doc.metadata))
                    current_heading = stripped
                    buffer = []
                else:
                    buffer.append(stripped)

            # Add last section
            if current_heading and buffer:
                content = "\n".join(buffer).strip()
                self.sections.append((current_heading, content, doc.metadata))

    def _match_section(self, query: str) -> Document | None:
        """
        Match user query to the most relevant section using keyword search.
        """
        query_lower = query.strip().lower()

        best_match = None
        for heading, content, metadata in self.sections:
            # Check if query matches heading or appears in section content
            if query_lower in heading.lower() or query_lower in content.lower():
                best_match = Document(
                    page_content=f"{heading}\n{content}",
                    metadata=metadata
                )
                break

        return best_match

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

        # Attempt dynamic section match first
        section_doc = self._match_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # Retrieve candidates (similarity search)
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Include keyword hits across all docs
        keyword_hits = [
            d for d in self.documents if question_text.lower() in d.page_content.lower()
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
