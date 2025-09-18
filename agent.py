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
        Load PDF, chunk it, build vector store, and map numbered questions.
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
        Detect numbered questions dynamically and store in question_map.
        """
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_section(self, query: str) -> List[Document]:
        """
        Dynamically find sections matching query keywords across all chunks.
        Works even if headings are not uppercase,
        returning the heading + full section content until the next heading.
        """
        query_keywords = set(query.lower().split())
        matched_docs = []

        for doc in self.documents:
            lines = doc.page_content.splitlines()
            section_lines = []
            inside_section = False

            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Treat any line containing the keyword as a start of a section
                if any(word in clean_line.lower() for word in query_keywords):
                    inside_section = True

                if inside_section:
                    section_lines.append(clean_line)

                    # Detect next heading or empty line as end of section
                    if (
                        i + 1 < len(lines)
                        and lines[i + 1].isupper()
                        and len(lines[i + 1].split()) <= 8
                    ):
                        inside_section = False
                        break

            # Append remaining section if any
            if section_lines:
                matched_docs.append(Document(
                    page_content="\n".join(section_lines),
                    metadata=doc.metadata
                ))

        return matched_docs

    def answer(self, user_input: str, top_k: int = 5):
        """
        Answer a question about the PDF:
        1. Section-aware search
        2. Vector similarity retrieval
        3. Keyword fallback
        4. Rerank with LLM
        """
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

        # 1. Section-aware search
        section_docs = self.find_section(question_text)
        if section_docs:
            return {
                "answer": "\n\n".join(d.page_content for d in section_docs),
                "sources": section_docs
            }

        # 2. Vector similarity search
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # 3. Keyword fallback
        keyword_hits = [
            d for d in self.documents
            if question_text.lower() in d.page_content.lower()
        ]
        candidates.extend(keyword_hits)

        if not candidates:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        # 4. Rerank with LLM
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join(
            [f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs]
        )

        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
        prompt = f"""
You are a strict PDF answering agent.

User question: "{question_text}"

Relevant text:
{context}

Instructions:
- Answer using ONLY this text, return it VERBATIM.
- Preserve formatting, bullets, line breaks.
- If it comes from OCR/image, prefix with "Extracted from image:".
- If nothing matches, return only: ⚠️ No relevant content found.
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
