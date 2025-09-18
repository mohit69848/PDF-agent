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
        """Load PDF, map numbered questions, build vector store & QA chain"""
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
        """Map numbered questions dynamically from PDF text"""
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_section(self, query: str) -> Document | None:
        """
        Dynamically find a section by heading and return all text under it
        until the next heading or large gap.
        """
        query_keywords = set(query.lower().split())

        for doc in self.documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Detect headings dynamically
                is_heading = (
                    clean_line.isupper()
                    or re.match(r'^[A-Z][A-Za-z\s\-:]{2,50}$', clean_line)
                )

                # Check if heading matches any query keyword
                if is_heading and any(word in clean_line.lower() for word in query_keywords):
                    section_lines = [clean_line]

                    # Collect text until next heading
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
        return None

    def answer(self, user_input: str, top_k: int = 5):
        """
        Answer user query dynamically:
        - First try exact section match
        - Then fallback to retriever + reranker
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

        # 1️⃣ Try exact section match
        section_doc = self.find_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # 2️⃣ Retrieve candidates (similarity search)
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # 🔹 Filter candidates by keywords in first 20 words to avoid irrelevant sections
        query_keywords = set(question_text.lower().split())
        filtered_candidates = []
        for d in candidates:
            snippet = " ".join(d.page_content.split()[:20]).lower()
            if any(word in snippet for word in query_keywords):
                filtered_candidates.append(d)

        # Deduplicate
        filtered_candidates = list({id(d): d for d in filtered_candidates}.values())

        if not filtered_candidates:
            return {"answer": "⚠️ No relevant section found in the document.", "sources": []}

        # 3️⃣ Rerank candidates using LLM
        reranked_docs = rerank_with_llm(question_text, filtered_candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join(
            [f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs]
        )
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

        prompt = f"""
Answer strictly from the document text.

User question: "{question_text}"

Relevant extracted text:
{context}

Instructions:
- Return the answer verbatim, preserve formatting.
- Prefix "Extracted from image:" if OCR text.
- Never summarize unless text exceeds max length.
- If no relevant text exists, reply: "⚠️ No relevant section found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
