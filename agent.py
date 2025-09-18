from loader import load_pdf
from vector_store import VectorStore
from retriever import build_qa_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from typing import List, Callable, Dict
from reranker import rerank_with_llm
from config import LLM_MODEL, GOOGLE_API_KEY, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, util
import re

class PDFQAAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.qa_chain = None
        self.documents: List[Document] = []
        self.question_map: Dict[int, str] = {}
        self.section_embedder = None

        # Initialize local embedding model if provider is local
        if EMBEDDING_PROVIDER == "local" and LOCAL_EMBEDDING_MODEL:
            self.section_embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    # -----------------------------
    # PDF ingestion
    # -----------------------------
    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF.")

        self.documents = docs

        # Map numbered questions
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Build vector store for QA fallback
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        # Build QA chain
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 8, "score_threshold": 0.3},
        )
        self.qa_chain = build_qa_chain(retriever)
        return len(docs)

    # -----------------------------
    # Numbered question mapping
    # -----------------------------
    def _map_questions(self, text: str):
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    # -----------------------------
    # Section-aware search (keyword + semantic)
    # -----------------------------
    def find_section(self, query: str, top_k: int = 3) -> List[Document]:
        query_keywords = set(query.lower().split())
        matched_docs = []

        # Keyword-based section search
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            section_lines = []
            inside_section = False

            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                if any(word in clean_line.lower() for word in query_keywords):
                    inside_section = True

                if inside_section:
                    section_lines.append(clean_line)
                    if i + 1 < len(lines) and lines[i + 1].isupper() and len(lines[i + 1].split()) <= 8:
                        inside_section = False
                        break

            if section_lines:
                matched_docs.append(Document(
                    page_content="\n".join(section_lines),
                    metadata=doc.metadata
                ))

        # Semantic similarity search (if local embedder exists)
        if self.section_embedder:
            query_vec = self.section_embedder.encode(query, convert_to_tensor=True)
            section_scores = []
            for doc in self.documents:
                doc_vec = self.section_embedder.encode(doc.page_content, convert_to_tensor=True)
                score = util.cos_sim(query_vec, doc_vec).item()
                section_scores.append((score, doc))

            section_scores.sort(reverse=True, key=lambda x: x[0])
            for score, doc in section_scores[:top_k]:
                if doc not in matched_docs:
                    matched_docs.append(doc)

        return matched_docs

    # -----------------------------
    # Answer function
    # -----------------------------
    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        # Numeric question handling
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        # Section-aware search
        section_docs = self.find_section(question_text, top_k=top_k)
        if section_docs:
            merged_text = "\n\n".join(d.page_content for d in section_docs)
            return {"answer": merged_text.strip(), "sources": section_docs}

        # Vector similarity retrieval
        retriever = self.vector_store.vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k * 3})
        candidates = retriever.get_relevant_documents(question_text)

        # Include keyword hits
        keyword_hits = [d for d in self.documents if question_text.lower() in d.page_content.lower()]
        seen = set()
        merged_candidates = []
        for d in candidates + keyword_hits:
            doc_id = d.metadata.get("source") or d.metadata.get("page_number") or id(d)
            if doc_id not in seen:
                merged_candidates.append(d)
                seen.add(doc_id)

        if not merged_candidates:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        # Rerank with LLM
        reranked_docs = rerank_with_llm(question_text, merged_candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join([f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs])
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
        prompt = f"""
You are a strict PDF answering agent.

User question: "{question_text}"

Relevant text:
{context}

Instructions:
- Answer using ONLY the text above, VERBATIM.
- Preserve formatting, bullets, line breaks.
- If text is from OCR, prefix with: "Extracted from image:"
- If nothing matches, return only: "⚠️ No relevant section found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
