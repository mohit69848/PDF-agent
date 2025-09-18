# pdf_qa_agent.py
from loader import load_pdf
from vector_store import VectorStore
from retriever import build_qa_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage
from reranker import rerank_with_llm
from config import LLM_MODEL, GOOGLE_API_KEY, EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Callable, Dict
import re

# # Optional: local semantic search
# if EMBEDDING_PROVIDER == "local" and LOCAL_EMBEDDING_MODEL:
from sentence_transformers import util
# else:
#     SentenceTransformer = None


class PDFQAAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.qa_chain = None
        self.documents: List[Document] = []
        self.question_map: Dict[int, str] = {}

        # # Initialize local semantic search if available
        # if SentenceTransformer:
        #     self.section_detector = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        # else:
        self.section_detector = None

    # -----------------------------
    # PDF ingestion
    # -----------------------------
    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
        """Load PDF, split into chunks, build vector store, and map numbered questions."""
        docs: List[Document] = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No valid content found in PDF.")

        self.documents = docs
        full_text = "\n".join([d.page_content for d in docs])
        self._map_questions(full_text)

        # Build vector store
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
    # Section-aware search
    # -----------------------------
    def find_section(self, query: str) -> List[Document]:
        """Keyword-based section matching across chunks."""
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

                if any(word in clean_line.lower() for word in query_keywords):
                    inside_section = True

                if inside_section:
                    section_lines.append(clean_line)
                    # End at next heading
                    if (
                        i + 1 < len(lines)
                        and lines[i + 1].isupper()
                        and len(lines[i + 1].split()) <= 8
                    ):
                        inside_section = False
                        break

            if section_lines:
                matched_docs.append(Document(page_content="\n".join(section_lines), metadata=doc.metadata))

        return matched_docs

    # -----------------------------
    # Optional: semantic section search
    # -----------------------------
    def find_section_semantic(self, query: str, threshold: float = 0.6) -> List[Document]:
        """Use local embeddings to find semantically relevant sections."""
        if not self.documents or not self.section_detector:
            return []

        query_emb = self.section_detector.encode(query, convert_to_tensor=True)
        matched_docs = []

        for doc in self.documents:
            doc_emb = self.section_detector.encode(doc.page_content, convert_to_tensor=True)
            score = util.cos_sim(query_emb, doc_emb).item()
            if score >= threshold:
                matched_docs.append(doc)

        return matched_docs

    # -----------------------------
    # Main answer function
    # -----------------------------
    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        # Numeric questions e.g., "5 question"
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                question_text = self.question_map[q_num]
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        # 1. Semantic search if available
        semantic_docs = self.find_section_semantic(question_text) if self.section_detector else []
        if semantic_docs:
            return {"answer": "\n\n".join(d.page_content for d in semantic_docs), "sources": semantic_docs}

        # 2. Keyword section search
        section_docs = self.find_section(question_text)
        if section_docs:
            return {"answer": "\n\n".join(d.page_content for d in section_docs), "sources": section_docs}

        # 3. Vector similarity search
        retriever = self.vector_store.vectordb.as_retriever(search_type="mmr", search_kwargs={"k": top_k * 3})
        candidates = retriever.get_relevant_documents(question_text)

        # 4. Keyword fallback
        candidates.extend([d for d in self.documents if question_text.lower() in d.page_content.lower()])

        if not candidates:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        # 5. Rerank with LLM
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
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
- Answer using ONLY this text, return it VERBATIM.
- Preserve formatting, bullets, line breaks.
- If it comes from OCR/image, prefix with "Extracted from image:".
- If nothing matches, return only: ⚠️ No relevant content found.
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
