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
import string

class PDFQAAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.qa_chain = None
        self.question_map: Dict[int, str] = {}

    def ingest(self, pdf_path: str, progress_callback: Callable = None) -> int:
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
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    def find_references_section(self, documents: List[Document], user_query: str) -> Document | None:
        """
        Dynamically detect references-like sections by analyzing document structure,
        formatting patterns, and content relevance to the user's query.
        """
        # Lowercase user query for relevance checking
        query_terms = set(word.strip(string.punctuation).lower() for word in user_query.split())

        # Score candidate sections based on heading format and query relevance
        best_score = 0
        best_doc = None

        for doc in documents:
            lines = doc.page_content.splitlines()
            # Focus only on the last 20% of pages
            if not self._is_in_last_part(doc, documents):
                continue

            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Heading detection: long, uppercase or capitalized line
                heading_score = self._heading_score(clean_line)

                # Look ahead for citations, numbered lists, or URLs
                citation_score = self._citation_score(lines, i + 1)

                # Check if the section content is relevant to user query
                content_score = self._content_relevance_score(lines[i:], query_terms)

                total_score = heading_score + citation_score + content_score

                if total_score > best_score:
                    best_score = total_score
                    best_doc = Document(page_content="\n".join(lines[i:]), metadata=doc.metadata)

        return best_doc if best_score > 1 else None

    def _is_in_last_part(self, doc: Document, documents: List[Document]) -> bool:
        idx = documents.index(doc)
        return idx >= int(len(documents) * 0.8)

    def _heading_score(self, line: str) -> float:
        if len(line) < 5:
            return 0
        if line.isupper():
            return 2
        if line.istitle() and len(line.split()) > 1:
            return 1
        return 0

    def _citation_score(self, lines: List[str], start_idx: int) -> float:
        score = 0
        for line in lines[:10]:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r'^(\d+[\.\)]|\*|-)', stripped):
                score += 1
            if "http" in stripped or "www." in stripped:
                score += 1
        return min(score, 3)

    def _content_relevance_score(self, lines: List[str], query_terms: set) -> float:
        text = " ".join(lines).lower()
        terms_found = sum(1 for term in query_terms if term in text)
        return min(terms_found / max(len(query_terms), 1), 1)

    def answer(self, user_input: str, top_k: int = 5):
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

        references_doc = self.find_references_section(candidates, question_text)
        if references_doc:
            return {"answer": references_doc.page_content.strip(), "sources": [references_doc]}

        for doc in candidates:
            if question_text.lower() in doc.page_content.lower():
                return {"answer": doc.page_content.strip(), "sources": [doc]}

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
