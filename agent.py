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
from difflib import SequenceMatcher

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

    def similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_section_by_pattern(self, documents: List[Document], query: str) -> Document | None:
        query_lower = query.lower()
        results = []

        for doc in documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if len(clean_line) < 4:
                    continue

                # Detect heading: uppercase, title case, or surrounded by whitespace
                is_heading = (
                    clean_line.isupper() or
                    clean_line.istitle() or
                    (len(clean_line) > 10 and clean_line.lower().startswith(query_lower[:3]))
                )

                # Check similarity to the query
                sim_score = self.similar(clean_line, query)

                if is_heading and sim_score > 0.4:
                    # Collect following lines that form the section
                    section_lines = [clean_line]
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            break
                        section_lines.append(next_line)
                    content = "\n".join(section_lines).strip()
                    results.append((sim_score, Document(page_content=content, metadata=doc.metadata)))

        if results:
            # Return the highest similarity result
            results.sort(key=lambda x: x[0], reverse=True)
            return results[0][1]
        return None

    def answer(self, user_input: str, top_k: int = 5):
        question_text = user_input.strip()

        if not self.vector_store.vectordb:
            raise ValueError("Vector store is empty. Please ingest a PDF first.")

        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # First attempt: find section by pattern and similarity
        exact_section = self.find_section_by_pattern(candidates, question_text)
        if exact_section:
            return {
                "answer": exact_section.page_content.strip(),
                "sources": [exact_section]
            }

        # Fallback: rerank with LLM if no exact match found
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
