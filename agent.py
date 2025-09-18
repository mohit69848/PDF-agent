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
         Find the section by heading (e.g., DISCLAIMER) and return all text under it
        until the next heading or large gap.
        """
        query_lower = query.strip().lower()

        for doc in self.documents:
            lines = doc.page_content.splitlines()
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Heuristic to detect headers: upper case or capitalized long lines
                is_heading = (
                    clean_line.isupper() and len(clean_line) > 4
                ) or (clean_line.istitle() and len(clean_line) > 4)

                # Check if query matches heading
                if is_heading and query_lower in clean_line.lower():
                    # Collect the heading and subsequent content till next heading or page end
                    section_lines = [clean_line]
                    for next_line in lines[i+1:]:
                        next_clean = next_line.strip()
                        if not next_clean:
                            continue
                        # Stop if next heading detected
                        if (next_clean.isupper() and len(next_clean) > 4) or (next_clean.istitle() and len(next_clean) > 4):
                            break
                        section_lines.append(next_clean)
                    return Document(page_content="\n".join(section_lines), metadata=doc.metadata)
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

        # First, attempt exact section match
        section_doc = self.find_section(question_text)
        if section_doc:
            return {"answer": section_doc.page_content.strip(), "sources": [section_doc]}

        # If not found, fallback to similarity search
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3},
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Try exact text match in candidates
        for doc in candidates:
            if question_text.lower() in doc.page_content.lower():
                return {"answer": doc.page_content.strip(), "sources": [doc]}

        # Fallback to reranker
        reranked_docs = rerank_with_llm(question_text, candidates, top_k=top_k)
        if not reranked_docs:
            return {"answer": "⚠️ No relevant content found.", "sources": []}

        context = "\n\n".join([
            f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs
        ])
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

        prompt = f"""
You are a PDF Question Answering AI. The user has uploaded a document and is asking questions about it.

User Question:
{question_text}

Relevant Extracted Text (may include OCR output, bullet points, or paragraphs):
{context}

Instructions:
1. Summarize concisely.
2. If the answer is explicitly stated in the extracted text, return it VERBATIM (do not rephrase).
3. If the extracted text contains multiple bullet points or paragraphs relevant to the question, return the FULL block of text without summarizing.
4. If the text was extracted from an image (OCR), prefix the response with: "Extracted from image:".
5. Always preserve the original formatting (line breaks, bullet points, capitalization).
6. Include the source page number(s) at the end of your answer in this format:  
   Source: Page X (or Pages X–Y if multiple).
7. Do NOT add interpretations, assumptions, or extra commentary. Only return what is explicitly present in the document.
8. If no relevant content is found, reply: "⚠️ No relevant content found in the document."
"""
        result = llm.invoke([HumanMessage(content=prompt)])
        summary = result.content if result else "⚠️ No relevant content found."

        return {"answer": summary, "sources": reranked_docs}
