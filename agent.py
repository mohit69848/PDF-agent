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

        # Build vector store for retrieval
        self.vector_store.build(docs, source_file=pdf_path, progress_callback=progress_callback)

        # Build QA chain (if needed)
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
        pattern = re.compile(r'(\d+)[\.:]\s*(.+?)(?=(?:\n\d+[\.:])|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_num = int(match[0])
            q_text = match[1].strip().replace("\n", " ")
            self.question_map[q_num] = q_text

    # -----------------------------
    # Section-aware keyword + semantic search
    # -----------------------------
    def find_section(self, query: str, top_k: int = 3) -> List[Document]:
        query_lower = query.lower()
        matched_docs = []

        # Keyword scan
        for doc in self.documents:
            lines = doc.page_content.splitlines()
            section_lines = []
            capture = False
            # inside_section = False

            for line in lines:
                clean_line = line.strip()
                if not clean_line:
                    continue

                # Dynamically detect section heading matching query
                is_heading = clean_line.isupper() and len(clean_line.split()) <= 8
                # Start capturing if heading matches query

                if is_heading and query_lower in clean_line.lower():
                    capture = True
                    section_lines.append(clean_line)
                    continue
                    
                 # Capture content until next heading
                if capture:
                    if is_heading:
                        break
                    section_lines.append(clean_line)
                       
                      
            if section_lines:
                matched_docs.append(Document(
                    page_content="\n".join(section_lines),
                    metadata=doc.metadata
                    ))
                     # Return top matches if found
                # if matched_docs:
                    
        # Semantic similarity as backup
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

        return matched_docs[:top_k]

    # -----------------------------
    # Core RAG Answer function
    # -----------------------------
    def answer(self, user_input: str, top_k: int = 5, use_section_search: bool = True):
        if not self.vector_store.vectordb:
            raise ValueError("Vector store not initialized. Please ingest a PDF.")

        question_text = user_input.strip()

        # Step 1: Check for numeric question reference like "1 question"
        numeric_match = re.match(r'(\d+)\s*question', question_text.lower())
        if numeric_match:
            q_num = int(numeric_match.group(1))
            if q_num in self.question_map:
                # Return the question itself first
                return {"answer": self.question_map[q_num], "sources": []}
            else:
                return {"answer": f"⚠️ Question {q_num} not found in PDF.", "sources": []}

        # Step 2: Retrieve from PDF for answer
        retriever = self.vector_store.vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": top_k * 3}
        )
        candidates = retriever.get_relevant_documents(question_text)

        # Step 3: Optional section-aware augmentation
        section_docs = []
        if use_section_search:
            section_docs = self.find_section(question_text, top_k=top_k)
            # Prioritize section docs first
            candidates = section_docs + candidates
            
        # Step 4: Deduplicate
        seen = set()
        unique_candidates = []
        for d in candidates:
            doc_id = f"{d.metadata.get('page_number','N/A')}:{hash(d.page_content)}"
            if doc_id not in seen:
                unique_candidates.append(d)
                seen.add(doc_id)

        # Step 5: Rerank
        if len(unique_candidates) <= top_k:
            reranked_docs = unique_candidates
        else:
            reranked_docs = rerank_with_llm(question_text, unique_candidates, top_k=top_k)

        # Step 6: If PDF has no relevant content, fallback to LLM
        if not reranked_docs:
            llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
            prompt = f"""
You are a helpful assistant. Answer the user's question based on your knowledge.
User question: "{question_text}"
"""
            result = llm.invoke([HumanMessage(content=prompt)])
            return {"answer": result.content if result else "⚠️ No answer could be generated.", "sources": []}

        # Step 7: Build context and generate answer
        context = "\n\n".join(
            [f"[Page {d.metadata.get('page_number','N/A')}] {d.page_content}" for d in reranked_docs]
        )

        if self.qa_chain:
            output = self.qa_chain({"query": question_text})
            answer_text = output.get("result", "⚠️ No relevant section found in the document.")
            reranked_docs = output.get("source_documents", [])
        else:
            llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)
            prompt = f"""
You are a Retrieval-Augmented Generation (RAG) agent.

User question: "{question_text}"

Relevant text from the document:
{context}

Instructions:
- Answer using ONLY the text above, VERBATIM where possible.
- Preserve formatting, bullets, and line breaks.
- If no relevant text is found, return: "⚠️ No relevant section found in the document."
"""
            result = llm.invoke([HumanMessage(content=prompt)])
            answer_text = result.content if result else "⚠️ No relevant section found in the document."

        return {"answer": answer_text, "sources": reranked_docs}
