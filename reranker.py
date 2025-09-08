# reranker.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import LLM_MODEL, GOOGLE_API_KEY
from langchain.docstore.document import Document
from typing import List

def rerank_with_llm(question: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    Use LLM to rerank candidate docs by relevance.
    Returns top_k most relevant.
    """
    if not docs:
        return []

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    candidates_text = "\n\n".join(
        [f"[{i}] {d.page_content[:400]}" for i, d in enumerate(docs)]
    )

    prompt = (
        f"You are ranking text chunks for relevance.\n\n"
        f"User Question: {question}\n\n"
        f"Candidate Chunks:\n{candidates_text}\n\n"
        f"Task: Select the {top_k} chunks most relevant to the question. "
        f"Return only their numbers (e.g., 0,2,5)."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    chosen = response.content.strip()

    # Parse indices from response
    indices = []
    for token in chosen.replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < len(docs):
                indices.append(idx)

    indices = list(dict.fromkeys(indices))[:top_k]
    return [docs[i] for i in indices]
