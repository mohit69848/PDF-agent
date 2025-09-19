from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL, GOOGLE_API_KEY

def build_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # Dynamically decide chain_type based on retriever settings
    if retriever.search_kwargs.get("k", 0) > 8:
        chain_type = "map_reduce"
    else:
        chain_type = "stuff"

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        return_source_documents=True
    )
