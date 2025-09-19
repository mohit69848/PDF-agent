from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL, GOOGLE_API_KEY

def build_qa_chain(retriever, chain_type: str = "stuff"):
    """
    Build a RetrievalQA chain using the given retriever and chain type.
    """
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
