from config import EMBEDDING_PROVIDER, GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding():
    """
    Returns an embedding model instance based on the configured provider.
    """
    if EMBEDDING_PROVIDER.lower() == "google":
        if not GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY is missing. Set it in your .env file."
            )
        # Use your Google embedding model
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Optional fallback for HuggingFace or OpenAI
    # elif EMBEDDING_PROVIDER.lower() == "huggingface":
    #     from langchain.embeddings import HuggingFaceEmbeddings
    #     return HuggingFaceEmbeddings()

    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
