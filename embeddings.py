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
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
