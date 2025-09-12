from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL")
# CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH","./chroma_db")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
LLM_MODEL = os.getenv("LLM_MODEL")

if EMBEDDING_PROVIDER == "google" and not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is required for google embeddings/LLM")
