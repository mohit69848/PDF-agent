import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Detect if running in Streamlit cloud environment
IS_STREAMLIT = "STREAMLIT_SERVER_RUN_ID" in os.environ

if IS_STREAMLIT:
    # Use Streamlit secrets in deployment
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    LOCAL_EMBEDDING_MODEL = st.secrets.get("LOCAL_EMBEDDING_MODEL")
    DATABASE_URL = st.secrets["DATABASE"]["URL"]
    EMBEDDING_PROVIDER = st.secrets.get("EMBEDDING_PROVIDER")
    LLM_MODEL = st.secrets.get("LLM_MODEL")
else:
    # Load from .env file in local environment
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL")
    DATABASE_URL = os.getenv("DATABASE_URL")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
    LLM_MODEL = os.getenv("LLM_MODEL")

# Validation example
if EMBEDDING_PROVIDER == "google" and not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is required for google embeddings/LLM")

# Optional: Print or log environment for debugging
print("Environment:", "Streamlit" if IS_STREAMLIT else "Local")
print("Database URL:", DATABASE_URL)
