# embeddings.py

import os
import asyncio
from config import EMBEDDING_PROVIDER, GOOGLE_API_KEY, LOCAL_EMBEDDING_MODEL
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding():
    # Ensure an asyncio event loop exists before creating the embedding
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if EMBEDDING_PROVIDER.lower() == "google":
        if not GOOGLE_API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY is missing.")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    elif EMBEDDING_PROVIDER.lower() == "local":
        if not LOCAL_EMBEDDING_MODEL:
            raise EnvironmentError("LOCAL_EMBEDDING_MODEL is missing.")
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}  # force CPU usage
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
