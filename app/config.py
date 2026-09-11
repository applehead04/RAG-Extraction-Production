"""Central configuration: environment variables, logging, and shared client factories.

All external clients (LLM, embeddings, vector store) are created lazily and
cached, so importing this module has no side effects beyond reading .env.
"""
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
BM25_DIR = os.getenv("BM25_DIR", "./bm25_index")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pillar3_disclosures")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def standardize_bank_name(name: str) -> str:
    """Canonical bank key used in vector-store metadata and BM25 file names."""
    return str(name).strip().upper()


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        max_retries=3,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )


@lru_cache(maxsize=1)
def get_chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )