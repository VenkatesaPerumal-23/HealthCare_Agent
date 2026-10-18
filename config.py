"""
Shared configuration, tracing setup, and model clients for HealthGuard AI.

Everything here is import-only (no Streamlit UI calls at module load time,
other than @st.cache_resource decorators), so retrieval.py, graph.py, and
main.py can all import from this file without import-order problems.
"""
import os

import streamlit as st
from dotenv import load_dotenv

try:
    from langfuse import get_client, observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

    def observe(*args, **kwargs):
        """No-op fallback so every call site works whether or not Langfuse is installed."""
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Tracing is OPTIONAL: if the keys are absent, get_client() still returns a
# client but every call becomes a local no-op, so nothing else needs to branch
# on whether Langfuse is configured.
LANGFUSE_ENABLED = LANGFUSE_AVAILABLE and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
# Only construct the client when keys are actually present — calling get_client()
# without a public_key logs a noisy "Authentication error" on every import.
langfuse_client = get_client() if LANGFUSE_ENABLED else None

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
KNOWLEDGE_BASE_DIR = "knowledge-base"
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "qwen/qwen3.8-27b"
RETRIEVAL_K = 4

VALID_INTENTS = {
    "medical_qa",
    "latest_information",
    "symptom_check",
    "general_health",
    "conversation_context",
    "clarification",
}


# ============================================================
# SHARED MODEL / EMBEDDINGS
# ============================================================

@st.cache_resource
def get_llm():
    """Create one cached LLM client for the application."""
    return ChatGroq(temperature=0, model_name=LLM_MODEL_ID)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@observe(as_type="generation", name="llm_call")
def invoke_llm(prompt: str) -> str:
    """
    Single choke point for every LLM call in the graph.
    Centralizing it here means Langfuse sees every prompt/response as a traced
    "generation" (with model name + token usage, when the provider returns it)
    without touching each node's own logic, retries, or error handling.
    """
    response = get_llm().invoke(prompt)
    content = response.content
    return content if isinstance(content, str) else str(content)
