from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Gemini (Primary)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # HuggingFace (Fallback)
    hf_api_key: str = ""
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Embeddings (always HuggingFace, no API key needed for sentence-transformers)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Store
    vector_store_path: str = "vector_store"

    # RAG settings
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
