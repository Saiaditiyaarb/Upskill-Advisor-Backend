"""
Core configuration module for environment variables and settings management.

Design choices:
- Uses python-dotenv to load environment variables from a .env file when present.
- Avoids pydantic BaseSettings to keep dependencies minimal and compatible with pydantic v1/v2.
- Provides a single get_settings() accessor with LRU caching to avoid repeated parsing.
- Keeps field names generic for forward-compatibility; new fields can be added without breaking consumers.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import importlib
try:
    _dotenv = importlib.import_module("dotenv")
    load_dotenv = getattr(_dotenv, "load_dotenv")
except Exception:
    # Fallback no-op to keep imports working when python-dotenv isn't installed
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False
from pydantic import BaseModel


class Settings(BaseModel):
    environment: str = "dev"
    # Pinecone (optional; service-level code should gracefully degrade if missing)
    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None

    # Embedding model used by ingestion (service falls back to simple ranking if unavailable)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # API versioning (useful for mounting routers and future deprecations)
    api_v1_prefix: str = "/api/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment variables (from .env if present) and build a Settings object.

    This function is cached so app startup and repeated imports are efficient.
    """
    load_dotenv()  # no-op if .env not present
    return Settings(
        environment=os.getenv("ENVIRONMENT", "dev"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index=os.getenv("PINECONE_INDEX"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
    )
