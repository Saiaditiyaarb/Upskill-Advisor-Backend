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
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # OpenRouter API Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "mistralai/mistral-7b-instruct"

    # Local LLM Configuration
    local_llm_model: str = "microsoft/DialoGPT-medium"
    use_local_llm: bool = True

    # Cross-encoder model for re-ranking
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder: bool = True

    # Model cache directory
    transformers_cache: str = "./models_cache"

    # Performance settings
    max_workers: int = 4
    batch_size: int = 32
    max_sequence_length: int = 512

    # Feature flags
    use_pinecone: bool = False
    enable_jd_ingestion: bool = True

    # File paths
    courses_json: str = "courses.json"
    jd_source_path: str = "job_descriptions"

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
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "microsoft/DialoGPT-medium"),
        use_local_llm=os.getenv("USE_LOCAL_LLM", "true").lower() == "true",
        cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        use_cross_encoder=os.getenv("USE_CROSS_ENCODER", "true").lower() == "true",
        transformers_cache=os.getenv("TRANSFORMERS_CACHE", "./models_cache"),
        max_workers=int(os.getenv("MAX_WORKERS", "4")),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
        use_pinecone=os.getenv("USE_PINECONE", "false").lower() == "true",
        enable_jd_ingestion=os.getenv("ENABLE_JD_INGESTION", "true").lower() == "true",
        courses_json=os.getenv("COURSES_JSON", "courses.json"),
        jd_source_path=os.getenv("JD_SOURCE_PATH", "job_descriptions"),
        api_v1_prefix=os.getenv("API_V1_PREFIX", "/api/v1"),
    )
