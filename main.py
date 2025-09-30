"""
Application entry point for Upskill Advisor MVP - Backend.

Design choices for future-proofing:
- Mounts versioned routers using a configurable prefix from core.config Settings.
- Keeps legacy/basic routes for quick health checks while the API evolves.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging_config import configure_logging
from api.v1.routes import router as v1_router
from services.crawler_service import crawl_courses
from services.performance_service import get_performance_metrics
from services.retriever import get_retriever

# Set HF_HOME environment variable to avoid deprecated TRANSFORMERS_CACHE warning
_settings = get_settings()
os.environ["HF_HOME"] = _settings.hf_home

# Configure structured logging
configure_logging()

app = FastAPI(title="Upskill Advisor MVP - Backend", version="0.1.0")

# Basic CORS (can be restricted via settings in the future)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Server running"}



@app.get("/crawl")
async def crawl():
    """Crawls courses from various sources."""
    courses = crawl_courses()
    return {"crawled_courses": courses}


@app.get("/performance")
async def performance():
    """Returns performance metrics."""
    metrics = get_performance_metrics()
    return metrics


# Mount versioned API routers
app.include_router(v1_router, prefix=_settings.api_v1_prefix)


@app.on_event("startup")
async def startup_event():
    """Initialize heavy components on startup to improve first request performance."""
    logger = logging.getLogger("startup")
    logger.info("Starting application initialization...")

    try:
        # Preload retriever and compute embeddings
        retriever = get_retriever()
        await retriever._ensure_bm25()
        await retriever._ensure_vectors()

        # Preload local LLM if enabled
        if _settings.use_local_llm:
            from services.local_llm import get_local_llm
            llm = get_local_llm()
            if llm:
                logger.info("Local LLM preloaded successfully")

        logger.info("Application initialization completed successfully")

    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        # Don't fail startup, just log the error