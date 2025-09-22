"""
Application entry point for Upskill Advisor MVP - Backend.

Design choices for future-proofing:
- Mounts versioned routers using a configurable prefix from core.config Settings.
- Keeps legacy/basic routes for quick health checks while the API evolves.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging_config import configure_logging
from api.v1.routes import router as v1_router

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
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# Mount versioned API routers
_settings = get_settings()
app.include_router(v1_router, prefix=_settings.api_v1_prefix)
