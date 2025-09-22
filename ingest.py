"""
Data ingestion script for Upskill Advisor.

Responsibilities:
- Read a courses.json file from the project root (or path provided via env VAR COURSES_JSON).
- Validate each entry using the future-proof Course schema (flexible metadata).
- Generate embeddings for each course using SentenceTransformer if available; otherwise use a deterministic hashing fallback.
- Upsert vectors into Pinecone if PINECONE_API_KEY and PINECONE_INDEX are configured.

Re-runnable:
- Uses course_id as the vector ID so repeated runs upsert/overwrite deterministically.
- Batches upserts to handle large datasets.

Design choices for future-proofing:
- Validation via Course allows new unknown fields to be ignored (model_config extra="ignore").
- Embedding function is abstracted with a graceful fallback so environments without the model can still run the pipeline.
- Pinecone interaction is optional; if not configured, the script completes validation and embedding generation without error.
"""
from __future__ import annotations

import json
import os
import logging
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional

from core.config import get_settings
from schemas.course import Course

logger = logging.getLogger("ingest")


def _load_courses_from_file(path: str) -> List[Course]:
    if not os.path.exists(path):
        logger.warning("courses file not found", extra={"path": path})
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("courses", [])
    valid: List[Course] = []
    invalid_count = 0
    for it in items:
        try:
            valid.append(Course(**it))
        except Exception:
            invalid_count += 1
    logger.info("loaded courses", extra={"valid": len(valid), "invalid": invalid_count})
    return valid


def _build_embedding_text(c: Course) -> str:
    skills = ", ".join(c.skills)
    return (
        f"{c.title}. Skills: {skills}. Difficulty: {c.difficulty}. "
        f"Duration weeks: {c.duration_weeks}. Provider: {c.provider or ''}"
    )


def _get_embedder():
    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(settings.embedding_model_name)

        def _embed(texts: List[str]) -> List[List[float]]:
            return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]

        logger.info("using SentenceTransformer", extra={"model": settings.embedding_model_name})
        return _embed
    except Exception as e:
        logger.warning("SentenceTransformer unavailable; using hashing fallback", extra={"error": str(e)})

        def _hashing_embed(texts: List[str], dim: int = 256) -> List[List[float]]:
            vecs: List[List[float]] = []
            for t in texts:
                buckets = [0.0] * dim
                for token in t.lower().split():
                    h = int(sha256(token.encode("utf-8")).hexdigest(), 16)
                    idx = h % dim
                    sign = -1.0 if (h // dim) % 2 else 1.0
                    buckets[idx] += sign
                # L2 normalize
                norm = sum(x * x for x in buckets) ** 0.5 or 1.0
                vecs.append([x / norm for x in buckets])
            return vecs

        return _hashing_embed


def _upsert_pinecone(vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> None:
    settings = get_settings()
    api_key = settings.pinecone_api_key
    index_name = settings.pinecone_index
    if not api_key or not index_name:
        logger.info("Pinecone not configured. Skipping upsert.")
        return

    # Support both older and newer pinecone clients where possible
    pc = None
    index = None
    try:
        import pinecone  # type: ignore

        pinecone.init(api_key=api_key)
        if index_name not in [i.name if hasattr(i, 'name') else i for i in pinecone.list_indexes()]:
            # Create index if supported by client; if not, assume exists
            try:
                pinecone.create_index(index_name, dimension=len(vectors[0]["values"]))
            except Exception:
                pass
        index = pinecone.Index(index_name)
    except Exception:
        try:
            from pinecone import Pinecone  # type: ignore

            pc = Pinecone(api_key=api_key)
            try:
                index = pc.Index(index_name)
            except Exception:
                # Attempt create then get
                pc.create_index(name=index_name, dimension=len(vectors[0]["values"]))
                index = pc.Index(index_name)
        except Exception as e:
            logger.warning("Pinecone client not available. Skipping upsert.", extra={"error": str(e)})
            return

    # Prepare payload depending on client variant
    def _batch(iterable: List[Any], size: int = 100) -> Iterable[List[Any]]:
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    upserted = 0
    for batch in _batch(vectors, size=100):
        try:
            if hasattr(index, "upsert"):
                index.upsert(vectors=batch, namespace=namespace)
            else:
                # New client expects dict with vectors key
                index.upsert({"vectors": batch, "namespace": namespace})
            upserted += len(batch)
        except Exception as e:
            logger.error("upsert batch failed", extra={"error": str(e)})
    logger.info("upsert completed", extra={"upserted": upserted, "index": index_name})


def run(path: Optional[str] = None, namespace: Optional[str] = None) -> None:
    settings = get_settings()
    courses_path = path or os.getenv("COURSES_JSON", os.path.join(os.getcwd(), "courses.json"))
    courses = _load_courses_from_file(courses_path)
    if not courses:
        logger.info("no courses to process")
        return

    embed = _get_embedder()
    texts = [_build_embedding_text(c) for c in courses]
    vectors_f = embed(texts)

    # Build Pinecone-ready vectors
    vectors: List[Dict[str, Any]] = []
    for c, v in zip(courses, vectors_f):
        meta = {
            "title": c.title,
            "skills": c.skills,
            "difficulty": c.difficulty,
            "duration_weeks": c.duration_weeks,
            **(c.metadata or {}),  # allow future arbitrary fields
        }
        if c.provider:
            meta["provider"] = c.provider
        if c.url:
            meta["url"] = c.url
        vectors.append({"id": c.course_id, "values": v, "metadata": meta})

    logger.info("prepared vectors", extra={"count": len(vectors), "dim": len(vectors[0]['values'])})

    # Optional upsert
    _upsert_pinecone(vectors, namespace=namespace)


if __name__ == "__main__":
    # Allow simple CLI usage; environment variables control configuration
    run()
