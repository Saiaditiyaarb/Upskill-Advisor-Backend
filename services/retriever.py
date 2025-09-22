"""
Hybrid Retriever for Upskill Advisor.

Responsibilities:
- Perform hybrid retrieval combining semantic vector search (Pinecone) and keyword search (BM25).
- Provide an async interface usable by FastAPI and service layer.
- Gracefully degrade when external services are not configured by falling back to local JSON courses.

Notes:
- This is a minimal, production-lean implementation that can be extended later.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from rank_bm25 import BM25Okapi  # type: ignore

from core.config import get_settings
from schemas.course import Course


@dataclass
class RetrievalQuery:
    skills: List[str]
    target_skills: List[str]
    text: Optional[str] = None  # optional free-text query


class Retriever:
    def __init__(self, courses_path: str = "courses.json") -> None:
        self._settings = get_settings()
        self._courses_path = courses_path
        self._bm25 = None  # type: ignore
        self._bm25_corpus_tokens: List[List[str]] = []
        self._courses_cache: List[Course] = []
        # Lazy initialize BM25 on first use

    async def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        # Load courses corpus from file
        courses = await self._load_local_courses()
        self._courses_cache = courses
        if not courses:
            self._bm25_corpus_tokens = []
            self._bm25 = None
            return
        corpus: List[str] = [self._course_text(c) for c in courses]
        self._bm25_corpus_tokens = [self._tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(self._bm25_corpus_tokens)

    async def _load_local_courses(self) -> List[Course]:
        path = self._courses_path
        if not os.path.exists(path):
            return []
        try:
            # file IO off main loop
            def _load() -> List[Course]:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                items = raw if isinstance(raw, list) else raw.get("courses", [])
                out: List[Course] = []
                for it in items:
                    try:
                        out.append(Course(**it))
                    except Exception:
                        continue
                return out

            return await asyncio.to_thread(_load)
        except Exception:
            return []

    @staticmethod
    def _course_text(c: Course) -> str:
        skills = ", ".join(c.skills)
        return f"{c.title}. Skills: {skills}. Difficulty: {c.difficulty}. Duration: {c.duration_weeks} weeks. {c.provider or ''}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.strip() for t in text.lower().split() if t.strip()]

    async def _pinecone_semantic(self, query: RetrievalQuery, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return list of (course_id, score). Gracefully handle missing config/client."""
        settings = self._settings
        api_key = settings.pinecone_api_key
        index_name = settings.pinecone_index
        if not api_key or not index_name:
            return []
        # Build a very simple embedding using skills text; in production, use a real embedder.
        query_text = " ".join(query.target_skills or query.skills or [])

        try:
            # Support both old and new pinecone clients in a simple sync call wrapped with to_thread
            def _query_sync() -> List[Tuple[str, float]]:
                try:
                    import pinecone  # type: ignore
                    pinecone.init(api_key=api_key)
                    index = pinecone.Index(index_name)
                    # naive vector placeholder; rely on server-side sparse/dense hybrid if enabled
                    # Here just return empty since we don't have embeddings at query time
                    return []
                except Exception:
                    try:
                        from pinecone import Pinecone  # type: ignore
                        pc = Pinecone(api_key=api_key)
                        index = pc.Index(index_name)
                        # We cannot embed here without a model; return empty to rely on BM25
                        return []
                    except Exception:
                        return []

            return await asyncio.to_thread(_query_sync)
        except Exception:
            return []

    async def _bm25_keyword(self, query: RetrievalQuery, top_k: int = 10) -> List[Tuple[int, float]]:
        await self._ensure_bm25()
        if self._bm25 is None:
            return []
        tokens: List[str] = []
        if query.text:
            tokens.extend(self._tokenize(query.text))
        tokens.extend(self._tokenize(" ".join(query.target_skills)))
        tokens.extend(self._tokenize(" ".join(query.skills)))
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # take top indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    async def hybrid_search(self, query: RetrievalQuery, top_k: int = 5) -> List[Course]:
        """Hybrid retrieval: combine Pinecone semantic results with BM25 keyword results.
        Returns unique Courses ranked by combined score (BM25 primary in this minimal version).
        """
        # semantic = await self._pinecone_semantic(query, top_k=top_k)
        bm25_ranked = await self._bm25_keyword(query, top_k=top_k * 2)
        # Map BM25 indices to courses
        courses = self._courses_cache
        # score combine: currently BM25 only; placeholder for fusion with semantic later
        selected: List[Tuple[Course, float]] = []
        for idx, score in bm25_ranked:
            if 0 <= idx < len(courses):
                selected.append((courses[idx], float(score)))
        # Unique by course_id preserving order
        seen = set()
        unique: List[Course] = []
        for c, _ in selected:
            if c.course_id in seen:
                continue
            seen.add(c.course_id)
            unique.append(c)
            if len(unique) >= top_k:
                break
        # If BM25 yields nothing, fallback to first N local courses
        if not unique:
            if not courses:
                courses = await self._load_local_courses()
            unique = courses[:top_k]
        return unique


# Dependency provider for FastAPI
_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
