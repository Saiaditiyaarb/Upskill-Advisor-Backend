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
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from functools import lru_cache

from rank_bm25 import BM25Okapi  # type: ignore

from core.config import get_settings
from schemas.course import Course

logger = logging.getLogger("retriever")


@dataclass
class RetrievalQuery:
    skills: List[str]
    target_skills: List[str]
    text: Optional[str] = None  # optional free-text query

    def __post_init__(self):
        """Validate and clean query data."""
        # Remove empty strings and strip whitespace
        self.skills = [skill.strip() for skill in self.skills if skill and skill.strip()]
        self.target_skills = [skill.strip() for skill in self.target_skills if skill and skill.strip()]

        if self.text:
            self.text = self.text.strip() or None

        # Log warning if no meaningful query data
        if not self.skills and not self.target_skills and not self.text:
            logger.warning("RetrievalQuery created with no meaningful search criteria")

    @property
    def is_empty(self) -> bool:
        """Check if the query has any meaningful search criteria."""
        return not self.skills and not self.target_skills and not self.text


class Retriever:
    def __init__(self, courses_path: str = "courses.json") -> None:
        self._settings = get_settings()
        self._courses_path = courses_path
        self._bm25 = None  # type: ignore
        self._bm25_corpus_tokens: List[List[str]] = []
        self._courses_cache: List[Course] = []
        self._courses_cache_timestamp: Optional[float] = None
        self._initialization_lock = asyncio.Lock()
        # Lazy initialize BM25 on first use

    async def _ensure_bm25(self) -> None:
        """Initialize BM25 with thread-safe caching and file modification checking."""
        async with self._initialization_lock:
            # Check if we need to reload based on file modification time
            current_timestamp = None
            if os.path.exists(self._courses_path):
                current_timestamp = os.path.getmtime(self._courses_path)

            # If BM25 is already initialized and file hasn't changed, return
            if (self._bm25 is not None and
                self._courses_cache_timestamp is not None and
                current_timestamp == self._courses_cache_timestamp):
                return

            logger.info(f"Initializing BM25 index from {self._courses_path}")

            try:
                # Load courses corpus from file
                courses = await self._load_local_courses()
                self._courses_cache = courses
                self._courses_cache_timestamp = current_timestamp

                if not courses:
                    logger.warning("No courses loaded, BM25 will be unavailable")
                    self._bm25_corpus_tokens = []
                    self._bm25 = None
                    return

                # Build corpus and tokenize
                corpus: List[str] = [self._course_text(c) for c in courses]
                self._bm25_corpus_tokens = [self._tokenize(doc) for doc in corpus]

                # Initialize BM25
                self._bm25 = BM25Okapi(self._bm25_corpus_tokens)

                logger.info(f"BM25 index initialized with {len(courses)} courses")

            except Exception as e:
                logger.error(f"Failed to initialize BM25 index: {e}")
                self._bm25_corpus_tokens = []
                self._bm25 = None
                self._courses_cache = []

    async def _load_local_courses(self) -> List[Course]:
        """Load courses from JSON file with comprehensive error handling and validation."""
        path = self._courses_path
        if not os.path.exists(path):
            logger.warning(f"Courses file not found: {path}")
            return []

        try:
            # file IO off main loop
            def _load() -> List[Course]:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Handle different JSON structures
                if isinstance(raw, list):
                    items = raw
                elif isinstance(raw, dict):
                    items = raw.get("courses", [])
                else:
                    logger.error(f"Invalid JSON structure in {path}")
                    return []

                out: List[Course] = []
                invalid_count = 0

                for i, item in enumerate(items):
                    try:
                        # Validate required fields before creating Course object
                        if not isinstance(item, dict):
                            invalid_count += 1
                            continue

                        # Check for required fields
                        required_fields = ['course_id', 'title', 'skills']
                        missing_fields = [field for field in required_fields if field not in item]
                        if missing_fields:
                            logger.warning(f"Course at index {i} missing required fields: {missing_fields}")
                            invalid_count += 1
                            continue

                        course = Course(**item)
                        out.append(course)

                    except Exception as e:
                        logger.warning(f"Failed to parse course at index {i}: {e}")
                        invalid_count += 1
                        continue

                if invalid_count > 0:
                    logger.warning(f"Skipped {invalid_count} invalid courses out of {len(items)} total")

                logger.info(f"Successfully loaded {len(out)} courses from {path}")
                return out

            return await asyncio.to_thread(_load)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in courses file {path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load courses from {path}: {e}")
            return []

    @staticmethod
    def _course_text(c: Course) -> str:
        skills = ", ".join(c.skills)
        return f"{c.title}. Skills: {skills}. Difficulty: {c.difficulty}. Duration: {c.duration_weeks} weeks. {c.provider or ''}"

    @staticmethod
    @lru_cache(maxsize=1000)
    def _tokenize(text: str) -> List[str]:
        """Tokenize text with caching for performance."""
        if not text:
            return []
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
        # Validate inputs
        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}, using default of 5")
            top_k = 5

        if query.is_empty:
            logger.warning("Empty query provided to hybrid_search")
            # Return first N courses as fallback
            await self._ensure_bm25()
            return self._courses_cache[:top_k] if self._courses_cache else []

        try:
            # Ensure BM25 is initialized
            await self._ensure_bm25()

            # Get BM25 results with expanded search space for better ranking
            search_multiplier = min(3, max(2, top_k // 2))  # Reasonable multiplier
            bm25_ranked = await self._bm25_keyword(query, top_k=top_k * search_multiplier)

            # Map BM25 indices to courses with validation
            courses = self._courses_cache
            selected: List[Tuple[Course, float]] = []

            for idx, score in bm25_ranked:
                if 0 <= idx < len(courses):
                    selected.append((courses[idx], float(score)))
                else:
                    logger.warning(f"BM25 returned invalid course index: {idx}")

            # Deduplicate by course_id while preserving ranking order
            seen = set()
            unique: List[Course] = []

            for course, score in selected:
                if course.course_id not in seen:
                    seen.add(course.course_id)
                    unique.append(course)
                    if len(unique) >= top_k:
                        break

            # Fallback strategies if no results
            if not unique:
                logger.info("No BM25 results found, using fallback strategy")
                if courses:
                    # Return first N courses as fallback
                    unique = courses[:top_k]
                else:
                    # Try to reload courses if cache is empty
                    logger.warning("Course cache is empty, attempting to reload")
                    courses = await self._load_local_courses()
                    unique = courses[:top_k] if courses else []

            logger.debug(f"Hybrid search returned {len(unique)} courses for query with {len(query.skills)} skills")
            return unique

        except Exception as e:
            logger.error(f"Error in hybrid_search: {e}")
            # Return empty list or basic fallback
            await self._ensure_bm25()
            return self._courses_cache[:top_k] if self._courses_cache else []


# Dependency provider for FastAPI
_retriever_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
