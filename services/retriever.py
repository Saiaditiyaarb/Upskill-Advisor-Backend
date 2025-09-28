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
        # Embedding-related caches
        self._embedder = None  # callable or None
        self._course_vectors: Optional[List[List[float]]] = None
        # Lazy initialize indices on first use

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

    # -------- Local vector search (for ablations) --------
    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(self._settings.embedding_model_name)
            def _embed(texts: List[str]) -> List[List[float]]:
                return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]
            logger.info("retriever_using_sentence_transformer", extra={"model": self._settings.embedding_model_name})
            self._embedder = _embed
            return self._embedder
        except Exception as e:
            logger.warning("SentenceTransformer unavailable; using hashing fallback", extra={"error": str(e)})
            def _hashing_embed(texts: List[str], dim: int = 256) -> List[List[float]]:
                from hashlib import sha256
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
            self._embedder = _hashing_embed
            return self._embedder

    @staticmethod
    def _build_embedding_text(c: Course) -> str:
        skills = ", ".join(c.skills)
        return f"{c.title}. Skills: {skills}. Difficulty: {c.difficulty}. Duration weeks: {c.duration_weeks}. Provider: {c.provider or ''}"

    async def _ensure_vectors(self) -> None:
        # Ensure courses are loaded
        await self._ensure_bm25()
        if not self._courses_cache:
            self._course_vectors = []
            return
        # If vectors are already computed and timestamp unchanged, skip
        if self._course_vectors is not None:
            return
        embed = self._get_embedder()
        texts = [self._build_embedding_text(c) for c in self._courses_cache]
        try:
            self._course_vectors = await asyncio.to_thread(embed, texts)
        except Exception:
            # If embedder isn't thread-safe, call directly
            self._course_vectors = embed(texts)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        # vectors already normalized; but guard anyway
        return float(dot)

    async def vector_search(self, query: RetrievalQuery, top_k: int = 10) -> List[Tuple[int, float]]:
        await self._ensure_vectors()
        if not self._course_vectors:
            return []
        # Build query text prioritizing target skills then skills then free text
        parts: List[str] = []
        if query.target_skills:
            parts.append(" ".join(query.target_skills))
        if query.skills:
            parts.append(" ".join(query.skills))
        if query.text:
            parts.append(query.text)
        if not parts:
            return []
        qtext = " ".join(parts)
        embed = self._get_embedder()
        try:
            qvec = (await asyncio.to_thread(embed, [qtext]))[0]
        except Exception:
            qvec = embed([qtext])[0]
        scores = [(idx, self._cosine(vec, qvec)) for idx, vec in enumerate(self._course_vectors or [])]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    async def keyword_search(self, query: RetrievalQuery, top_k: int = 10) -> List[Course]:
        ranked = await self._bm25_keyword(query, top_k=top_k)
        await self._ensure_bm25()
        courses = self._courses_cache
        results: List[Course] = []
        for idx, _score in ranked:
            if 0 <= idx < len(courses):
                results.append(courses[idx])
                if len(results) >= top_k:
                    break
        return results

    async def vector_search_courses(self, query: RetrievalQuery, top_k: int = 10) -> List[Course]:
        ranked = await self.vector_search(query, top_k=top_k)
        await self._ensure_bm25()
        courses = self._courses_cache
        results: List[Course] = []
        for idx, _score in ranked:
            if 0 <= idx < len(courses):
                results.append(courses[idx])
                if len(results) >= top_k:
                    break
        return results

    async def hybrid_search(self, query: RetrievalQuery, top_k: int = 5) -> List[Course]:
        """Hybrid retrieval: combine local vector search and BM25 keyword scores.
        If only one signal is available, fall back gracefully.
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
            # Ensure indices are initialized
            await self._ensure_bm25()
            await self._ensure_vectors()

            # Retrieve from both modalities
            search_multiplier = min(3, max(2, top_k // 2))
            bm25_ranked = await self._bm25_keyword(query, top_k=top_k * search_multiplier)
            vec_ranked = await self.vector_search(query, top_k=top_k * search_multiplier)

            courses = self._courses_cache
            # Normalize scores to [0,1]
            def _normalize(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
                if not pairs:
                    return []
                scores = [s for _, s in pairs]
                smin, smax = min(scores), max(scores)
                if smax - smin < 1e-9:
                    return [(i, 1.0) for i, _ in pairs]
                return [(i, (s - smin) / (smax - smin)) for i, s in pairs]

            bm25_n = _normalize(bm25_ranked)
            vec_n = _normalize(vec_ranked)

            # Weighted combine
            w_bm25, w_vec = 0.6, 0.4
            combined: dict[int, float] = {}
            for i, s in bm25_n:
                combined[i] = combined.get(i, 0.0) + w_bm25 * s
            for i, s in vec_n:
                combined[i] = combined.get(i, 0.0) + w_vec * s

            # If one side empty, fall back to the other
            if not combined:
                source = bm25_ranked or vec_ranked
                results: List[Course] = []
                for idx, _ in source[:top_k]:
                    if 0 <= idx < len(courses):
                        results.append(courses[idx])
                if results:
                    return results
                # last resort: first N
                return courses[:top_k]

            ranked_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)

            # Deduplicate by course_id while preserving ranking order
            seen_ids = set()
            results: List[Course] = []
            for idx, _score in ranked_indices:
                if 0 <= idx < len(courses):
                    c = courses[idx]
                    if c.course_id not in seen_ids:
                        seen_ids.add(c.course_id)
                        results.append(c)
                        if len(results) >= top_k:
                            break

            if not results:
                return courses[:top_k] if courses else []

            logger.debug(f"Hybrid search returned {len(results)} courses for query with {len(query.skills)} skills")
            return results

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
