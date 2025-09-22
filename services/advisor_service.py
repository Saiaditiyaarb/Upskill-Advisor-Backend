"""
Advisor service implementing a minimal, extensible RAG-style recommendation.

Design choices for future-proofing:
- Accepts flexible profile fields (skills, target_skills) to avoid strict coupling to a profile schema.
- Uses simple, explainable scoring (skill overlap) with room to plug in BM25 or embeddings later.
- Loads courses from a JSON file if not provided, validated via the Course schema to ensure consistency.
- Returns AdviseResult which can expand over time while the external response wrapper stays stable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from schemas.api import AdviseRequest, AdviseResult, UserProfile
from schemas.course import Course
from services.retriever import Retriever, RetrievalQuery

logger = logging.getLogger("advisor")




def _overlap_score(course: Course, target_skills: Set[str]) -> float:
    if not target_skills:
        return 0.0
    overlap = set(map(str.lower, course.skills)) & set(map(str.lower, target_skills))
    return len(overlap) / float(len(target_skills))


def _build_gap_map(user_skills: Set[str], target_skills: Set[str]) -> Dict[str, List[str]]:
    """Map each target skill to missing sub-skills (placeholder: just the skill itself if missing)."""
    missing = {s for s in target_skills if s.lower() not in {u.lower() for u in user_skills}}
    return {s: [] for s in sorted(missing)}


async def _make_plan_heuristic(missing_skills: Iterable[str], top_courses: List[Course]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    course_by_skill: Dict[str, Optional[Course]] = {}
    for skill in missing_skills:
        chosen: Optional[Course] = None
        for c in top_courses:
            if any(skill.lower() == s.lower() for s in c.skills):
                chosen = c
                break
        course_by_skill[skill] = chosen
    for skill, course in course_by_skill.items():
        step = {
            "skill": skill,
            "action": "learn",
            "resource": course.title if course else None,
            "course_id": course.course_id if course else None,
        }
        plan.append(step)
    return plan


async def _make_plan_llm(user_skills: List[str], target_skills: List[str], courses: List[Course]) -> Optional[List[Dict[str, Any]]]:
    """Try to use a simple LLM chain via LangChain; return None if unavailable."""
    try:
        from langchain.prompts import PromptTemplate  # type: ignore
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception:
        return None

    try:
        # Build prompt
        tmpl = (
            "You are a learning advisor. Given user skills: {user_skills} and target skills: {target_skills}, "
            "and these courses: {courses}, produce a step-by-step learning plan as JSON list where each item has keys: "
            "skill, action, resource, course_id. Keep it concise."
        )
        prompt = PromptTemplate.from_template(tmpl)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # configurable later
        inputs = {
            "user_skills": ", ".join(user_skills),
            "target_skills": ", ".join(target_skills),
            "courses": "; ".join(f"{c.title} (skills: {', '.join(c.skills)})" for c in courses),
        }
        # Simple call; parse naive JSON if present
        text = await llm.apredict(prompt.format(**inputs))  # type: ignore[attr-defined]
        import json as _json
        plan = _json.loads(text) if text.strip().startswith("[") else None
        if isinstance(plan, list):
            return plan  # type: ignore[return-value]
        return None
    except Exception:
        return None


async def advise(request: AdviseRequest, retriever: Retriever, top_k: int = 5) -> AdviseResult:
    """Asynchronous RAG-based advisor using a retriever and LLM-backed plan generation."""
    # Typed profile usage
    profile = request.profile
    user_skills = set(map(str, profile.skills))
    target_skills = set(map(str, profile.target_skills))

    # Retrieve a small, relevant subset of courses (hybrid search)
    query = RetrievalQuery(skills=list(user_skills), target_skills=list(target_skills))
    retrieved: List[Course] = await retriever.hybrid_search(query, top_k=max(top_k, 5))

    # Score and rank retrieved courses by overlap with target skills to refine ordering
    scored: List[Tuple[Course, float]] = []
    for c in retrieved:
        scored.append((c, _overlap_score(c, target_skills)))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_courses = [c for c, s in scored[:top_k] if s > 0.0] or (retrieved[:top_k] if retrieved else [])

    gap_map = _build_gap_map(user_skills, target_skills)

    # Try LLM-generated plan; fallback to heuristic
    plan = await _make_plan_llm(list(user_skills), list(target_skills), top_courses)
    if not plan:
        plan = await _make_plan_heuristic(gap_map.keys(), top_courses)
        notes = "Plan generated via heuristic; consider configuring LLM for richer guidance."
    else:
        notes = "Plan generated via LLM based on retrieved courses and user profile."

    logger.info(
        "advisor_completed",
        extra={
            "top_k": top_k,
            "retrieved": len(retrieved),
            "selected": len(top_courses),
            "missing_skills": list(gap_map.keys()),
        },
    )

    return AdviseResult(plan=plan, gap_map=gap_map, recommended_courses=top_courses, notes=notes)
