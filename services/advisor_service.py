"""
Advisor service implementing a minimal, extensible RAG-style recommendation.

Design choices for future-proofing:
- Accepts flexible profile fields (skills, target_skills) to avoid strict coupling to a profile schema.
- Uses simple, explainable scoring (skill overlap) with room to plug in BM25 or embeddings later.
- Loads courses from a JSON file if not provided, validated via the Course schema to ensure consistency.
- Returns AdviseResult which can expand over time while the external response wrapper stays stable.
- Enhanced with cross-encoder re-ranking for improved relevance scoring.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from schemas.api import AdviseRequest, AdviseResult, UserProfile
from schemas.course import Course
from services.retriever import Retriever, RetrievalQuery

logger = logging.getLogger("advisor")


def _get_cross_encoder():
    """Get cross-encoder model for re-ranking, with graceful fallback."""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Cross-encoder unavailable, falling back to overlap scoring: {e}")
        return None


def _rerank_with_cross_encoder(query_text: str, courses: List[Course], cross_encoder, top_k: int) -> List[Course]:
    """Re-rank courses using cross-encoder model."""
    if not cross_encoder or not courses:
        return courses[:top_k]

    try:
        # Create query-document pairs
        pairs = []
        for course in courses:
            doc_text = f"{course.title}. Skills: {', '.join(course.skills)}. Difficulty: {course.difficulty}. Duration: {course.duration_weeks} weeks."
            pairs.append([query_text, doc_text])

        # Get relevance scores
        scores = cross_encoder.predict(pairs)

        # Sort courses by scores in descending order
        scored_courses = list(zip(courses, scores))
        scored_courses.sort(key=lambda x: x[1], reverse=True)

        # Return top-k courses
        reranked_courses = [course for course, score in scored_courses[:top_k]]

        logger.info(f"Re-ranked {len(courses)} courses using cross-encoder, selected top {len(reranked_courses)}")
        return reranked_courses

    except Exception as e:
        logger.warning(f"Cross-encoder re-ranking failed, using original order: {e}")
        return courses[:top_k]




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


async def _make_plan_llm(user_skills: List[str], target_skills: List[str], courses: List[Course], years_experience: Optional[int] = None, goal_role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Enhanced LLM-based plan generation with sophisticated prompt and JSON output parsing."""
    try:
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.exceptions import OutputParserException
    except Exception as e:
        logger.warning(f"LangChain dependencies unavailable: {e}")
        return None

    try:
        # Enhanced prompt template acting as expert career coach
        tmpl = """You are an expert career coach and learning advisor with deep knowledge of professional development paths.

CONTEXT:
- Current Skills: {user_skills}
- Target Skills: {target_skills}
- Years of Experience: {years_experience}
- Goal Role: {goal_role}
- Available Courses: {courses}

TASK:
Create a comprehensive, personalized learning plan that bridges the gap between current and target skills. Consider the user's experience level and career goals.

OUTPUT FORMAT:
Return a JSON object with the following structure:
{{
    "plan": [
        {{
            "course_id": "course_identifier",
            "why": "Detailed explanation of why this course is crucial for the user's goals",
            "order": 1,
            "estimated_weeks": 4
        }}
    ],
    "timeline": {{
        "total_weeks": 12,
        "phases": [
            {{
                "phase": "Foundation",
                "weeks": "1-4",
                "focus": "Building core skills"
            }}
        ]
    }},
    "gap_map": {{
        "skill_name": ["specific sub-skills or concepts to learn"]
    }},
    "notes": "Overall strategy summary and additional recommendations"
}}

GUIDELINES:
1. Prioritize courses based on prerequisite relationships and learning progression
2. Consider the user's experience level when recommending difficulty
3. Provide specific, actionable explanations for each course selection
4. Create a realistic timeline with clear phases
5. Include practical advice in the notes section
6. Map target skills to specific learning objectives

Generate the learning plan:"""

        # Set up the JSON output parser
        parser = JsonOutputParser()

        prompt = PromptTemplate(
            template=tmpl,
            input_variables=["user_skills", "target_skills", "years_experience", "goal_role", "courses"]
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # Create the chain
        chain = prompt | llm | parser

        # Prepare course information with more detail
        course_info = []
        for c in courses:
            course_detail = (
                f"ID: {c.course_id}, Title: {c.title}, "
                f"Skills: {', '.join(c.skills)}, "
                f"Difficulty: {c.difficulty}, "
                f"Duration: {c.duration_weeks} weeks"
            )
            if c.provider:
                course_detail += f", Provider: {c.provider}"
            course_info.append(course_detail)

        inputs = {
            "user_skills": ", ".join(user_skills) if user_skills else "None specified",
            "target_skills": ", ".join(target_skills) if target_skills else "General professional development",
            "years_experience": str(years_experience) if years_experience is not None else "Not specified",
            "goal_role": goal_role or "Professional development",
            "courses": "\n".join(course_info) if course_info else "No specific courses available"
        }

        # Execute the chain
        result = await chain.ainvoke(inputs)

        # Validate the result structure
        if isinstance(result, dict) and "plan" in result:
            logger.info("LLM plan generation successful with enhanced prompt")
            return result
        else:
            logger.warning("LLM returned invalid structure, falling back to heuristic")
            return None

    except OutputParserException as e:
        logger.warning(f"JSON parsing failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM plan generation failed: {e}")
        return None


async def advise(request: AdviseRequest, retriever: Retriever, top_k: int = 5) -> AdviseResult:
    """Asynchronous RAG-based advisor using a retriever with cross-encoder re-ranking and LLM-backed plan generation."""
    # Typed profile usage
    profile = request.profile
    user_skills = set(map(str, profile.skills))
    target_skills = set(map(str, profile.target_skills))

    # Retrieve a larger set of initial candidates for re-ranking (5x the final top_k)
    initial_candidates = max(top_k * 5, 25)  # Ensure we have enough candidates
    query = RetrievalQuery(skills=list(user_skills), target_skills=list(target_skills))
    retrieved: List[Course] = await retriever.hybrid_search(query, top_k=initial_candidates)

    # Construct query text from user profile for cross-encoder
    goal_role = getattr(profile, 'goal_role', None) or "professional development"
    query_text = f"Goal role: {goal_role}. Target skills: {', '.join(target_skills)}. Current skills: {', '.join(user_skills)}"

    # Initialize cross-encoder model
    cross_encoder = _get_cross_encoder()

    # Re-rank using cross-encoder if available, otherwise fall back to overlap scoring
    if cross_encoder:
        top_courses = _rerank_with_cross_encoder(query_text, retrieved, cross_encoder, top_k)
        reranking_method = "cross-encoder"
    else:
        # Fallback to original overlap scoring method
        scored: List[Tuple[Course, float]] = []
        for c in retrieved:
            scored.append((c, _overlap_score(c, target_skills)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_courses = [c for c, s in scored[:top_k] if s > 0.0] or (retrieved[:top_k] if retrieved else [])
        reranking_method = "overlap-scoring"

    # Build initial gap map as fallback
    fallback_gap_map = _build_gap_map(user_skills, target_skills)

    # Prioritize LLM-generated plan with enhanced context
    years_experience = getattr(profile, 'years_experience', None)
    llm_result = await _make_plan_llm(
        list(user_skills),
        list(target_skills),
        top_courses,
        years_experience=years_experience,
        goal_role=goal_role
    )

    if llm_result and isinstance(llm_result, dict):
        # Use LLM-generated comprehensive plan
        plan = llm_result.get("plan", [])
        gap_map = llm_result.get("gap_map", fallback_gap_map)
        timeline = llm_result.get("timeline", {"total_weeks": 12})
        notes = llm_result.get("notes", f"Comprehensive plan generated via LLM with {reranking_method} re-ranking.")

        # Add timeline information to notes if available
        if timeline and "total_weeks" in timeline:
            notes += f" Estimated completion: {timeline['total_weeks']} weeks."

        logger.info("Using LLM-generated comprehensive plan")
    else:
        # Fallback to heuristic method
        plan = await _make_plan_heuristic(fallback_gap_map.keys(), top_courses)
        gap_map = fallback_gap_map
        notes = f"Plan generated via heuristic with {reranking_method} re-ranking; consider configuring LLM (OpenAI API key) for richer guidance."
        logger.info("Using heuristic fallback plan")

    logger.info(
        "advisor_completed",
        extra={
            "top_k": top_k,
            "initial_candidates": len(retrieved),
            "selected": len(top_courses),
            "missing_skills": list(gap_map.keys()),
            "reranking_method": reranking_method,
        },
    )

    return AdviseResult(plan=plan, gap_map=gap_map, recommended_courses=top_courses, notes=notes)
