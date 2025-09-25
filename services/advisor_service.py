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
from services.crawler_service import crawl_courses # Import the new crawler service
import json
import os
import asyncio
from pathlib import Path

logger = logging.getLogger("advisor")


def _get_cross_encoder():
    """Get cross-encoder model for re-ranking, with graceful fallback."""
    from core.config import get_settings
    settings = get_settings()

    if not settings.use_cross_encoder:
        logger.info("Cross-encoder disabled in configuration")
        return None

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(settings.cross_encoder_model)
        logger.info(f"Cross-encoder model loaded successfully: {settings.cross_encoder_model}")
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


async def _make_plan_llm(current_skills: List[Dict[str, str]], goal_role: str, courses: List[Course], years_experience: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Enhanced LLM-based plan generation using OpenRouter API with sophisticated prompt and JSON output parsing."""
    try:
        from langchain_openai import ChatOpenAI
        from core.config import get_settings
        import json
    except Exception as e:
        logger.warning(f"OpenRouter dependencies unavailable: {e}")
        return None

    try:
        # Get settings and configure OpenRouter
        settings = get_settings()

        if not settings.openrouter_api_key:
            logger.warning("OpenRouter API key not configured")
            return None

        # Initialize ChatOpenAI with OpenRouter configuration
        llm = ChatOpenAI(
            base_url=settings.openrouter_api_base,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            temperature=0.1,
            max_tokens=512
        )

        # Enhanced prompt template acting as expert career coach
        prompt_template = """You are an expert career coach and learning advisor with deep knowledge of professional development paths.

CONTEXT:
- Current Skills with Expertise: {current_skills}
- Years of Experience: {years_experience}
- Goal Role: {goal_role}
- Available Courses: {courses}

TASK:
Create a comprehensive, personalized learning plan to help the user advance toward their goal role. Consider their current skill levels and experience to recommend appropriate courses and learning progression.

OUTPUT FORMAT:
Return ONLY a valid JSON object with the following structure (no additional text):
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
7. Return ONLY valid JSON, no markdown or additional formatting

Generate the learning plan:"""

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

        # Format the prompt with actual values
        current_skills_text = ", ".join([f"{skill['name']} ({skill['expertise']})" for skill in current_skills]) if current_skills else "None specified"

        formatted_prompt = prompt_template.format(
            current_skills=current_skills_text,
            years_experience=str(years_experience) if years_experience is not None else "Not specified",
            goal_role=goal_role,
            courses="\n".join(course_info) if course_info else "No specific courses available"
        )

        # Generate response using OpenRouter
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)

        # Parse JSON from response content with improved error handling
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Log the raw response for debugging (truncated)
        logger.debug(f"OpenRouter raw response (first 500 chars): {response_text[:500]}")

        # Extract JSON from the response with multiple fallback strategies
        result = None

        # Strategy 1: Try to parse the entire response as JSON
        try:
            result = json.loads(response_text)
            logger.debug("Successfully parsed response as direct JSON")
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")

            # Strategy 2: Try to extract JSON from markdown code blocks
            import re
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in markdown code blocks
                r'```\s*(\{.*?\})\s*```',      # JSON in generic code blocks
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Balanced braces
                r'\{.*\}',  # Simple brace matching (fallback)
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_text = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        result = json.loads(json_text)
                        logger.debug(f"Successfully extracted JSON using pattern: {pattern}")
                        break
                    except json.JSONDecodeError:
                        continue

            # Strategy 3: Try to clean and parse common formatting issues
            if result is None:
                try:
                    # Remove common prefixes/suffixes that might interfere
                    cleaned_text = response_text.strip()

                    # Remove markdown formatting
                    cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text, flags=re.MULTILINE)
                    cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE)

                    # Try to find the first complete JSON object
                    start_idx = cleaned_text.find('{')
                    if start_idx != -1:
                        brace_count = 0
                        end_idx = start_idx
                        for i, char in enumerate(cleaned_text[start_idx:], start_idx):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break

                        if brace_count == 0:
                            json_candidate = cleaned_text[start_idx:end_idx]
                            result = json.loads(json_candidate)
                            logger.debug("Successfully parsed JSON after cleaning")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Cleaning strategy failed: {e}")

        if result is None:
            logger.warning(f"Failed to parse JSON from OpenRouter response. Response preview: {response_text[:200]}...")
            return None

        # Validate the result structure
        if isinstance(result, dict) and "plan" in result:
            logger.info("OpenRouter LLM plan generation successful")
            return result
        else:
            logger.warning("OpenRouter LLM returned invalid structure, falling back to heuristic")
            return None

    except Exception as e:
        logger.warning(f"OpenRouter LLM plan generation failed: {e}")
        return None


async def _add_courses_to_json(new_courses: List[Course], courses_file: str = "courses.json") -> bool:
    """
    Add new courses to the courses.json file asynchronously.

    Args:
        new_courses: List of Course objects to add
        courses_file: Path to the courses JSON file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        def _update_file():
            courses_path = Path(courses_file)

            # Load existing courses
            existing_courses = []
            if courses_path.exists():
                try:
                    with open(courses_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle different JSON structures
                    if isinstance(data, list):
                        existing_courses = data
                    elif isinstance(data, dict) and 'courses' in data:
                        existing_courses = data['courses']
                    else:
                        logger.warning(f"Unexpected JSON structure in {courses_file}, treating as empty")
                        existing_courses = []

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in {courses_file}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Error reading {courses_file}: {e}")
                    return False

            # Get existing course IDs to avoid duplicates
            existing_ids = set()
            existing_titles = set()

            for course in existing_courses:
                if isinstance(course, dict):
                    existing_ids.add(course.get('course_id', ''))
                    existing_titles.add(course.get('title', '').lower())

            # Convert new courses to dict format and filter duplicates
            new_course_dicts = []
            added_count = 0

            for course in new_courses:
                # Skip if course already exists
                if course.course_id in existing_ids or course.title.lower() in existing_titles:
                    logger.debug(f"Skipping duplicate course: {course.title}")
                    continue

                # Convert Course object to dict
                course_dict = {
                    'course_id': course.course_id,
                    'title': course.title,
                    'skills': course.skills,
                    'difficulty': course.difficulty,
                    'duration_weeks': course.duration_weeks,
                    'provider': course.provider,
                    'url': course.url,
                    'metadata': course.metadata
                }

                new_course_dicts.append(course_dict)
                existing_ids.add(course.course_id)
                existing_titles.add(course.title.lower())
                added_count += 1

            if not new_course_dicts:
                logger.info("No new courses to add (all were duplicates)")
                return True

            # Combine existing and new courses
            all_courses = existing_courses + new_course_dicts

            # Create backup of original file
            if courses_path.exists():
                backup_path = courses_path.with_suffix(f'.backup.{int(__import__("time").time())}.json')
                try:
                    import shutil
                    shutil.copy2(courses_path, backup_path)
                    logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")

            # Write updated courses back to file
            try:
                with open(courses_path, 'w', encoding='utf-8') as f:
                    json.dump(all_courses, f, indent=2, ensure_ascii=False)

                logger.info(f"Successfully added {added_count} new courses to {courses_file}")
                return True

            except Exception as e:
                logger.error(f"Error writing to {courses_file}: {e}")
                return False

        # Run file operations in thread to avoid blocking
        return await asyncio.to_thread(_update_file)

    except Exception as e:
        logger.error(f"Error in _add_courses_to_json: {e}")
        return False


async def advise(request: AdviseRequest, retriever: Retriever, top_k: int = 5) -> AdviseResult:
    """
    Asynchronous RAG-based advisor using a retriever with cross-encoder re-ranking and LLM-backed plan generation.
    It now supports an optional online search for courses.
    """
    try:
        profile = request.profile
        if not profile:
            raise ValueError("Profile is required")

        user_skills = set(skill.name for skill in profile.current_skills) if profile.current_skills else set()
        logger.debug(f"User skills extracted: {user_skills}")

        # For target skills, we'll derive them from the goal role and current skills
        # This is a simplified approach - in practice, you might want to have a more sophisticated mapping
        target_skills = set()  # Will be populated based on goal role analysis

        # Retrieve a larger set of initial candidates for re-ranking (5x the final top_k)
        initial_candidates = max(top_k * 5, 25)  # Ensure we have enough candidates
        query = RetrievalQuery(skills=list(user_skills), target_skills=list(target_skills))

        try:
            retrieved: List[Course] = await retriever.hybrid_search(query, top_k=initial_candidates)
            logger.debug(f"Retrieved {len(retrieved)} initial candidates")
        except Exception as e:
            logger.error(f"Failed to retrieve courses: {e}")
            retrieved = []

        # Construct query text from user profile for cross-encoder
        goal_role = profile.goal_role or "general professional development"

        # Create detailed skill context including expertise levels
        if profile.current_skills:
            current_skills_context = ", ".join([f"{skill.name} ({skill.expertise})" for skill in profile.current_skills])
        else:
            current_skills_context = "No specific skills listed"

        query_text = f"Goal role: {goal_role}. Current skills: {current_skills_context}. Looking for courses to advance toward {goal_role} role."

        # Initialize cross-encoder model
        cross_encoder = _get_cross_encoder()
        reranking_method = "skill-overlap-scoring"

        # Re-rank using cross-encoder if available, otherwise fall back to overlap scoring
        if cross_encoder and retrieved:
            try:
                top_courses = _rerank_with_cross_encoder(query_text, retrieved, cross_encoder, top_k)
                reranking_method = "cross-encoder"
            except Exception as e:
                logger.warning(f"Cross-encoder re-ranking failed, falling back to overlap scoring: {e}")
                cross_encoder = None

        if not cross_encoder or not retrieved:
            # Fallback to enhanced scoring based on user skills and goal role relevance
            scored: List[Tuple[Course, float]] = []
            for c in retrieved:
                try:
                    score = 0.0

                    # Score based on goal role relevance (highest priority)
                    goal_role_lower = goal_role.lower()
                    course_skills_lower = [skill.lower() for skill in c.skills]
                    course_title_lower = c.title.lower()

                    # Define goal role to skill mappings
                    goal_role_skills = {
                        'public speaker': ['public speaking', 'communication', 'presentation', 'storytelling', 'confidence'],
                        'data scientist': ['data science', 'machine learning', 'statistics', 'python', 'data analysis'],
                        'web developer': ['javascript', 'html', 'css', 'react', 'web development', 'frontend', 'backend'],
                        'software engineer': ['programming', 'software development', 'algorithms', 'data structures'],
                        'project manager': ['project management', 'leadership', 'communication', 'planning'],
                        'business analyst': ['business analysis', 'data analysis', 'communication', 'requirements gathering']
                    }

                    # Check for direct goal role match in course skills or title
                    if any(goal_word in course_title_lower or goal_word in ' '.join(course_skills_lower)
                           for goal_word in goal_role_lower.split()):
                        score += 2.0  # High bonus for direct goal role match

                    # Check for relevant skills based on goal role
                    if goal_role_lower in goal_role_skills:
                        relevant_skills = goal_role_skills[goal_role_lower]
                        skill_matches = sum(1 for skill in relevant_skills
                                          if any(skill in course_skill for course_skill in course_skills_lower))
                        score += skill_matches * 0.5

                    # Score based on user's current skills (lower priority)
                    if profile.current_skills:
                        user_skills_lower = [skill.name.lower() for skill in profile.current_skills]
                        skill_overlap = len(set(user_skills_lower) & set(course_skills_lower))
                        score += skill_overlap * 0.3

                    # Bonus for beginner courses if user has beginner skills
                    if profile.current_skills and any(skill.expertise.lower() == 'beginner' for skill in profile.current_skills):
                        if c.difficulty.lower() == 'beginner':
                            score += 0.2

                    # Default minimum score
                    if score == 0.0:
                        score = 0.1

                    scored.append((c, score))
                except Exception as e:
                    logger.warning(f"Error scoring course {c.course_id}: {e}")
                    scored.append((c, 0.0))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_courses = [c for c, s in scored[:top_k]] or (retrieved[:top_k] if retrieved else [])

        # Perform online search if requested or if local results are insufficient
        online_courses = []
        should_search_online = (
            getattr(request, 'search_online', False) or
            len(top_courses) < top_k or
            not any(skill.lower() in [s.lower() for course in top_courses for s in course.skills]
                   for skill in [goal_role.lower()] + [skill.name.lower() for skill in profile.current_skills or []])
        )

        if should_search_online:
            try:
                # Create search query based on user profile
                search_queries = []
                if goal_role:
                    search_queries.append(goal_role)
                if profile.current_skills:
                    # Add skills that might need advancement
                    search_queries.extend([skill.name for skill in profile.current_skills[:3]])

                # Default to goal role or "programming" if no specific queries
                primary_query = search_queries[0] if search_queries else "programming"

                logger.info(f"Performing online search for courses with query: {primary_query}")
                crawled_data = crawl_courses(query=primary_query, max_courses=15)

                existing_titles = {c.title.lower() for c in top_courses}
                existing_course_ids = {c.course_id for c in top_courses}

                new_courses_to_add = []

                for course_data in crawled_data:
                    try:
                        title = course_data.get('title', '').strip()
                        if not title or title.lower() in existing_titles:
                            continue

                        # Generate a more robust course ID
                        course_id = f"online-{course_data.get('provider', 'unknown').lower()}-{abs(hash(title)) % 10000:04d}"

                        # Skip if course ID already exists
                        if course_id in existing_course_ids:
                            continue

                        # Create Course object with enhanced data
                        new_course = Course(
                            course_id=course_id,
                            title=title,
                            skills=course_data.get('skills', []),
                            difficulty=course_data.get('difficulty', 'intermediate'),
                            duration_weeks=course_data.get('duration_weeks', 4),
                            provider=course_data.get('provider', 'Online'),
                            url=course_data.get('url'),
                            metadata={
                                'description': course_data.get('description', ''),
                                'source': course_data.get('source', 'Online'),
                                'search_query': course_data.get('search_query', primary_query),
                                'crawled_at': __import__('datetime').datetime.now().isoformat(),
                                'auto_added': True
                            }
                        )

                        online_courses.append(new_course)
                        new_courses_to_add.append(new_course)
                        existing_titles.add(title.lower())
                        existing_course_ids.add(course_id)

                        logger.debug(f"Added online course: {title} from {course_data.get('provider')}")

                    except Exception as e:
                        logger.warning(f"Error processing crawled course data: {e}")
                        continue

                # Automatically add new courses to courses.json
                if new_courses_to_add:
                    await _add_courses_to_json(new_courses_to_add)
                    logger.info(f"Added {len(new_courses_to_add)} new courses to courses.json")

            except Exception as e:
                logger.error(f"Online course search failed: {e}")
                online_courses = []

        # Combine and deduplicate course lists
        all_courses = top_courses + online_courses
        final_courses = []
        seen_ids = set()
        for course in all_courses:
            if course.course_id not in seen_ids:
                final_courses.append(course)
                seen_ids.add(course.course_id)

        top_courses = final_courses[:top_k]

        fallback_gap_map = {goal_role: ["Skills needed for this role will be identified through course recommendations"]}

        # Prioritize LLM-generated plan with enhanced context
        years_experience = profile.years_experience

        # Convert current skills to dict format for LLM
        current_skills_dict = [{"name": skill.name, "expertise": skill.expertise} for skill in profile.current_skills] if profile.current_skills else []

        try:
            llm_result = await _make_plan_llm(
                current_skills_dict,
                goal_role,
                top_courses,
                years_experience=years_experience
            )
        except Exception as e:
            logger.warning(f"LLM plan generation failed: {e}")
            llm_result = None

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
            try:
                plan = await _make_plan_heuristic(fallback_gap_map.keys(), top_courses)
            except Exception as e:
                logger.warning(f"Heuristic plan generation failed: {e}")
                plan = []

            gap_map = fallback_gap_map
            notes = f"Plan generated via heuristic with {reranking_method} re-ranking; consider configuring OpenRouter API key for richer guidance."
            logger.info("Using heuristic fallback plan")

        if getattr(request, 'search_online', False):
            notes += " Included results from an online search."

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

    except Exception as e:
        logger.error(f"Critical error in advisor service: {e}")
        # Return a minimal fallback result
        fallback_notes = f"An error occurred during processing: {str(e)}. Returning minimal recommendations."
        return AdviseResult(
            plan=[],
            gap_map={"error": ["Unable to generate detailed gap analysis due to processing error"]},
            recommended_courses=[],
            notes=fallback_notes
        )
