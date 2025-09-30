"""
Enhanced advisor service with local LLM support and offline capabilities.

This service provides:
- Local LLM integration for offline recommendations
- Skill mapping and timeline generation
- Performance optimization for sub-2.5s response times
- Real-time metrics and scoring explanations
"""
from __future__ import annotations

import logging
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from functools import lru_cache

from schemas.api import AdviseRequest, AdviseResult, UserProfile
from schemas.course import Course
from services.retriever import Retriever, RetrievalQuery
from services.local_llm import get_local_llm, LocalLLMChain, extract_json_from_text
from services.gemini_service import generate_learning_plan_with_fallback
from services.metrics_service import get_metrics_collector, ComponentType
from core.config import get_settings

logger = logging.getLogger("enhanced_advisor")

# Simple in-memory cache for advisor results
_advisor_cache: Dict[str, Tuple[AdviseResult, float]] = {}
_cache_ttl = 300  # 5 minutes
_max_cache_size = 100


def _get_cache_key(request: AdviseRequest) -> str:
    """Generate cache key for advisor request."""
    # Create a deterministic string representation
    key_data = {
        "goal_role": request.profile.goal_role if request.profile else "",
        "skills": sorted([skill.name for skill in (request.profile.current_skills or [])]) if request.profile else [],
        "target_skills": sorted(request.target_skills or []),
        "years_experience": request.profile.years_experience if request.profile else 0
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cached_result(cache_key: str) -> Optional[AdviseResult]:
    """Get cached result if valid."""
    if cache_key in _advisor_cache:
        result, timestamp = _advisor_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            logger.info(f"Cache hit for advisor request: {cache_key[:8]}...")
            return result
        else:
            # Remove expired entry
            del _advisor_cache[cache_key]
    return None


def _cache_result(cache_key: str, result: AdviseResult) -> None:
    """Cache advisor result with size management."""
    # Remove oldest entries if cache is full
    if len(_advisor_cache) >= _max_cache_size:
        oldest_key = min(_advisor_cache.keys(), key=lambda k: _advisor_cache[k][1])
        del _advisor_cache[oldest_key]

    _advisor_cache[cache_key] = (result, time.time())
    logger.info(f"Cached advisor result: {cache_key[:8]}...")


async def _make_plan_with_optimized_llm(
    current_skills: List[Dict[str, str]],
    goal_role: str,
    courses: List[Course],
    years_experience: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Generate learning plan using Gemini 2.0 Flash with local LLM fallback."""

    try:
        # Prepare course information for the LLM
        course_data = []
        for c in courses[:10]:  # Limit to top 10 for performance
            course_data.append({
                "course_id": c.course_id,
                "title": c.title,
                "skills": c.skills,
                "difficulty": c.difficulty,
                "duration_weeks": c.duration_weeks,
                "provider": c.provider
            })

        # Use Gemini service with automatic fallback
        result = await generate_learning_plan_with_fallback(
            current_skills=current_skills,
            goal_role=goal_role,
            courses=course_data,
            years_experience=years_experience
        )

        if result and isinstance(result, dict):
            # Validate and fix structure
            if "plan" not in result:
                result["plan"] = []
            if "timeline" not in result:
                result["timeline"] = {"total_weeks": 12}
            if "gap_map" not in result:
                result["gap_map"] = {goal_role: ["Skills will be identified through course analysis"]}
            if "notes" not in result:
                result["notes"] = "Learning plan generated using AI assistant"

            logger.info("Optimized LLM plan generation successful")
            return result
        else:
            logger.warning("Optimized LLM returned invalid structure")
            return None

    except Exception as e:
        logger.error(f"Optimized LLM plan generation failed: {e}")
        return None


def _calculate_skill_map(user_skills: Set[str], target_skills: Set[str], courses: List[Course]) -> Dict[str, Any]:
    """Calculate comprehensive skill mapping with progress indicators."""
    
    # Calculate skill coverage
    covered_skills = set()
    skill_courses = {}
    
    for course in courses:
        course_skills = set(skill.lower() for skill in course.skills)
        for skill in course_skills:
            if skill not in skill_courses:
                skill_courses[skill] = []
            skill_courses[skill].append(course)
            
            # Check if this skill matches any target skills
            for target_skill in target_skills:
                if skill in target_skill.lower() or target_skill.lower() in skill:
                    covered_skills.add(target_skill)
    
    # Calculate skill gaps
    missing_skills = target_skills - covered_skills
    
    # Calculate progress percentages
    total_target_skills = len(target_skills) if target_skills else 1
    coverage_percentage = (len(covered_skills) / total_target_skills) * 100
    
    return {
        "covered_skills": list(covered_skills),
        "missing_skills": list(missing_skills),
        "coverage_percentage": round(coverage_percentage, 1),
        "skill_courses": {skill: [{"id": c.course_id, "title": c.title, "difficulty": c.difficulty} for c in courses] 
                         for skill, courses in skill_courses.items()},
        "total_skills_available": len(skill_courses)
    }


def _generate_timeline(plan: List[Dict[str, Any]], courses: List[Course]) -> Dict[str, Any]:
    """Generate detailed timeline with phases and milestones."""
    
    if not plan:
        return {"total_weeks": 0, "phases": [], "milestones": []}
    
    # Calculate total duration
    total_weeks = sum(step.get("estimated_weeks", 4) for step in plan)
    
    # Create phases based on plan steps
    phases = []
    current_week = 1
    
    for i, step in enumerate(plan):
        weeks = step.get("estimated_weeks", 4)
        phase_name = f"Phase {i + 1}"
        
        # Get course details if available
        course_id = step.get("course_id")
        course = next((c for c in courses if c.course_id == course_id), None)
        
        focus = step.get("why", "Skill development")
        if course:
            focus = f"Learn {course.title}"
        
        phases.append({
            "phase": phase_name,
            "weeks": f"{current_week}-{current_week + weeks - 1}",
            "focus": focus,
            "course_id": course_id,
            "difficulty": course.difficulty if course else "intermediate"
        })
        
        current_week += weeks
    
    # Generate milestones
    milestones = []
    milestone_weeks = [4, 8, 12, 16, 20]
    
    for week in milestone_weeks:
        if week <= total_weeks:
            # Find which phase this milestone falls in
            phase = next((p for p in phases if int(p["weeks"].split("-")[0]) <= week <= int(p["weeks"].split("-")[1])), None)
            if phase:
                milestones.append({
                    "week": week,
                    "description": f"Complete {phase['focus']}",
                    "phase": phase["phase"]
                })
    
    return {
        "total_weeks": total_weeks,
        "phases": phases,
        "milestones": milestones,
        "estimated_completion_date": f"{total_weeks} weeks from start"
    }


def _calculate_recommendation_scores(courses: List[Course], user_skills: Set[str], target_skills: Set[str]) -> List[Dict[str, Any]]:
    """Calculate detailed recommendation scores with explanations."""
    
    scored_courses = []
    
    for course in courses:
        # Skill match score (40% weight)
        course_skills = set(skill.lower() for skill in course.skills)
        skill_match_score = 0
        
        # Check overlap with user skills (prerequisite matching)
        user_overlap = len(course_skills & set(skill.lower() for skill in user_skills))
        skill_match_score += min(user_overlap * 10, 30)  # Max 30 points for prerequisites
        
        # Check overlap with target skills (goal alignment)
        target_overlap = len(course_skills & set(skill.lower() for skill in target_skills))
        skill_match_score += min(target_overlap * 15, 40)  # Max 40 points for goal alignment
        
        # Difficulty appropriateness (20% weight)
        difficulty_score = 0
        if course.difficulty:
            difficulty_lower = course.difficulty.lower()
            if difficulty_lower == "beginner" and len(user_skills) < 3:
                difficulty_score = 20
            elif difficulty_lower == "intermediate" and 3 <= len(user_skills) < 6:
                difficulty_score = 20
            elif difficulty_lower == "advanced" and len(user_skills) >= 6:
                difficulty_score = 20
            else:
                difficulty_score = 10  # Partial match
        
        # Duration score (15% weight) - prefer shorter courses
        duration_score = max(0, 15 - (course.duration_weeks or 8) * 1.5)
        
        # Provider reputation (15% weight)
        provider_score = 0
        if course.provider:
            reputable_providers = ["coursera", "edx", "udacity", "pluralsight", "linkedin learning"]
            if course.provider.lower() in reputable_providers:
                provider_score = 15
            else:
                provider_score = 8
        
        # Rating score (10% weight)
        rating_score = 0
        if course.metadata and course.metadata.get("rating"):
            rating = float(course.metadata["rating"])
            rating_score = min(rating * 2, 10)  # Convert 5-star to 10-point scale
        
        # Calculate total score
        total_score = skill_match_score + difficulty_score + duration_score + provider_score + rating_score
        
        # Generate explanation
        explanations = []
        if skill_match_score > 0:
            explanations.append(f"Skill alignment: {skill_match_score:.0f}/70 points")
        if difficulty_score > 0:
            explanations.append(f"Difficulty match: {difficulty_score:.0f}/20 points")
        if duration_score > 0:
            explanations.append(f"Duration: {duration_score:.0f}/15 points")
        if provider_score > 0:
            explanations.append(f"Provider reputation: {provider_score:.0f}/15 points")
        if rating_score > 0:
            explanations.append(f"Course rating: {rating_score:.0f}/10 points")
        
        scored_courses.append({
            "course": course,
            "total_score": round(total_score, 1),
            "skill_match_score": round(skill_match_score, 1),
            "difficulty_score": round(difficulty_score, 1),
            "duration_score": round(duration_score, 1),
            "provider_score": round(provider_score, 1),
            "rating_score": round(rating_score, 1),
            "explanations": explanations,
            "skill_match_percentage": round((skill_match_score / 70) * 100, 1)
        })
    
    # Sort by total score
    scored_courses.sort(key=lambda x: x["total_score"], reverse=True)
    return scored_courses


async def enhanced_advise(request: AdviseRequest, retriever: Retriever, top_k: int = 5) -> AdviseResult:
    """
    Enhanced advisor with local LLM support, skill mapping, and performance optimization.
    Optimized for sub-2.5s response times.
    """
    start_time = time.time()

    # Check cache first
    cache_key = _get_cache_key(request)
    cached_result = _get_cached_result(cache_key)
    if cached_result:
        # Update metrics for cache hit
        try:
            mc = get_metrics_collector()
            mc.record_latency(
                ComponentType.AGENT,
                operation="enhanced_advise_cache_hit",
                duration_ms=1,  # Very fast for cache hit
                success=True,
                retrieval_mode="cache",
                courses_analyzed=0,
                courses_selected=len(cached_result.recommended_courses or []),
                skill_map_generated=bool(cached_result.metrics and cached_result.metrics.get("skill_map"))
            )
        except Exception:
            pass  # Ignore metrics errors for cache hits

        return cached_result

    try:
        profile = request.profile
        if not profile:
            raise ValueError("Profile is required")

        user_skills = set(skill.name for skill in profile.current_skills) if profile.current_skills else set()
        target_skills = set((request.target_skills or []))
        goal_role = profile.goal_role or "general professional development"
        
        logger.info(f"Enhanced advisor started for goal: {goal_role}")

        # Fast retrieval with performance optimization
        query = RetrievalQuery(skills=list(user_skills), target_skills=list(target_skills), text=goal_role)
        
        # Use hybrid search for best results - limit candidates to reduce processing time
        retrieved = await retriever.hybrid_search(query, top_k=min(top_k + 3, 12))  # Further reduced for speed
        
        if not retrieved:
            logger.warning("No courses retrieved")
            return AdviseResult(
                plan=[],
                gap_map={goal_role: ["No courses available for analysis"]},
                recommended_courses=[],
                notes="No courses found matching your criteria"
            )

        # Calculate detailed recommendation scores
        scored_courses = _calculate_recommendation_scores(retrieved, user_skills, target_skills)
        top_courses = [item["course"] for item in scored_courses[:top_k]]

        # Try optimized LLM first for plan generation
        years_experience = profile.years_experience
        current_skills_dict = [{"name": skill.name, "expertise": skill.expertise} for skill in profile.current_skills] if profile.current_skills else []

        llm_result = await _make_plan_with_optimized_llm(
            current_skills_dict, goal_role, top_courses, years_experience
        )

        # Generate plan
        if llm_result and isinstance(llm_result, dict):
            plan = llm_result.get("plan", [])
            gap_map = llm_result.get("gap_map", {goal_role: ["Skills will be identified through course analysis"]})
            timeline = llm_result.get("timeline", {"total_weeks": 12})
            notes = llm_result.get("notes", "Learning plan generated using local LLM")
        else:
            # Fallback to heuristic plan with better structure
            plan = []
            for i, course in enumerate(top_courses[:min(3, len(top_courses))]):  # Limit to top 3 for performance
                plan.append({
                    "course_id": course.course_id,
                    "skill": course.skills[0] if course.skills else "General Skills",
                    "action": "LEARN",
                    "why": f"Essential for {goal_role} role - covers key skills: {', '.join(course.skills[:3]) if course.skills else 'general skills'}",
                    "order": i + 1,
                    "estimated_weeks": course.duration_weeks or 4
                })
            
            # Generate initial gap map - will be enhanced later
            gap_map = {goal_role: ["Skills will be identified through course analysis"]}
            timeline = {"total_weeks": sum(step.get("estimated_weeks", 4) for step in plan)}
            notes = "Fast plan generated for immediate response"

        # Generate skill map and enhanced gap analysis (optimized)
        skill_map = _generate_skill_map(top_courses, goal_role, user_skills)
        enhanced_gap_map = _generate_detailed_gap_map(top_courses, goal_role, skill_map, user_skills)
        
        # Generate comprehensive notes based on skill analysis
        if not notes or notes == "Fast plan generated for immediate response":
            notes = _generate_comprehensive_notes(plan, top_courses, skill_map, goal_role, enhanced_gap_map)
        else:
            # Enhance existing notes with additional insights
            additional_insights = _generate_additional_insights(plan, top_courses, skill_map, goal_role)
            if additional_insights:
                notes += "\n\n" + additional_insights
        
        # Generate enhanced timeline
        enhanced_timeline = _generate_timeline(plan, top_courses)

        # Calculate performance metrics
        processing_time = time.time() - start_time
        
        # Record comprehensive metrics
        try:
            mc = get_metrics_collector()
            
            # Record latency
            mc.record_latency(
                ComponentType.AGENT,
                operation="enhanced_advise",
                duration_ms=processing_time * 1000,
                success=True,
                retrieval_mode="hybrid",
                courses_analyzed=len(retrieved),
                courses_selected=len(top_courses),
                skill_map_generated=bool(skill_map)
            )
            
            # Record cost (estimate based on processing time and complexity)
            estimated_cost = 0.001 + (processing_time * 0.0005)  # Base cost + time-based cost
            mc.record_cost(
                ComponentType.AGENT,
                operation="enhanced_advise",
                cost_usd=estimated_cost,
                tokens_used=int(processing_time * 100),  # Estimate tokens
                model_name="local-llm",
                retrieval_mode="hybrid",
                courses_analyzed=len(retrieved)
            )
            
            # Record accuracy metrics
            if skill_map and skill_map.get("coverage_percentage"):
                mc.record_accuracy(
                    ComponentType.AGENT,
                    operation="skill_coverage",
                    accuracy_score=skill_map["coverage_percentage"] / 100.0,
                    total_items=skill_map.get("total_skills_available", 0),
                    correct_items=int(skill_map.get("coverage_percentage", 0) * skill_map.get("total_skills_available", 0) / 100),
                    retrieval_mode="hybrid"
                )
            
            # Flush metrics immediately to ensure they're available for the frontend
            await mc.flush_metrics()
            logger.info(f"Metrics recorded and flushed: latency={processing_time*1000:.2f}ms, cost=${estimated_cost:.4f}")
                
        except Exception as e:
            logger.warning(f"Metrics recording failed: {e}")

        # Add detailed metrics to result
        detailed_metrics = {
            "processing_time_ms": round(processing_time * 1000, 2),
            "skill_map": skill_map,
            "recommendation_scores": scored_courses[:top_k],
            "timeline": enhanced_timeline,
            "performance_target_met": processing_time < 2.5,
            "courses_analyzed": len(retrieved),
            "courses_selected": len(top_courses)
        }

        logger.info(f"Enhanced advisor completed in {processing_time:.2f}s")

        result = AdviseResult(
            plan=plan,
            gap_map=enhanced_gap_map,
            recommended_courses=top_courses,
            notes=notes,
            timeline=enhanced_timeline,
            metrics=detailed_metrics
        )

        # Cache the result
        _cache_result(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Enhanced advisor failed: {e}")
        processing_time = time.time() - start_time
        
        return AdviseResult(
            plan=[],
            gap_map={"error": ["Unable to generate recommendations due to processing error"]},
            recommended_courses=[],
            notes=f"Error occurred: {str(e)}. Processing time: {processing_time:.2f}s"
        )


def _generate_skill_map(courses: List[Course], goal_role: str, user_skills: Set[str] = None) -> Dict[str, Any]:
    """Generate comprehensive skill map from courses."""
    if not courses:
        return {
            "covered_skills": [],
            "missing_skills": [],
            "coverage_percentage": 0.0,
            "skill_courses": {},
            "total_skills_available": 0
        }
    
    # Collect all skills from courses
    all_skills = set()
    skill_courses = {}
    
    for course in courses:
        for skill in course.skills:
            skill_lower = skill.lower()
            all_skills.add(skill_lower)
            if skill_lower not in skill_courses:
                skill_courses[skill_lower] = []
            skill_courses[skill_lower].append({
                "id": course.course_id,
                "title": course.title,
                "difficulty": course.difficulty
            })
    
    # Generate target skills based on goal role
    target_skills = _get_target_skills_for_role(goal_role)
    target_skills_lower = {skill.lower() for skill in target_skills}
    
    # Identify skills user already has
    user_skills_lower = {skill.lower() for skill in (user_skills or set())}
    
    # Calculate coverage - only count skills from courses that user doesn't already have
    covered_skills = list(all_skills & target_skills_lower)
    missing_skills = list(target_skills_lower - all_skills - user_skills_lower)
    
    # Coverage includes user's existing skills
    total_needed_skills = len(target_skills_lower)
    covered_by_courses = len(covered_skills)
    covered_by_user = len(user_skills_lower & target_skills_lower)
    
    coverage_percentage = ((covered_by_courses + covered_by_user) / total_needed_skills * 100) if total_needed_skills else 0.0
    
    return {
        "covered_skills": covered_skills,
        "missing_skills": missing_skills,
        "user_existing_skills": list(user_skills_lower & target_skills_lower),
        "coverage_percentage": round(coverage_percentage, 1),
        "skill_courses": skill_courses,
        "total_skills_available": total_needed_skills
    }


def _generate_detailed_gap_map(courses: List[Course], goal_role: str, skill_map: Dict[str, Any], user_skills: Set[str] = None) -> Dict[str, List[str]]:
    """Generate detailed skill gap analysis based on goal role, courses, and user skills."""
    gap_map = {}
    
    # Get target skills for the role
    target_skills = _get_target_skills_for_role(goal_role)
    target_skills_lower = {skill.lower() for skill in target_skills}
    
    # Get skills covered by courses
    covered_skills_lower = {skill.lower() for skill in skill_map.get("covered_skills", [])}
    
    # Get user's existing skills
    user_skills_lower = {skill.lower() for skill in (user_skills or set())}
    
    # Calculate actual skill gaps: skills needed for role - (user skills + course skills)
    actual_gaps = target_skills_lower - covered_skills_lower - user_skills_lower
    
    if actual_gaps:
        gap_map[f"{goal_role} - Skills to Develop"] = list(actual_gaps)
    
    # Add category-specific gap analysis based on goal role
    role_lower = goal_role.lower()
    
    if any(keyword in role_lower for keyword in ["engineer", "developer", "programmer"]):
        tech_skills = {"algorithms", "data structures", "design patterns", "testing", "debugging", "version control"}
        tech_gaps = tech_skills - user_skills_lower - covered_skills_lower
        if tech_gaps:
            gap_map["Technical Fundamentals"] = list(tech_gaps)
    
    if any(keyword in role_lower for keyword in ["data", "scientist", "analyst", "ml", "ai"]):
        data_skills = {"statistics", "machine learning", "data visualization", "sql", "python", "r"}
        data_gaps = data_skills - user_skills_lower - covered_skills_lower
        if data_gaps:
            gap_map["Data Science Skills"] = list(data_gaps)
    
    if any(keyword in role_lower for keyword in ["manager", "lead", "product", "project"]):
        mgmt_skills = {"leadership", "communication", "project management", "stakeholder management", "strategy"}
        mgmt_gaps = mgmt_skills - user_skills_lower - covered_skills_lower
        if mgmt_gaps:
            gap_map["Management Skills"] = list(mgmt_gaps)
    
    if any(keyword in role_lower for keyword in ["devops", "cloud", "infrastructure"]):
        devops_skills = {"docker", "kubernetes", "ci/cd", "aws", "monitoring", "automation"}
        devops_gaps = devops_skills - user_skills_lower - covered_skills_lower
        if devops_gaps:
            gap_map["DevOps & Cloud Skills"] = list(devops_gaps)
    
    # If no specific gaps identified, provide encouragement
    if not gap_map:
        gap_map["Next Steps"] = [
            "Continue building expertise in current skill areas",
            "Explore advanced topics and specializations",
            "Gain practical experience through projects",
            "Stay updated with industry trends and technologies"
        ]
    
    return gap_map


def _get_target_skills_for_role(goal_role: str) -> List[str]:
    """Get target skills based on role."""
    role_skills_map = {
        "software engineer": [
            "programming", "algorithms", "data structures", "software development",
            "testing", "debugging", "version control", "agile", "scrum"
        ],
        "data scientist": [
            "python", "r", "statistics", "machine learning", "data analysis",
            "sql", "pandas", "numpy", "visualization", "deep learning"
        ],
        "product manager": [
            "product strategy", "market research", "user experience", "agile",
            "project management", "stakeholder management", "analytics", "roadmapping"
        ],
        "devops engineer": [
            "docker", "kubernetes", "ci/cd", "aws", "azure", "monitoring",
            "infrastructure", "automation", "security", "cloud computing"
        ],
        "frontend developer": [
            "javascript", "react", "vue", "angular", "html", "css", "responsive design",
            "web development", "ui/ux", "typescript"
        ],
        "backend developer": [
            "python", "java", "node.js", "database", "api", "microservices",
            "rest", "graphql", "caching", "security"
        ],
        "automation test": [
            "selenium", "pytest", "test automation", "python", "java", "test frameworks",
            "ci/cd", "jenkins", "git", "api testing", "web automation", "mobile testing"
        ],
        "qa engineer": [
            "testing", "automation", "selenium", "test cases", "bug tracking",
            "quality assurance", "test planning", "regression testing", "performance testing"
        ],
        "sdet": [
            "test automation", "programming", "selenium", "api testing", "ci/cd",
            "test frameworks", "python", "java", "rest api", "jenkins"
        ]
    }
    
    goal_role_lower = goal_role.lower()
    
    # Check for exact and partial matches
    for role, skills in role_skills_map.items():
        if role in goal_role_lower or any(word in goal_role_lower for word in role.split()):
            return skills
    
    # Check for common keywords
    if "test" in goal_role_lower or "qa" in goal_role_lower:
        return role_skills_map.get("automation test", [])
    
    # Default skills for any role (last resort)
    return [
        "problem solving", "communication", "teamwork", "analytical thinking",
        "attention to detail", "time management", "adaptability", "learning agility"
    ]


def _generate_comprehensive_notes(plan: List[Dict], courses: List[Course], skill_map: Dict[str, Any], 
                                 goal_role: str, gap_map: Dict[str, List[str]]) -> str:
    """Generate comprehensive learning notes without emojis."""
    notes_parts = []
    
    # Overview
    notes_parts.append(f"LEARNING PLAN FOR {goal_role.upper()}")
    notes_parts.append("="*50)
    notes_parts.append("")
    
    # Skill coverage analysis
    coverage = skill_map.get("coverage_percentage", 0)
    total_skills = skill_map.get("total_skills_available", 0)
    covered_skills = len(skill_map.get("covered_skills", []))
    
    notes_parts.append("SKILL COVERAGE ANALYSIS")
    notes_parts.append(f"  Coverage: {coverage}% ({covered_skills} of {total_skills} target skills)")
    notes_parts.append(f"  Courses analyzed: {len(courses)}")
    notes_parts.append(f"  Recommended courses: {len(plan)}")
    notes_parts.append("")
    
    # Learning path insights
    if plan:
        notes_parts.append("LEARNING PATH INSIGHTS")
        total_weeks = sum(step.get("estimated_weeks", 4) for step in plan)
        notes_parts.append(f"  Estimated duration: {total_weeks} weeks ({total_weeks//4} months)")
        notes_parts.append(f"  Approach: Structured progression from foundational to advanced concepts")
        notes_parts.append("")
    
    # Skill gaps
    if gap_map:
        notes_parts.append("KEY SKILL GAPS IDENTIFIED")
        for category, gaps in gap_map.items():
            if gaps and len(gaps) > 0:
                gap_list = ', '.join(gaps[:5])
                if len(gaps) > 5:
                    gap_list += f" (and {len(gaps)-5} more)"
                notes_parts.append(f"  {category}: {gap_list}")
        notes_parts.append("")
    
    # Recommendations
    notes_parts.append("RECOMMENDATIONS")
    notes_parts.append("  1. Focus on hands-on practice with real-world projects")
    notes_parts.append("  2. Join professional communities and participate in discussions")
    notes_parts.append("  3. Track your progress and adjust learning pace as needed")
    notes_parts.append("  4. Build a portfolio showcasing practical implementations")
    notes_parts.append("  5. Seek mentorship and code reviews from experienced professionals")
    notes_parts.append("")
    
    # Success metrics
    notes_parts.append("SUCCESS METRICS")
    notes_parts.append("  - Complete all courses with hands-on projects")
    notes_parts.append("  - Build 2-3 portfolio projects demonstrating skills")
    notes_parts.append("  - Contribute to open-source projects in your domain")
    notes_parts.append("  - Apply skills in work projects or freelance assignments")
    notes_parts.append("  - Stay updated with industry trends and best practices")
    
    return "\n".join(notes_parts)


def _generate_additional_insights(plan: List[Dict], courses: List[Course], skill_map: Dict[str, Any], goal_role: str) -> str:
    """Generate additional insights to enhance existing notes without emojis."""
    insights = []
    
    insights.append("\nADDITIONAL INSIGHTS")
    
    # Add skill coverage insights
    coverage = skill_map.get("coverage_percentage", 0)
    if coverage > 0:
        insights.append(f"  Skill Coverage: {coverage}% of target skills covered by recommended courses")
    
    # Add course diversity insights
    if courses:
        providers = set(course.provider for course in courses if course.provider)
        difficulties = set(course.difficulty for course in courses if course.difficulty)
        
        if len(providers) > 1:
            insights.append(f"  Provider Diversity: Courses from {len(providers)} different platforms")
        
        if len(difficulties) > 1:
            insights.append(f"  Difficulty Progression: {', '.join(sorted(difficulties))} level courses included")
    
    # Add learning path insights
    if plan:
        total_weeks = sum(step.get("estimated_weeks", 4) for step in plan)
        months = total_weeks // 4
        insights.append(f"  Time Investment: {total_weeks} weeks (~{months} months) total learning time")
    
    return "\n".join(insights) if len(insights) > 1 else ""
