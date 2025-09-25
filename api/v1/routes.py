"""
Versioned API v1 routes.

Design choices:
- The router does not hardcode a version prefix; main.py mounts it using settings.api_v1_prefix. This allows changing the prefix centrally.
- Responses are wrapped in the generic ApiResponse to keep a stable envelope while inner data evolves.
"""
from __future__ import annotations

from uuid import uuid4
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from schemas.api import AdviseRequest, AdviseResult, ApiResponse
from schemas.course import Course
from services.advisor_service import advise
from services.retriever import Retriever, get_retriever
from services.course_manager import CourseManager
from core.logging_config import set_request_id
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import BackgroundTasks, Query

# Optional caching decorator; no-op if fastapi-cache2 is not installed
try:
    from fastapi_cache.decorator import cache  # type: ignore
except Exception:  # pragma: no cover
    def cache(expire: int = 60):  # type: ignore
        def _wrap(func):
            return func
        return _wrap

router = APIRouter(tags=["advisor"])  # mounted under /api/v1 by main.py
logger = logging.getLogger("api")

# Initialize course manager
course_manager = CourseManager()


# Request/Response models for new endpoints
class CourseSearchRequest(BaseModel):
    query: Optional[str] = ""
    providers: Optional[List[str]] = None
    difficulties: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_free: Optional[bool] = None


class CourseUpdateRequest(BaseModel):
    search_queries: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    limit_per_query: Optional[int] = 50


class RecommendationRequest(BaseModel):
    user_skills: List[str]
    user_interests: Optional[List[str]] = None
    difficulty_preference: Optional[str] = None
    limit: Optional[int] = 10


@router.post("/advise", response_model=ApiResponse[AdviseResult])
@cache(expire=60)
async def post_advise(request: AdviseRequest, retriever: Retriever = Depends(get_retriever)) -> ApiResponse[AdviseResult]:
    """Return course recommendations and a learning plan based on the user's profile.

    The response is wrapped in ApiResponse to future-proof client integrations by keeping
    request_id and status fields stable across versions.
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        # Validate request data
        if not request.profile:
            logger.error("advise_request_invalid", extra={"request_id": req_id, "error": "Missing profile"})
            raise HTTPException(status_code=400, detail="Profile is required")

        if not request.profile.goal_role:
            logger.error("advise_request_invalid", extra={"request_id": req_id, "error": "Missing goal_role"})
            raise HTTPException(status_code=400, detail="Goal role is required")

        logger.info("advise_request_received", extra={
            "request_id": req_id,
            "goal_role": request.profile.goal_role,
            "current_skills_count": len(request.profile.current_skills) if request.profile.current_skills else 0,
            "years_experience": request.profile.years_experience
        })

        result = await advise(request, retriever)

        logger.info("advise_request_completed", extra={
            "request_id": req_id,
            "recommended_courses_count": len(result.recommended_courses) if result.recommended_courses else 0,
            "plan_steps": len(result.plan) if result.plan else 0
        })

        return ApiResponse[AdviseResult](request_id=req_id, status="ok", data=result)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("advise_request_failed", extra={
            "request_id": req_id,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request. Please try again later."
        )


# New course management endpoints
@router.get("/courses/search")
async def search_courses(
    query: Optional[str] = Query("", description="Search query for course titles, descriptions, and skills"),
    providers: Optional[List[str]] = Query(None, description="Filter by course providers"),
    difficulties: Optional[List[str]] = Query(None, description="Filter by difficulty levels"),
    skills: Optional[List[str]] = Query(None, description="Filter by required skills"),
    categories: Optional[List[str]] = Query(None, description="Filter by course categories"),
    is_free: Optional[bool] = Query(None, description="Filter by free/paid courses"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """
    Search and filter courses with advanced options
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        logger.info("course_search_request", extra={
            "request_id": req_id,
            "query": query,
            "providers": providers,
            "limit": limit,
            "offset": offset
        })

        # Build filters dictionary
        filters = {}
        if providers:
            filters['provider'] = providers
        if difficulties:
            filters['difficulty'] = difficulties
        if skills:
            filters['skills'] = skills
        if categories:
            filters['category'] = categories
        if is_free is not None:
            filters['is_free'] = is_free

        # Perform search
        results = course_manager.search_courses(
            query=query,
            filters=filters,
            limit=limit,
            offset=offset
        )

        logger.info("course_search_completed", extra={
            "request_id": req_id,
            "total_results": results.get('pagination', {}).get('total', 0),
            "returned_results": len(results.get('courses', []))
        })

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=results)

    except Exception as e:
        logger.error("course_search_failed", extra={
            "request_id": req_id,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/courses/search")
async def search_courses_post(request: CourseSearchRequest):
    """
    Search courses using POST request with complex filters
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        logger.info("course_search_post_request", extra={
            "request_id": req_id,
            "query": request.query,
            "providers": request.providers
        })

        # Build filters dictionary
        filters = {}
        if request.providers:
            filters['provider'] = request.providers
        if request.difficulties:
            filters['difficulty'] = request.difficulties
        if request.skills:
            filters['skills'] = request.skills
        if request.categories:
            filters['category'] = request.categories
        if request.is_free is not None:
            filters['is_free'] = request.is_free

        # Perform search
        results = course_manager.search_courses(
            query=request.query or "",
            filters=filters,
            limit=50,
            offset=0
        )

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=results)

    except Exception as e:
        logger.error("course_search_post_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/courses/{course_id}")
async def get_course(course_id: str):
    """
    Get a specific course by ID
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        logger.info("get_course_request", extra={
            "request_id": req_id,
            "course_id": course_id
        })

        course = course_manager.get_course_by_id(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        return ApiResponse[Course](request_id=req_id, status="ok", data=course)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_course_failed", extra={
            "request_id": req_id,
            "course_id": course_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve course: {str(e)}")


@router.get("/courses/stats")
async def get_course_statistics():
    """
    Get comprehensive statistics about the course database
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        logger.info("get_stats_request", extra={"request_id": req_id})

        stats = course_manager.get_statistics()

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=stats)

    except Exception as e:
        logger.error("get_stats_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.post("/courses/recommend")
async def recommend_courses(request: RecommendationRequest):
    """
    Get course recommendations based on user profile
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        logger.info("recommend_courses_request", extra={
            "request_id": req_id,
            "user_skills": request.user_skills,
            "difficulty_preference": request.difficulty_preference
        })

        recommendations = course_manager.recommend_courses(
            user_skills=request.user_skills,
            user_interests=request.user_interests,
            difficulty_preference=request.difficulty_preference,
            limit=request.limit or 10
        )

        result = {
            "recommendations": recommendations,
            "user_profile": {
                "skills": request.user_skills,
                "interests": request.user_interests,
                "difficulty_preference": request.difficulty_preference
            },
            "total_recommendations": len(recommendations)
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("recommend_courses_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/courses/update")
async def update_courses(background_tasks: BackgroundTasks, request: CourseUpdateRequest = None):
    """
    Trigger course database update by scraping new data
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        if request is None:
            request = CourseUpdateRequest()

        logger.info("course_update_request", extra={
            "request_id": req_id,
            "search_queries": request.search_queries,
            "platforms": request.platforms
        })

        # Run update in background
        background_tasks.add_task(
            course_manager.update_courses,
            search_queries=request.search_queries,
            platforms=request.platforms,
            limit_per_query=request.limit_per_query or 50
        )

        result = {
            "message": "Course update started in background",
            "status": "initiated",
            "parameters": {
                "search_queries": request.search_queries,
                "platforms": request.platforms,
                "limit_per_query": request.limit_per_query
            }
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("course_update_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to start course update: {str(e)}")


@router.get("/courses/providers")
async def get_providers():
    """
    Get list of available course providers
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        stats = course_manager.get_statistics()
        providers = list(stats.get('providers', {}).keys())

        result = {
            "providers": providers,
            "total_providers": len(providers)
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("get_providers_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve providers: {str(e)}")


@router.get("/courses/categories")
async def get_categories():
    """
    Get list of available course categories
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        stats = course_manager.get_statistics()
        categories = list(stats.get('categories', {}).keys())

        result = {
            "categories": categories,
            "total_categories": len(categories)
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("get_categories_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve categories: {str(e)}")


@router.get("/courses/skills")
async def get_skills(limit: int = Query(50, ge=1, le=200, description="Number of top skills to return")):
    """
    Get list of most popular skills from courses
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        stats = course_manager.get_statistics()
        top_skills = stats.get('top_skills', {})

        # Get top N skills
        skills_list = list(top_skills.items())[:limit]

        result = {
            "skills": [{"skill": skill, "course_count": count} for skill, count in skills_list],
            "total_unique_skills": len(top_skills)
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("get_skills_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve skills: {str(e)}")


@router.get("/courses/difficulties")
async def get_difficulties():
    """
    Get list of available difficulty levels
    """
    req_id = str(uuid4())
    set_request_id(req_id)

    try:
        stats = course_manager.get_statistics()
        difficulties = list(stats.get('difficulties', {}).keys())

        result = {
            "difficulties": difficulties,
            "total_difficulties": len(difficulties)
        }

        return ApiResponse[Dict[str, Any]](request_id=req_id, status="ok", data=result)

    except Exception as e:
        logger.error("get_difficulties_failed", extra={
            "request_id": req_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve difficulties: {str(e)}")