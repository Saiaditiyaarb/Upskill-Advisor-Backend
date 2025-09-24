"""
API contract schemas for versioned endpoints.

Future-proofing notes:
- AdviseRequest keeps `profile` and `user_context` as flexible Dict[str, Any] to allow evolution without breaking changes.
- AdviseResult uses broadly-typed fields to accommodate future additions.
- ApiResponse is a generic wrapper model so different endpoints can return consistent envelopes while varying `data` types.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from .course import Course


class SkillDetail(BaseModel):
    name: str = Field(description="Name of the skill")
    expertise: Literal["Beginner", "Intermediate", "Advanced"] = Field(description="Level of expertise in this skill")


class UserProfile(BaseModel):
    current_skills: List[SkillDetail] = Field(default_factory=list, description="Current user skills with expertise levels")
    goal_role: str = Field(description="Target job role or career goal")
    years_experience: Optional[int] = Field(default=None, ge=0, description="Years of professional experience")


class AdviseRequest(BaseModel):
    profile: UserProfile = Field(description="Typed user profile with skills and targets")
    user_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context like preferences, time constraints, prior courses"
    )


class AdviseResult(BaseModel):
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended learning plan steps")
    gap_map: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of target skills to missing sub-skills")
    recommended_courses: List[Course] = Field(default_factory=list, description="Top recommended courses")
    notes: Optional[str] = Field(default=None, description="Additional notes or rationale for recommendations")


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic response wrapper to stabilize external API while allowing inner schema evolution.

    Always return this envelope so clients can rely on `request_id` and `status`, irrespective of changes in `data`.
    """
    request_id: str
    status: str
    data: T
