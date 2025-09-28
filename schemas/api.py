"""
API contract schemas for versioned endpoints.

Future-proofing notes:
- AdviseRequest keeps `profile` and `user_context` as flexible Dict[str, Any] to allow evolution without breaking changes.
- AdviseResult uses broadly-typed fields to accommodate future additions.
- ApiResponse is a generic wrapper model so different endpoints can return consistent envelopes while varying `data` types.
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from .course import Course


class SkillDetail(BaseModel):
    name: str = Field(description="Name of the skill", min_length=1, max_length=100)
    expertise: Literal["Beginner", "Intermediate", "Advanced"] = Field(description="Level of expertise in this skill")

    @field_validator('name')
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        """Validate and clean skill name."""
        if not v or not v.strip():
            raise ValueError("Skill name cannot be empty")

        # Clean and normalize the skill name
        cleaned = v.strip()
        if len(cleaned) < 1:
            raise ValueError("Skill name must contain at least one non-whitespace character")

        return cleaned


# Backward-compatible alias for test scripts and older code
UserSkill = SkillDetail


class UserProfile(BaseModel):
    current_skills: List[SkillDetail] = Field(default_factory=list, description="Current user skills with expertise levels", max_items=50)
    goal_role: str = Field(description="Target job role or career goal", min_length=1, max_length=200)
    years_experience: Optional[int] = Field(default=None, ge=0, le=70, description="Years of professional experience")

    @field_validator('goal_role')
    @classmethod
    def validate_goal_role(cls, v: str) -> str:
        """Validate and clean goal role."""
        if not v or not v.strip():
            raise ValueError("Goal role cannot be empty")

        cleaned = v.strip()
        if len(cleaned) < 1:
            raise ValueError("Goal role must contain at least one non-whitespace character")

        return cleaned

    @field_validator('current_skills')
    @classmethod
    def validate_unique_skills(cls, v: List[SkillDetail]) -> List[SkillDetail]:
        """Ensure skill names are unique (case-insensitive)."""
        if not v:
            return v

        seen_skills = set()
        unique_skills = []

        for skill in v:
            skill_name_lower = skill.name.lower()
            if skill_name_lower not in seen_skills:
                seen_skills.add(skill_name_lower)
                unique_skills.append(skill)

        if len(unique_skills) != len(v):
            # Log warning but don't fail - just deduplicate
            pass

        return unique_skills

    @model_validator(mode='after')
    def validate_profile_completeness(self) -> 'UserProfile':
        """Validate overall profile completeness."""
        if not self.goal_role:
            raise ValueError("Goal role is required")

        # Optional: warn if no skills provided but don't fail
        if not self.current_skills:
            # This is allowed but might result in less personalized recommendations
            pass

        return self


class AdviseRequest(BaseModel):
    profile: UserProfile = Field(description="Typed user profile with skills and targets")
    user_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context like preferences, time constraints, prior courses"
    )
    search_online: Optional[bool] = Field(default=True, description="Whether to include online course search")
    retrieval_mode: Literal["vector", "keyword", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid", description="Retrieval mode for ablation studies"
    )
    target_skills: Optional[List[str]] = Field(default=None, description="Explicit target skills extracted from JD")
    generate_pdf: Optional[bool] = Field(default=False, description="Generate a PDF plan and save to reports/")

    @field_validator('user_context')
    @classmethod
    def validate_user_context(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate user context doesn't contain sensitive data."""
        if v is None:
            return v

        # Limit the size of user context to prevent abuse
        if len(str(v)) > 10000:  # 10KB limit
            raise ValueError("User context is too large")

        return v

    @field_validator('target_skills')
    @classmethod
    def validate_target_skills(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        cleaned = [s.strip() for s in v if s and s.strip()]
        # Deduplicate case-insensitively while preserving order
        seen = set()
        out = []
        for s in cleaned:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @model_validator(mode='after')
    def validate_request(self) -> 'AdviseRequest':
        """Validate the complete request."""
        if not self.profile:
            raise ValueError("Profile is required")

        return self


class AdviseResult(BaseModel):
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended learning plan steps")
    gap_map: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of target skills to missing sub-skills")
    recommended_courses: List[Course] = Field(default_factory=list, description="Top recommended courses")
    notes: Optional[str] = Field(default=None, description="Additional notes or rationale for recommendations")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Per-run metrics for ablation/compare endpoints")
    alternative_plan: Optional[AdviseResult] = Field(default=None, description="Alternative plan to showcase trade-offs (e.g., different courses or shorter duration)")


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic response wrapper to stabilize external API while allowing inner schema evolution.

    Always return this envelope so clients can rely on `request_id` and `status`, irrespective of changes in `data`.
    """
    request_id: str
    status: str
    data: T


class DemoPersonaRequest(BaseModel):
    """Request model for the centralized demo endpoint.

    - persona: a short key identifying the demo persona (e.g., "qa_to_sdet").
    - override: optional full AdviseRequest to use instead of built-in persona mapping.
    """
    persona: str = Field(description="Demo persona key (e.g., 'qa_to_sdet')", min_length=1, max_length=100)
    override: Optional[AdviseRequest] = Field(default=None, description="Optional AdviseRequest overriding the persona mapping")


class DemoPersonaResponse(BaseModel):
    """Aggregated payload for the demo persona endpoint.

    Includes a primary plan (with possible alternative), ablation/compare results,
    and parsed historical metrics reports for the frontend to render in one go.
    """
    persona: str
    primary: AdviseResult
    ablation_results: List[AdviseResult]
    reports: Dict[str, Any]
