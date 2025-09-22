"""
Course schema definition for future-proof data ingestion and retrieval.

Design choices supporting future-proofing:
- Includes a flexible `metadata` dictionary to store arbitrary attributes without schema migrations.
- Keeps core fields minimal and stable while allowing the `metadata` to evolve.
- Uses typing primitives only (no complex unions) to keep backward compatibility easier across pydantic versions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class Course(BaseModel):
    course_id: str
    title: str
    skills: List[str] = Field(default_factory=list, description="Key skills covered by the course")
    difficulty: str = Field(description="Difficulty level such as beginner, intermediate, advanced")
    duration_weeks: int = Field(ge=0, description="Estimated duration in weeks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary attributes for future expansion")

    # Optional convenience fields that can be embedded into metadata later if necessary
    provider: Optional[str] = None
    url: Optional[str] = None

    # Pydantic v2-style configuration for forward-compatibility
    model_config = ConfigDict(extra="ignore")
