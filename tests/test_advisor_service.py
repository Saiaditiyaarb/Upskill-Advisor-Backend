import asyncio
import pytest

from services.advisor_service import advise
from schemas.api import AdviseRequest, UserProfile
from schemas.course import Course


class FakeRetriever:
    def __init__(self, courses):
        self._courses = courses

    async def hybrid_search(self, query, top_k: int = 5):
        # return the provided courses regardless of query for testing
        return self._courses[:top_k]


@pytest.mark.asyncio
async def test_advise_heuristic_plan_and_ranking():
    courses = [
        Course(course_id="c1", title="Intro to Python", skills=["python", "basics"], difficulty="beginner", duration_weeks=4),
        Course(course_id="c2", title="Machine Learning 101", skills=["ml", "math"], difficulty="intermediate", duration_weeks=8),
    ]
    retriever = FakeRetriever(courses)
    req = AdviseRequest(profile=UserProfile(skills=["python"], target_skills=["ml"]))

    result = await advise(req, retriever, top_k=2)

    # Should recommend the ML course first due to overlap with target skill
    assert result.recommended_courses, "Expected at least one recommended course"
    assert result.recommended_courses[0].course_id == "c2"

    # Gap map should include the target skill 'ml' since user lacks it
    assert "ml" in result.gap_map

    # Plan should have at least one step and reference the ML skill
    assert result.plan, "Plan should not be empty"
    assert any(step.get("skill") == "ml" for step in result.plan)
