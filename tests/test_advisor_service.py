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
    req = AdviseRequest(profile=UserProfile(
        current_skills=[{"name": "Python", "expertise": "Beginner"}],
        goal_role="Machine Learning Engineer"
    ))

    result = await advise(req, retriever, top_k=2)

    # Should recommend courses relevant to the goal role
    assert result.recommended_courses, "Expected at least one recommended course"

    # Gap map should include relevant skills for the goal role
    assert result.gap_map, "Gap map should not be empty"
    # The LLM should identify relevant skills like "Machine Learning" or "Python"
    gap_map_keys = list(result.gap_map.keys())
    assert any("Machine Learning" in key or "Python" in key for key in gap_map_keys), f"Expected ML or Python related skills in gap map, got: {gap_map_keys}"

    # Plan should have at least one step
    assert result.plan, "Plan should not be empty"

    # Should have notes explaining the approach
    assert result.notes, "Notes should be provided"
