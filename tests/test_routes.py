from fastapi.testclient import TestClient

from main import app
from schemas.api import AdviseRequest, UserProfile
from services.retriever import get_retriever
from schemas.course import Course


class FakeRetriever:
    def __init__(self, courses):
        self._courses = courses

    async def hybrid_search(self, query, top_k: int = 5):
        return self._courses[:top_k]


def test_post_advise_endpoint_contract():
    # Override retriever dependency
    courses = [
        Course(course_id="c1", title="Data Engineering", skills=["python", "spark"], difficulty="intermediate", duration_weeks=6),
        Course(course_id="c2", title="Deep Learning", skills=["ml", "python"], difficulty="advanced", duration_weeks=10),
    ]

    def _override_get_retriever():
        return FakeRetriever(courses)

    app.dependency_overrides[get_retriever] = _override_get_retriever

    client = TestClient(app)

    payload = {
        "profile": {
            "skills": ["sql"],
            "target_skills": ["python"],
            "years_experience": 3
        }
    }

    resp = client.post("/api/v1/advise", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    # Envelope fields
    assert "request_id" in body and isinstance(body["request_id"], str)
    assert body["status"] == "ok"
    assert "data" in body

    data = body["data"]
    assert "plan" in data
    assert "gap_map" in data
    assert "recommended_courses" in data

    # Ensure recommended courses are serialized
    assert isinstance(data["recommended_courses"], list)
    if data["recommended_courses"]:
        first = data["recommended_courses"][0]
        assert "course_id" in first and "title" in first

    # Clean up override
    app.dependency_overrides.clear()
