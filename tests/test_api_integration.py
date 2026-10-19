"""Integration tests for API routes wired to the real scoring engine.

The optional sentence-transformers package is not installed by runtime requirements,
so ScoringEngine should exercise its deterministic TF-IDF fallback in CI.
"""
import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_engine", None)
    return TestClient(api.app)


def test_real_score_route_runs_scoring_engine(client):
    response = client.post(
        "/score",
        json={
            "candidate_name": "Integration Candidate",
            "resume_text": (
                "Software Engineer. Bachelor of Technology. Python, FastAPI, SQL. "
                "Built REST APIs and automated tests for backend services."
            ),
            "jd_text": (
                "Software Engineer with Python, FastAPI, SQL, REST API development, "
                "and Bachelor degree requirements."
            ),
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["candidate_name"] == "Integration Candidate"
    assert 0 <= result["final_score"] <= 100
    assert 0 <= result["embedding_similarity"] <= 1
    assert 0 <= result["keyword_coverage"] <= 1
    assert isinstance(result["matched_skills"], list)
    assert isinstance(result["missing_skills"], list)
    assert isinstance(result["insights"], dict)


def test_real_rank_route_scores_and_orders_candidates(client):
    response = client.post(
        "/rank",
        json={
            "jd_text": "Python FastAPI SQL REST API developer",
            "resumes": [
                {
                    "candidate_name": "Relevant",
                    "resume_text": "Python FastAPI SQL REST API developer",
                },
                {
                    "candidate_name": "Less Relevant",
                    "resume_text": "Graphic design, illustration, and brand identity",
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    ranked = response.json()["ranked"]
    assert len(ranked) == 2
    assert [item["rank"] for item in ranked] == [1, 2]
    assert ranked[0]["candidate_name"] == "Relevant"
    assert ranked[0]["final_score"] >= ranked[1]["final_score"]


def test_real_pdf_route_rejects_corrupt_pdf(client):
    response = client.post(
        "/score-pdf",
        files={"resume": ("broken.pdf", b"not a valid PDF", "application/pdf")},
        data={"jd_text": "Python developer"},
    )

    assert response.status_code == 422
    assert "Could not read PDF" in response.json()["detail"]
