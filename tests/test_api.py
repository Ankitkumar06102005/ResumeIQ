from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def client(monkeypatch):
    fake_result = SimpleNamespace(
        candidate_name="Test Candidate",
        final_score=82.5,
        embedding_similarity=0.81,
        keyword_coverage=0.75,
        entity_match_score=0.9,
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Docker"],
        insights={
            "gaps": [{"title": "Docker", "detail": "Add verified Docker experience."}],
            "suggestions": [{"title": "Add evidence", "action": "Quantify a real result."}],
            "sample_tailored_resume": "Example template",
        },
        weights_used={"skills_match": 0.5, "experience_match": 0.3, "education_match": 0.2},
        extracted_entities={"SKILL": ["Python"]},
    )

    class FakeEngine:
        def score(self, resume_text, jd_text, candidate_name, weights):
            if not resume_text.strip():
                raise ValueError("Resume text cannot be empty")
            if not jd_text.strip():
                raise ValueError("Job description cannot be empty")
            return SimpleNamespace(**{**fake_result.__dict__, "candidate_name": candidate_name})

        def rank(self, resumes, jd_text, weights):
            if not jd_text.strip():
                raise ValueError("Job description cannot be empty")
            if not resumes:
                raise ValueError("Provide at least one resume")
            return [
                SimpleNamespace(**{**fake_result.__dict__, "candidate_name": name})
                for name, _ in sorted(resumes, key=lambda pair: pair[0])
            ]

    monkeypatch.setattr(api, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(api, "parse_resume", lambda payload: "Python FastAPI experience")
    return TestClient(api.app)


def test_root_and_health_endpoints(client):
    root = client.get("/")
    health = client.get("/health")

    assert root.status_code == 200
    assert root.json()["service"] == "ResumeIQ API"
    assert root.json()["status"] == "ok"
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_score_endpoint_returns_expected_schema(client):
    response = client.post(
        "/score",
        json={
            "resume_text": "Python developer with FastAPI experience",
            "jd_text": "Python FastAPI developer",
            "candidate_name": "Test Candidate",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_name"] == "Test Candidate"
    assert payload["final_score"] == 82.5
    assert payload["matched_skills"] == ["Python", "FastAPI"]
    assert payload["missing_skills"] == ["Docker"]
    assert payload["insights"]["suggestions"]


@pytest.mark.parametrize(
    "body",
    [
        {"jd_text": "Python developer"},
        {"resume_text": "Python developer"},
        {
            "resume_text": "Python developer",
            "jd_text": "Python developer",
            "weight_skills": 1.2,
        },
    ],
)
def test_score_endpoint_rejects_invalid_request(client, body):
    response = client.post("/score", json=body)
    assert response.status_code == 422


def test_score_endpoint_rejects_empty_resume_and_job_description(client):
    empty_resume = client.post(
        "/score", json={"resume_text": " ", "jd_text": "Python developer"}
    )
    empty_jd = client.post(
        "/score", json={"resume_text": "Python developer", "jd_text": " "}
    )

    assert empty_resume.status_code == 422
    assert empty_jd.status_code == 422


def test_rank_endpoint_returns_ranked_candidates(client):
    response = client.post(
        "/rank",
        json={
            "jd_text": "Python developer",
            "resumes": [
                {"candidate_name": "Candidate B", "resume_text": "Python"},
                {"candidate_name": "Candidate A", "resume_text": "Python FastAPI"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()["ranked"]
    assert len(payload) == 2
    assert payload[0]["rank"] == 1
    assert payload[0]["candidate_name"] == "Candidate A"
    assert "missing_skills" in payload[0]


def test_rank_endpoint_rejects_empty_candidate_list(client):
    response = client.post("/rank", json={"jd_text": "Python developer", "resumes": []})
    assert response.status_code == 422


def test_score_pdf_endpoint_parses_and_scores_pdf(client):
    response = client.post(
        "/score-pdf",
        files={"resume": ("resume.pdf", b"%PDF-fake-content", "application/pdf")},
        data={"jd_text": "Python developer", "candidate_name": "PDF Candidate"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["candidate_name"] == "PDF Candidate"
    assert payload["final_score"] == 82.5
    assert payload["matched_skills"] == ["Python", "FastAPI"]


def test_score_pdf_rejects_non_pdf_upload(client):
    response = client.post(
        "/score-pdf",
        files={"resume": ("resume.txt", b"Python developer", "text/plain")},
        data={"jd_text": "Python developer"},
    )
    assert response.status_code == 400


def test_score_pdf_rejects_empty_job_description(client):
    response = client.post(
        "/score-pdf",
        files={"resume": ("resume.pdf", b"%PDF-fake-content", "application/pdf")},
        data={"jd_text": " "},
    )
    assert response.status_code == 422


def test_openapi_schema_is_available(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/score" in schema["paths"]
    assert "/rank" in schema["paths"]
    assert "/score-pdf" in schema["paths"]
    assert "/health" in schema["paths"]
