"""
api.py — FastAPI backend for ResumeIQ.

Endpoints:
  POST /score          — score a single resume vs JD
  POST /rank           — rank multiple resumes vs one JD
  GET  /health         — health check

Run:
    uvicorn src.api:app --reload --port 8000
"""

import os
import sys
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.scoring_engine import ScoringEngine, ScoringWeights, parse_resume

app = FastAPI(
    title="ResumeIQ API",
    description="AI-powered resume-to-JD matching and ATS scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared engine instance (model loaded once on startup)
_engine: Optional[ScoringEngine] = None


def get_engine() -> ScoringEngine:
    global _engine
    if _engine is None:
        _engine = ScoringEngine()
    return _engine


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    resume_text: str
    jd_text: str
    candidate_name: str = "Candidate"
    weight_skills: float = 0.50
    weight_experience: float = 0.30
    weight_education: float = 0.20

    @validator("weight_skills", "weight_experience", "weight_education")
    def validate_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0 and 1")
        return v


class ScoreResponse(BaseModel):
    candidate_name: str
    final_score: float
    embedding_similarity: float
    keyword_coverage: float
    entity_match_score: float
    matched_skills: list[str]
    missing_skills: list[str]


class RankEntry(BaseModel):
    candidate_name: str
    resume_text: str


class RankRequest(BaseModel):
    resumes: list[RankEntry]
    jd_text: str
    weight_skills: float = 0.50
    weight_experience: float = 0.30
    weight_education: float = 0.20


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "ResumeIQ"}


@app.post("/score", response_model=ScoreResponse)
def score_resume(req: ScoreRequest):
    """Score a single resume against a job description."""
    try:
        weights = ScoringWeights(
            skills_match=req.weight_skills,
            experience_match=req.weight_experience,
            education_match=req.weight_education,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = get_engine().score(
            req.resume_text, req.jd_text, req.candidate_name, weights
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ScoreResponse(
        candidate_name=result.candidate_name,
        final_score=result.final_score,
        embedding_similarity=result.embedding_similarity,
        keyword_coverage=result.keyword_coverage,
        entity_match_score=result.entity_match_score,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
    )


@app.post("/rank")
def rank_resumes(req: RankRequest):
    """Rank multiple resumes against one job description."""
    try:
        weights = ScoringWeights(
            skills_match=req.weight_skills,
            experience_match=req.weight_experience,
            education_match=req.weight_education,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resumes = [(r.candidate_name, r.resume_text) for r in req.resumes]
    if not resumes:
        raise HTTPException(status_code=422, detail="Provide at least one resume")
    try:
        ranked = get_engine().rank(resumes, req.jd_text, weights)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "ranked": [
            {
                "rank": i + 1,
                "candidate_name": r.candidate_name,
                "final_score": r.final_score,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
            }
            for i, r in enumerate(ranked)
        ]
    }


@app.post("/score-pdf")
async def score_pdf(
    resume: UploadFile = File(...),
    jd_text: str = Form(...),
    candidate_name: str = Form("Candidate"),
    weight_skills: float = Form(0.50),
    weight_experience: float = Form(0.30),
    weight_education: float = Form(0.20),
):
    """Score a PDF resume uploaded as a file."""
    if not resume.filename or not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_bytes = await resume.read()
        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty")
        resume_text = parse_resume(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")

    try:
        weights = ScoringWeights(
            skills_match=weight_skills,
            experience_match=weight_experience,
            education_match=weight_education,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        result = get_engine().score(resume_text, jd_text, candidate_name, weights)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "candidate_name": result.candidate_name,
        "final_score": result.final_score,
        "embedding_similarity": result.embedding_similarity,
        "keyword_coverage": result.keyword_coverage,
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
    }
