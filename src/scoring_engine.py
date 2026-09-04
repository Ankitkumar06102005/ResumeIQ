"""
scoring_engine.py — Phase 4: Weighted scoring engine combining all signals.

Combines:
  - Embedding cosine similarity  (Phase 2)
  - Keyword gap penalty          (Phase 2)
  - NER entity match score       (Phase 3)

into a single 0-100 score with configurable weights.

Also handles PDF and plain-text resume parsing.
"""

import os
import sys
import re
import io
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import preprocess
from src.embeddings import ResumeEmbedder, keyword_gap, extract_skills
from src.ner_model import RegexNER, normalize_degree, DEGREE_HIERARCHY
from src.advisor import ResumeAdvisor

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """Configurable weights for the three scoring signals. Must sum to 1.0."""
    skills_match: float = 0.50
    experience_match: float = 0.30
    education_match: float = 0.20

    def __post_init__(self):
        values = (self.skills_match, self.experience_match, self.education_match)
        if not all(isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("Weights must be finite values between 0 and 1")
        total = self.skills_match + self.experience_match + self.education_match
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


@dataclass
class ResumeScore:
    """Complete scoring result for one resume against one JD."""
    candidate_name: str
    final_score: float              # 0–100
    embedding_similarity: float     # raw cosine sim (0–1)
    keyword_coverage: float         # fraction of JD skills covered
    entity_match_score: float       # NER-based entity overlap
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extracted_entities: dict = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    insights: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file given its raw bytes."""
    text = ""
    # Try pdfminer.six first
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        extract_text_to_fp(io.BytesIO(file_bytes), output, laparams=LAParams())
        text = output.getvalue().strip()
    except Exception:
        text = ""

    # Fallback to PyPDF2 if pdfminer produced empty text or failed
    if not text:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            raise ValueError(f"Could not read PDF contents: {e}")

    if not text:
        raise ValueError("Could not extract any readable text from this PDF. It may be scanned or image-only.")
    return text


def parse_resume(source) -> str:
    """
    Accept: str (raw text), bytes (PDF content), or file path (str ending in .pdf).
    Returns plain text.
    """
    if isinstance(source, bytes):
        return extract_text_from_pdf(source)
    if isinstance(source, str) and source.lower().endswith(".pdf") and os.path.exists(source):
        with open(source, "rb") as f:
            return extract_text_from_pdf(f.read())
    if isinstance(source, str):
        return source
    raise ValueError(f"Unsupported input type: {type(source)}")


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Orchestrates all scoring signals into a final weighted score.

    Usage:
        engine = ScoringEngine()
        score = engine.score(resume_text, jd_text, candidate_name="Alice")
    """

    def __init__(
        self,
        embedder: Optional[ResumeEmbedder] = None,
        ner: Optional[RegexNER] = None,
        advisor: Optional[ResumeAdvisor] = None,
        embedder_model: str = "all-MiniLM-L6-v2",
    ):
        # Lazy-load so importing this module doesn't pull in sentence-transformers
        self._embedder = embedder
        self._embedder_model = embedder_model
        self._ner = ner or RegexNER()
        self._advisor = advisor or ResumeAdvisor()

    def _get_embedder(self) -> ResumeEmbedder:
        if self._embedder is None:
            self._embedder = ResumeEmbedder(self._embedder_model)
        return self._embedder

    def score(
        self,
        resume_text: str,
        jd_text: str,
        candidate_name: str = "Candidate",
        weights: Optional[ScoringWeights] = None,
    ) -> ResumeScore:
        """
        Score a single resume against a JD.
        All text is preprocessed before scoring.
        """
        w = weights or ScoringWeights()

        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("Resume text cannot be empty")
        if not isinstance(jd_text, str) or not jd_text.strip():
            raise ValueError("Job description cannot be empty")

        resume_clean = preprocess(resume_text)
        jd_clean = preprocess(jd_text)

        # Signal 1: semantic similarity. A TF-IDF fallback keeps the app
        # usable when the optional sentence-transformer model is unavailable.
        emb_sim = self._semantic_similarity(resume_clean, jd_clean)

        # Signal 2: Keyword gap / coverage
        gap = keyword_gap(resume_clean, jd_clean)
        coverage = gap["coverage"]

        # Signal 3: NER entity match (experience + education proxy)
        resume_entities = self._ner.extract(resume_text)
        jd_entities = self._ner.extract(jd_text)
        entity_score = _compute_entity_match(resume_entities, jd_entities)

        # Weighted combination
        # skills_match weight ← embedding sim + keyword coverage (averaged)
        skills_signal = (emb_sim + coverage) / 2.0
        # experience_match weight ← entity match (role + domain alignment)
        exp_signal = entity_score
        # education_match weight ← degree entity overlap specifically
        edu_signal = _degree_overlap(resume_entities, jd_entities)

        final = (
            w.skills_match * skills_signal
            + w.experience_match * exp_signal
            + w.education_match * edu_signal
        ) * 100.0
        final = max(0.0, min(100.0, round(final, 2)))

        # Signal 4: AI Diagnosis, Gaps & Actionable Optimizer
        insights = self._advisor.analyze(
            resume_text=resume_text,
            jd_text=jd_text,
            matched_skills=gap["matched"],
            missing_skills=gap["missing"],
            resume_entities=resume_entities,
            jd_entities=jd_entities,
        )

        return ResumeScore(
            candidate_name=candidate_name,
            final_score=final,
            embedding_similarity=round(emb_sim, 4),
            keyword_coverage=round(coverage, 4),
            entity_match_score=round(entity_score, 4),
            matched_skills=gap["matched"],
            missing_skills=gap["missing"],
            extracted_entities=resume_entities,
            weights_used={
                "skills_match": w.skills_match,
                "experience_match": w.experience_match,
                "education_match": w.education_match,
            },
            insights=insights,
        )

    def _semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Prefer dense embeddings, then fall back to deterministic TF-IDF."""
        try:
            embedder = self._get_embedder()
            r_emb = embedder.encode([resume_text])
            j_emb = embedder.encode([jd_text])
            similarity = float(np.einsum("ij,ij->i", r_emb, j_emb)[0])
        except (ImportError, OSError, RuntimeError):
            vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
                [resume_text, jd_text]
            )
            similarity = float(cosine_similarity(vectors[0], vectors[1])[0][0])
        return max(0.0, min(1.0, similarity))

    def rank(
        self,
        resumes: list[tuple[str, str]],  # [(candidate_name, resume_text), ...]
        jd_text: str,
        weights: Optional[ScoringWeights] = None,
    ) -> list[ResumeScore]:
        """
        Score and rank multiple resumes against one JD.
        Returns list sorted by final_score descending.
        """
        scores = [
            self.score(text, jd_text, name, weights)
            for name, text in resumes
        ]
        return sorted(scores, key=lambda s: s.final_score, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_entity_match(resume_ents: dict, jd_ents: dict) -> float:
    """
    Evaluate alignment on role titles and technical domains.
    Does not unfairly penalize candidates for not working at the JD's company.
    """
    scores = []
    # Primary signal: Role / Job title alignment
    r_titles = set(resume_ents.get("JOB_TITLE", []))
    j_titles = set(jd_ents.get("JOB_TITLE", []))
    if j_titles:
        title_overlap = len(r_titles & j_titles) / len(j_titles)
        scores.append(title_overlap)

    # Secondary signal: Domain skills context
    r_skills = set(resume_ents.get("SKILL", []))
    j_skills = set(jd_ents.get("SKILL", []))
    if j_skills:
        skill_overlap = len(r_skills & j_skills) / len(j_skills)
        scores.append(skill_overlap)

    # Company match is a bonus if candidate has company context matching JD
    r_comps = set(resume_ents.get("COMPANY", []))
    j_comps = set(jd_ents.get("COMPANY", []))
    if j_comps and r_comps:
        comp_overlap = len(r_comps & j_comps) / len(j_comps)
        if comp_overlap > 0:
            scores.append(1.0)

    return float(np.mean(scores)) if scores else 0.5  # default neutral


def _degree_overlap(resume_ents: dict, jd_ents: dict) -> float:
    """
    Hierarchical degree matching:
    If JD specifies a degree, candidate meets requirement if they hold
    that degree tier or higher (e.g. Master's satisfies Bachelor's).
    """
    r_degrees = [normalize_degree(d) for d in resume_ents.get("DEGREE", [])]
    j_degrees = [normalize_degree(d) for d in jd_ents.get("DEGREE", [])]

    if not j_degrees:
        return 0.75  # JD doesn't specify — neutral baseline credit

    if not r_degrees:
        return 0.0  # JD requires degree, resume has none

    jd_level = max(DEGREE_HIERARCHY.get(d, 0) for d in j_degrees)
    candidate_level = max(DEGREE_HIERARCHY.get(d, 0) for d in r_degrees)

    if candidate_level >= jd_level:
        return 1.0
    if candidate_level > 0:
        return round(candidate_level / max(jd_level, 1), 2)
    return 0.0
