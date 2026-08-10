"""
test_pipeline.py — Smoke tests for each module.

Run:
    python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
from src.preprocessing import clean_text, preprocess

def test_clean_text_removes_url():
    assert "http" not in clean_text("Visit https://example.com for info")

def test_clean_text_lowercase():
    assert clean_text("Python Developer") == "python developer"

def test_preprocess_returns_string():
    result = preprocess("Experienced Python developer with 5 years in ML")
    assert isinstance(result, str)
    assert len(result) > 0

# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
from src.features import FeatureBuilder, _keyword_overlap, _skill_count, extract_skills

def test_feature_builder_shape():
    resumes = ["python machine learning developer", "java backend engineer spring"]
    jds = ["python data scientist sklearn", "java developer spring boot"]
    fb = FeatureBuilder(max_features=100)
    X = fb.fit_transform(resumes, jds)
    assert X.shape == (2, 4)


def test_feature_builder_rejects_misaligned_pairs():
    with pytest.raises(ValueError, match="same number"):
        FeatureBuilder().fit(["python"], ["python", "java"])

def test_keyword_overlap():
    assert _keyword_overlap("python java sql", "java sql golang") == 2

def test_skill_count():
    assert _skill_count("python and docker and kubernetes") >= 3


def test_skill_extraction_avoids_single_letter_substrings():
    assert "r" not in extract_skills("experienced software engineer")

# ---------------------------------------------------------------------------
# ner_model (regex baseline only — no GPU needed)
# ---------------------------------------------------------------------------
from src.ner_model import RegexNER

def test_regex_ner_extracts_skills():
    ner = RegexNER()
    result = ner.extract("We use Python, Docker, and Kubernetes in our stack.")
    assert "python" in result["SKILL"]
    assert "docker" in result["SKILL"]

def test_regex_ner_extracts_degree():
    ner = RegexNER()
    result = ner.extract("Holds a Bachelor's degree in Computer Science.")
    assert any("bachelor" in d for d in result["DEGREE"])

# ---------------------------------------------------------------------------
# scoring_engine
# ---------------------------------------------------------------------------
from src.scoring_engine import ScoringWeights, ScoringEngine, keyword_gap

def test_scoring_weights_valid():
    w = ScoringWeights(0.5, 0.3, 0.2)
    assert abs(w.skills_match + w.experience_match + w.education_match - 1.0) < 1e-6

def test_scoring_weights_invalid():
    with pytest.raises(ValueError):
        ScoringWeights(0.5, 0.5, 0.5)


def test_scoring_weights_reject_negative_values():
    with pytest.raises(ValueError, match="between 0 and 1"):
        ScoringWeights(1.2, -0.2, 0.0)


class _FakeEmbedder:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


class _UnavailableEmbedder:
    def encode(self, texts):
        raise ImportError("optional model is not installed")


def test_scoring_engine_rejects_empty_text():
    engine = ScoringEngine(embedder=_FakeEmbedder())
    with pytest.raises(ValueError, match="Resume text"):
        engine.score("", "Python developer")


def test_scoring_engine_ranks_highest_score_first():
    engine = ScoringEngine(embedder=_FakeEmbedder())
    jd = "Python Docker Kubernetes Bachelor's degree"
    ranked = engine.rank(
        [("Strong", "Python Docker Kubernetes Bachelor's degree"),
         ("Weak", "Customer service and sales")],
        jd,
    )
    assert ranked[0].candidate_name == "Strong"
    assert ranked[0].final_score > ranked[1].final_score


def test_scoring_engine_uses_tfidf_when_embedder_is_unavailable():
    engine = ScoringEngine(embedder=_UnavailableEmbedder())
    result = engine.score("Python Docker", "Python Docker Kubernetes")
    assert 0 < result.embedding_similarity < 1

def test_keyword_gap_all_matched():
    gap = keyword_gap("python docker kubernetes", "python docker kubernetes")
    assert gap["missing"] == []
    assert gap["coverage"] == 1.0


def test_degree_requirement_without_candidate_degree_gets_no_credit():
    engine = ScoringEngine(embedder=_FakeEmbedder())
    result = engine.score(
        "Python developer",
        "Python developer. Bachelor's degree required.",
    )
    assert result.final_score == 80.0

def test_keyword_gap_none_matched():
    # Use text with no tokens from SKILLS_VOCAB on either side
    gap = keyword_gap("experienced candidate seeking opportunity", "looking for motivated team player")
    assert gap["coverage"] == 1.0 or gap["missing"] == []  # no JD skills → nothing missing
