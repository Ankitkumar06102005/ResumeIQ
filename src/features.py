"""
features.py — Feature engineering for Phase 1.

Builds TF-IDF vectors, cosine similarity, and keyword-overlap features
that feed into the classical ML baseline.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import joblib
import os
import re

# A curated tech-skills vocabulary used for keyword overlap and skill matching.
SKILLS_VOCAB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang",
    "rust", "scala", "kotlin", "swift", "php", "ruby", "sql", "r", "bash",
    # Frontend & Web
    "react", "angular", "vue", "svelte", "nextjs", "nodejs", "express",
    "html", "css", "tailwind", "bootstrap", "sass", "redux", "vite", "webpack",
    # Backend Frameworks
    "fastapi", "django", "flask", "spring", "asp.net", "laravel", "rails",
    # Databases & Storage
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery", "oracle",
    # Cloud & DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "helm", "linux", "git", "github", "gitlab", "nginx",
    # AI / ML & Data Engineering
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "generative ai", "langchain", "pytorch", "tensorflow", "keras",
    "scikit-learn", "pandas", "numpy", "scipy", "spark", "pyspark",
    "hadoop", "kafka", "airflow", "dbt", "tableau", "powerbi", "excel", "looker",
    # Architecture & Practices
    "rest api", "restful", "graphql", "microservices", "system design",
    "agile", "scrum", "data analysis",
]


class FeatureBuilder:
    """
    Fits TF-IDF vectorizers on a corpus and produces a feature matrix
    with: tfidf_similarity, keyword_overlap_count, skill_count_resume,
          skill_count_jd for each (resume, jd) pair.
    """

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        # Two separate vectorizers so resume/JD vocab is learned jointly
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,  # log-scale TF dampens common terms
        )
        self._fitted = False

    def fit(self, resumes: list[str], jds: list[str]) -> "FeatureBuilder":
        """
        Fit the vectorizer on the combined corpus (resumes + JDs) so both
        share the same vocabulary space — critical for cosine similarity to
        be meaningful.
        """
        if len(resumes) != len(jds):
            raise ValueError("resumes and jds must contain the same number of items")
        combined = resumes + jds
        if not combined or not any(str(text).strip() for text in combined):
            raise ValueError("Cannot fit features on an empty corpus")
        self.vectorizer.fit(combined)
        self._fitted = True
        return self

    def transform(
        self,
        resumes: list[str],
        jds: list[str],
    ) -> np.ndarray:
        """
        Returns a (n_samples, 4) feature matrix:
          col 0: tfidf_cosine_similarity
          col 1: keyword_overlap_count
          col 2: skill_count_resume
          col 3: skill_count_jd
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")
        if len(resumes) != len(jds):
            raise ValueError("resumes and jds must contain the same number of items")

        resume_vecs = self.vectorizer.transform(resumes)
        jd_vecs = self.vectorizer.transform(jds)

        # Cosine similarity between each aligned resume-JD pair
        similarities = np.array(
            [cosine_similarity(resume_vecs[i], jd_vecs[i])[0][0]
             for i in range(len(resumes))]
        )

        # Keyword overlap: number of shared tokens after splitting
        overlaps = np.array([
            _keyword_overlap(r, j) for r, j in zip(resumes, jds)
        ])

        # Skill counts from predefined vocabulary
        skill_counts_resume = np.array([_skill_count(r) for r in resumes])
        skill_counts_jd = np.array([_skill_count(j) for j in jds])

        return np.column_stack([
            similarities,
            overlaps,
            skill_counts_resume,
            skill_counts_jd,
        ])

    def fit_transform(
        self,
        resumes: list[str],
        jds: list[str],
    ) -> np.ndarray:
        return self.fit(resumes, jds).transform(resumes, jds)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureBuilder":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _keyword_overlap(resume: str, jd: str) -> int:
    """Count tokens that appear in both the resume and the JD."""
    resume_tokens = set(resume.split())
    jd_tokens = set(jd.split())
    return len(resume_tokens & jd_tokens)


def _skill_count(text: str) -> int:
    """Count how many items from SKILLS_VOCAB appear in the text."""
    return len(extract_skills(text))


def extract_skills(text: str, skills_list: list[str] = SKILLS_VOCAB) -> set[str]:
    """Find whole skill names and avoid substring false positives."""
    normalized = (
        text.lower()
        .replace("node.js", "nodejs")
        .replace("react.js", "react")
        .replace("vue.js", "vue")
        .replace("next.js", "nextjs")
        .replace("spring boot", "spring")
    )
    return {
        skill for skill in skills_list
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(skill) + r"(?![A-Za-z0-9_])", normalized)
    }


def compute_cosine_similarity(text_a: str, text_b: str, vectorizer: TfidfVectorizer) -> float:
    """
    Utility: compute cosine similarity between two raw strings
    using an already-fitted vectorizer. Used during inference.
    """
    vecs = vectorizer.transform([text_a, text_b])
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])
