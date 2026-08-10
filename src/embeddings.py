"""
embeddings.py — Phase 2: Sentence-transformer semantic similarity + keyword gap.

Replaces TF-IDF cosine similarity with dense embeddings from all-MiniLM-L6-v2,
which captures semantic meaning that TF-IDF misses (e.g., "engineered" ≈ "built").

Run Phase 2 comparison:
    python src/embeddings.py
"""

import os
import sys
import warnings
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import preprocess_batch
from src.features import FeatureBuilder, SKILLS_VOCAB, extract_skills as _extract_skills

warnings.filterwarnings("ignore")

if False:  # Imported lazily when semantic scoring is actually requested.
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
HAS_ST = False
if False:
    warnings.warn("sentence-transformers not installed — Phase 2 unavailable")

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "data/processed/resume_jd_pairs.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

class ResumeEmbedder:
    """
    Wraps SentenceTransformer to encode resumes and JDs into dense vectors
    and compute cosine similarity scores.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        if False:
            raise ImportError("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of strings → (n, embedding_dim) array."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 norm so dot product = cosine sim
        )

    def similarity(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """
        Pairwise cosine similarity between aligned rows of two embedding matrices.
        Returns a 1-D array of shape (n,).
        """
        # Since embeddings are L2-normalised, dot product = cosine similarity
        return np.einsum("ij,ij->i", embeddings_a, embeddings_b)

    def save(self, path: str) -> None:
        # We save the model name, not the weights (they download from HF Hub)
        joblib.dump({"model_name": self.model_name}, path)

    @staticmethod
    def load(path: str) -> "ResumeEmbedder":
        data = joblib.load(path)
        return ResumeEmbedder(model_name=data["model_name"])


# ---------------------------------------------------------------------------
# Keyword gap detector
# ---------------------------------------------------------------------------

def extract_skills(text: str, skills_list: list[str] = SKILLS_VOCAB) -> set[str]:
    """Return the subset of skills_list that appear in text."""
    return _extract_skills(text, skills_list)


def keyword_gap(resume_text: str, jd_text: str) -> dict:
    """
    Compare skills mentioned in the JD vs the resume.

    Returns:
        matched:  skills in both resume and JD
        missing:  skills in JD but not in resume
        extra:    skills in resume but not in JD (bonus context)
        gap_score: fraction of JD skills that are missing (lower = better)
    """
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    gap_score = len(missing) / max(len(jd_skills), 1)

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "gap_score": round(gap_score, 4),
        "coverage": round(1 - gap_score, 4),  # % of JD skills covered
    }


# ---------------------------------------------------------------------------
# Phase 2 training + comparison with Phase 1
# ---------------------------------------------------------------------------

def run_phase2_comparison() -> None:
    """
    Load the same dataset used in Phase 1, build embedding features,
    train a classifier, and compare F1 against the Phase 1 baseline.
    """
    import pandas as pd

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run baseline_model.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Preprocess (same as Phase 1 — fair comparison)
    print("Preprocessing...")
    resumes_clean = preprocess_batch(df["resume_text"].tolist())
    jds_clean = preprocess_batch(df["jd_text"].tolist())
    y = df["label"].values

    # Train/test split — same split as Phase 1
    indices = np.arange(len(df))
    _, _, _, _, idx_train, idx_test = train_test_split(
        resumes_clean, jds_clean, indices,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --------------- Phase 2: Embeddings ---------------
    embedder = ResumeEmbedder()
    print("Encoding resumes...")
    resume_embs = embedder.encode(resumes_clean)
    print("Encoding job descriptions...")
    jd_embs = embedder.encode(jds_clean)

    # Primary feature: embedding cosine similarity
    sim_scores = embedder.similarity(resume_embs, jd_embs).reshape(-1, 1)

    # Add keyword gap coverage as a second feature
    gap_features = np.array([
        keyword_gap(r, j)["coverage"]
        for r, j in zip(resumes_clean, jds_clean)
    ]).reshape(-1, 1)

    X_embed = np.hstack([sim_scores, gap_features])

    X_train_e = X_embed[idx_train]
    X_test_e = X_embed[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_e, y_train)
    y_pred_e = clf.predict(X_test_e)
    f1_embed = f1_score(y_test, y_pred_e, zero_division=0)

    print("\nPhase 2 — Embeddings + Gap Coverage:")
    print(classification_report(y_test, y_pred_e, target_names=["No Match", "Match"]))

    # --------------- Phase 1 baseline (load saved model) ---------------
    phase1_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    fb_path = os.path.join(MODEL_DIR, "feature_builder.pkl")
    f1_baseline = None

    if os.path.exists(phase1_path) and os.path.exists(fb_path):
        baseline_model = joblib.load(phase1_path)
        fb = FeatureBuilder.load(fb_path)
        X_tfidf = fb.transform(resumes_clean, jds_clean)
        X_test_t = X_tfidf[idx_test]
        y_pred_b = baseline_model.predict(X_test_t)
        f1_baseline = f1_score(y_test, y_pred_b, zero_division=0)
    else:
        print("Phase 1 model not found — run baseline_model.py first for comparison.")

    # --------------- Comparison table ---------------
    comparison = pd.DataFrame([
        {"approach": "Phase 1 — TF-IDF + Classical ML", "f1_score": round(f1_baseline, 4) if f1_baseline else "N/A"},
        {"approach": "Phase 2 — Embeddings (all-MiniLM-L6-v2)", "f1_score": round(f1_embed, 4)},
    ])
    print("\n" + "="*55)
    print("  COMPARISON: Phase 1 vs Phase 2")
    print("="*55)
    print(comparison.to_string(index=False))

    comparison.to_csv(os.path.join(RESULTS_DIR, "phase1_vs_phase2.csv"), index=False)

    # Save embedder reference
    embedder.save(os.path.join(MODEL_DIR, "embedder.pkl"))
    print(f"\nEmbedder config saved → {os.path.join(MODEL_DIR, 'embedder.pkl')}")


if __name__ == "__main__":
    run_phase2_comparison()
