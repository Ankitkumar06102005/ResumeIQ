"""
baseline_model.py — Phase 1: Classical ML baseline for resume-JD matching.

Pipeline:
  1. Load and preprocess a labeled resume-JD dataset
  2. Engineer TF-IDF + overlap features via features.py
  3. Train Logistic Regression, Random Forest, XGBoost
  4. Evaluate with cross-validation + held-out test set
  5. Save best model to models/baseline_model.pkl

Run:
    python src/baseline_model.py

Expected dataset format (CSV):
    resume_text, jd_text, label   (label: 1=match, 0=no-match)
    or
    resume_text, jd_text, score   (score: 0.0–1.0 for regression)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed — skipping XGBClassifier")

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import preprocess_batch
from src.features import FeatureBuilder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/processed/resume_jd_pairs.csv"
MODEL_DIR = "models"
RESULTS_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading + synthetic data generator (for when real data isn't ready)
# ---------------------------------------------------------------------------

def load_or_generate_data(path: str) -> pd.DataFrame:
    """
    Try to load a real dataset. If it doesn't exist yet, generate a small
    synthetic dataset so the full pipeline can be exercised immediately.
    Swap in your real Kaggle dataset at data/processed/resume_jd_pairs.csv
    and this function will use it automatically.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset: {len(df)} rows from {path}")
        _validate_columns(df)
        return df

    print(f"Dataset not found at {path}. Generating synthetic data for testing...")
    return _generate_synthetic_data()


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"resume_text", "jd_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}. "
            "Expected: resume_text, jd_text, label (0/1)"
        )


def _generate_synthetic_data(n_samples: int = 600) -> pd.DataFrame:
    """
    Realistic synthetic dataset with three difficulty tiers:
      - Easy positives/negatives  (clear match / clear mismatch)
      - Hard positives            (resume uses synonyms, partial skill overlap)
      - Hard negatives            (same domain but wrong seniority / missing key skills)

    This produces a dataset where F1 ~ 0.75–0.90 for classical models,
    which is meaningful to benchmark against Phase 2 embeddings.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Full resume-like blobs per domain (realistic length, some noise)
    resumes = {
        "ml_eng": (
            "senior machine learning engineer five years python scikit-learn xgboost pandas numpy "
            "feature engineering model deployment docker kubernetes sql postgresql experience "
            "building recommendation systems nlp pipelines aws sagemaker ci/cd mlflow"
        ),
        "ml_junior": (
            "junior data analyst python pandas matplotlib basic machine learning logistic regression "
            "two years experience tableau sql data cleaning excel statistics"
        ),
        "backend_java": (
            "backend software engineer java spring boot rest api postgresql kubernetes jenkins "
            "microservices seven years ci/cd agile scrum docker redis kafka"
        ),
        "frontend": (
            "frontend engineer react javascript typescript nodejs graphql mongodb css html "
            "agile three years webpack jest cypress responsive design"
        ),
        "data_scientist": (
            "data scientist pytorch tensorflow nlp transformers huggingface bert research "
            "python pandas numpy statistical modeling a/b testing sql four years publications"
        ),
        "devops": (
            "devops cloud engineer aws gcp terraform ansible kubernetes docker prometheus grafana "
            "ci/cd jenkins linux bash scripting monitoring five years infrastructure as code"
        ),
        "bi_analyst": (
            "business intelligence analyst sql tableau powerbi data warehouse etl looker "
            "reporting dashboards stakeholder management excel three years postgresql"
        ),
        "golang": (
            "backend engineer golang distributed systems kafka redis grpc protobuf "
            "microservices high availability postgresql linux six years"
        ),
    }

    jds = {
        "ml_eng": (
            "seeking senior machine learning engineer python scikit-learn model deployment "
            "docker kubernetes aws sql five plus years required mlops experience preferred"
        ),
        "ml_junior": (
            "data analyst role python pandas sql tableau two years experience "
            "strong communication skills stakeholder reporting data cleaning excel"
        ),
        "backend_java": (
            "java backend engineer spring boot microservices rest api kubernetes postgresql "
            "ci/cd experience required agile team docker kafka redis"
        ),
        "frontend": (
            "frontend developer react typescript javascript graphql nodejs jest "
            "three years minimum responsive design css html webpack"
        ),
        "data_scientist": (
            "data scientist deep learning pytorch nlp python sql research background preferred "
            "transformers bert pytorch statistical modeling a/b testing four years"
        ),
        "devops": (
            "cloud devops engineer aws terraform kubernetes docker ci/cd jenkins prometheus "
            "linux infrastructure as code five years required gcp a plus"
        ),
        "bi_analyst": (
            "business intelligence analyst tableau powerbi sql etl data warehouse "
            "looker dashboards stakeholder communication excel two years minimum"
        ),
        "golang": (
            "backend software engineer golang kafka distributed systems grpc redis "
            "postgresql microservices linux high availability five years experience"
        ),
    }

    # Filler phrases that add realistic noise without adding discriminative signal
    filler = [
        "strong communication skills", "team player", "fast learner",
        "detail oriented", "collaborative environment", "growth mindset",
        "proactive attitude", "results driven", "cross functional",
        "bachelor degree computer science", "master degree preferred",
        "competitive salary", "remote friendly", "flexible hours",
        "equal opportunity employer", "references available",
    ]

    domains = list(resumes.keys())
    records = []

    n_per_tier = n_samples // 6  # 6 tiers × n_per_tier ≈ n_samples

    # --- Tier 1: Easy positives — same domain, heavy overlap
    for _ in range(n_per_tier):
        d = rng.choice(domains)
        noise = " ".join(rng.choice(filler, size=rng.integers(2, 5), replace=False))
        records.append({
            "resume_text": resumes[d] + " " + noise,
            "jd_text": jds[d] + " " + noise[:len(noise)//2],
            "label": 1,
        })

    # --- Tier 2: Hard positives — same domain, only partial overlap (resume missing some JD terms)
    for _ in range(n_per_tier):
        d = rng.choice(domains)
        r_words = resumes[d].split()
        # Drop 30-50% of resume words to simulate an underqualified-but-matching candidate
        keep_n = int(len(r_words) * rng.uniform(0.5, 0.7))
        r_subset = " ".join(rng.choice(r_words, size=keep_n, replace=False))
        noise = " ".join(rng.choice(filler, size=2, replace=False))
        records.append({
            "resume_text": r_subset + " " + noise,
            "jd_text": jds[d],
            "label": 1,
        })

    # --- Tier 3: Hard positives — adjacent domains (e.g. ml_eng resume vs data_scientist JD)
    adjacent = [("ml_eng", "data_scientist"), ("backend_java", "golang"),
                ("devops", "backend_java"), ("bi_analyst", "data_scientist")]
    for _ in range(n_per_tier):
        r_domain, j_domain = adjacent[rng.integers(len(adjacent))]
        noise = " ".join(rng.choice(filler, size=2, replace=False))
        records.append({
            "resume_text": resumes[r_domain] + " " + noise,
            "jd_text": jds[j_domain],
            "label": 1,  # adjacent domain = still a reasonable match
        })

    # --- Tier 4: Easy negatives — clearly different domains
    for _ in range(n_per_tier):
        i, j = rng.choice(len(domains), size=2, replace=False)
        # Only use pairs that are actually far apart
        d_r, d_j = domains[i], domains[j]
        if (d_r, d_j) in [p for p in adjacent]:
            d_j = domains[(j + 3) % len(domains)]
        noise = " ".join(rng.choice(filler, size=rng.integers(1, 4), replace=False))
        records.append({
            "resume_text": resumes[d_r] + " " + noise,
            "jd_text": jds[d_j],
            "label": 0,
        })

    # --- Tier 5: Hard negatives — same domain but missing the critical required skill
    critical = {
        "ml_eng": "python", "backend_java": "java", "frontend": "react",
        "data_scientist": "pytorch", "devops": "aws", "bi_analyst": "sql",
        "golang": "golang", "ml_junior": "pandas",
    }
    for _ in range(n_per_tier):
        d = rng.choice(domains)
        crit = critical[d]
        # Remove the critical skill from the resume
        r_text = resumes[d].replace(crit, "").replace("  ", " ")
        noise = " ".join(rng.choice(filler, size=2, replace=False))
        records.append({
            "resume_text": r_text + " " + noise,
            "jd_text": jds[d],
            "label": 0,  # missing key required skill = no match
        })

    # --- Tier 6: Hard negatives — filler-only resumes vs real JDs
    for _ in range(n_per_tier):
        d = rng.choice(domains)
        r_text = " ".join(rng.choice(filler, size=rng.integers(4, 8), replace=False))
        records.append({
            "resume_text": r_text,
            "jd_text": jds[d],
            "label": 0,
        })

    rng.shuffle(records)
    df = pd.DataFrame(records)

    save_path = "data/processed/resume_jd_pairs.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Realistic synthetic dataset saved to {save_path} ({len(df)} rows)")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Compute and print a full classification report."""
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))

    _plot_confusion_matrix(y_test, y_pred, name)

    return metrics


def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Match", "Match"],
        yticklabels=["No Match", "Match"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    df = load_or_generate_data(DATA_PATH)
    print(f"\nClass distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Preprocess
    print("\nPreprocessing text (this may take a minute)...")
    resumes_clean = preprocess_batch(df["resume_text"].tolist())
    jds_clean = preprocess_batch(df["jd_text"].tolist())

    # 3. Feature engineering
    print("Building TF-IDF features...")
    feature_builder = FeatureBuilder(max_features=5000, ngram_range=(1, 2))
    X = feature_builder.fit_transform(resumes_clean, jds_clean)
    y = df["label"].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: [tfidf_cosine_sim, keyword_overlap, skill_count_resume, skill_count_jd]")

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # 5. Train + evaluate each model
    models = get_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_metrics = []
    best_f1 = -1
    best_model = None
    best_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set, evaluate on held-out test
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, name)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"] = round(cv_scores.std(), 4)
        all_metrics.append(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_name = name

    # 6. Save best model + feature builder
    model_path = os.path.join(MODEL_DIR, "baseline_model.pkl")
    fb_path = os.path.join(MODEL_DIR, "feature_builder.pkl")
    joblib.dump(best_model, model_path)
    feature_builder.save(fb_path)

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    print(f"Saved → {model_path}")
    print(f"Saved → {fb_path}")

    # 7. Save metrics summary
    results_df = pd.DataFrame(all_metrics)
    results_path = os.path.join(RESULTS_DIR, "phase1_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nMetrics summary:\n{results_df.to_string(index=False)}")
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    train()
