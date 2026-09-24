<div align="center">

# 📄 ResumeIQ
### *Explainable AI for Resume–Job Matching*

Turn resumes into actionable hiring insights.

[![Live App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://resumeiq-partner.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![NLP](https://img.shields.io/badge/NLP-Explainable%20Scoring-8E44AD?style=for-the-badge)

</div>

---

## 🎯 Overview

ResumeIQ is an explainable AI-assisted applicant-tracking prototype that scores and ranks resumes against a job description. It combines lexical similarity, semantic embeddings, skill-gap detection, entity extraction, and recruiter-adjustable weights to produce an interpretable 0–100 score.

The system also diagnoses resume weaknesses, suggests quantified X-Y-Z improvements, audits power verbs and metrics, and generates tailored sample resume templates.

## 🧠 Architecture

```mermaid
flowchart TD
    R[Resume PDF / TXT] --> APP[Streamlit Recruiter Dashboard]
    JD[Job Description] --> APP
    APP --> API[FastAPI Service]
    APP --> ENG[Scoring Engine]
    API --> ENG
    R --> EXT[Text Extraction]
    EXT --> NORM[Normalization / NLP Preprocessing]
    JD --> NORM
    NORM --> TF[TF-IDF Similarity]
    NORM --> EMB[MiniLM Sentence Embeddings]
    NORM --> SK[Skill Coverage + Gap Detection]
    NORM --> ENT[Entity Extraction]
    TF --> SCORE[Weighted Explainable Score]
    EMB --> SCORE
    SK --> SCORE
    ENT --> SCORE
    SCORE --> RANK[Candidate Ranking]
    SCORE --> DIAG[AI Resume Diagnosis]
    DIAG --> OPT[Actionable Suggestions + X-Y-Z Rewrites]
    OPT --> OUT[Recruiter Insights]
    RANK --> OUT
```

## 🔄 Processing workflow

1. **Ingest:** upload one or multiple resumes and provide a job description.
2. **Extract:** parse PDF/TXT content into normalized text.
3. **Represent:** compute lexical features with TF-IDF and optionally semantic vectors with `all-MiniLM-L6-v2`.
4. **Compare:** evaluate keyword coverage, semantic similarity, skills, entities, experience, and education signals.
5. **Score:** combine enabled signals using recruiter-controlled weights into an explainable 0–100 score.
6. **Rank:** sort candidates and expose matched and missing evidence.
7. **Improve:** generate diagnosis, quantified rewrite suggestions, power-verb audits, and tailored templates.

## 🧪 Modeling phases

| Phase | Implementation |
|---|---|
| 1 | TF-IDF, keyword overlap, classical classifiers, cross-validation, confusion matrices |
| 2 | `all-MiniLM-L6-v2` embeddings and skill-gap detection |
| 3 | Regex NER with optional DistilBERT NER training utility |
| 4 | Streamlit dashboard, uploads, adjustable weights, multi-resume ranking |
| 5 | FastAPI endpoints and Hugging Face Spaces-compatible entry point |
| 6 | AI diagnosis, X-Y-Z suggestions, metrics/power-verb audit, tailored templates |

## 📊 Phase 1 benchmark

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7565 | 0.7865 | 0.7042 | 0.7431 |
| Random Forest | 0.8179 | 0.8319 | 0.7968 | 0.8140 |
| XGBoost | 0.8159 | 0.8369 | 0.7847 | 0.8100 |

## 🔌 API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/score` | Score one resume/JD pair |
| POST | `/rank` | Rank multiple candidates |
| POST | `/score-pdf` | Score an uploaded PDF resume |

Interactive docs: `http://127.0.0.1:8000/docs`

## 🚀 Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/prepare_data.py
python src/baseline_model.py
streamlit run app.py
```

Optional training stack:

```bash
pip install -r requirements-training.txt
```

Run the API:

```bash
uvicorn src.api:app --reload --port 8000
```

Run tests:

```bash
python -m pytest tests -v -p no:cacheprovider
```

## 📁 Structure

```text
app/                 Streamlit recruiter dashboard
src/                 scoring, API, NLP, and training modules
tests/               regression tests
data/processed/      generated training and evaluation outputs
models/              serialized model artifacts
```

## ⚖️ Responsible-use note

ResumeIQ is a decision-support tool. Recruiters should validate outputs, inspect evidence, and avoid using an automated score as the sole basis for employment decisions.

---

<div align="center">

[🚀 Try ResumeIQ Live](https://resumeiq-partner.streamlit.app/)

</div>
