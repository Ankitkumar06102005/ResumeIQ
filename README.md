# ResumeIQ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resumeiq-partner.streamlit.app/)

> 🚀 **Live Demo:** [https://resumeiq-partner.streamlit.app/](https://resumeiq-partner.streamlit.app/)

An explainable AI-assisted applicant-tracking prototype that scores and ranks resumes against a job description. It provides a Streamlit recruiter dashboard and a FastAPI backend, with deterministic fallbacks when the optional embedding model is unavailable.

ResumeIQ scores resumes against a job description and ranks applicants for a recruiter. It combines keyword coverage, semantic similarity, and resume entities into an explainable 0-100 score.

## Included capabilities

- Phase 1: TF-IDF similarity, keyword overlap, classical classifiers, cross-validation, and confusion matrices.
- Phase 2: `all-MiniLM-L6-v2` sentence embeddings plus matched and missing skill detection.
- Phase 3: regex NER today and a DistilBERT NER training utility for BIO-labelled data.
- Phase 4: Streamlit recruiter dashboard, PDF/TXT uploads, adjustable weights, and multi-resume ranking.
- Phase 5: FastAPI endpoints and a root `app.py` entry point compatible with Hugging Face Spaces.
- Phase 6: AI Resume Diagnosis & Optimizer — detects critical gaps ("What It Lacks"), provides Google X-Y-Z formula before/after suggestions, audits quantified metrics & power verbs, and generates 95+ tailored sample resume templates.

## Quick start

Use Python 3.10-3.12 (the pinned PyTorch release is not compatible with Python 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/prepare_data.py
python src/baseline_model.py
streamlit run app.py
```

This installs only the lightweight application runtime. To run the optional XGBoost, transformer NER, and sentence-embedding training workflows, install the additional training stack:

```bash
pip install -r requirements-training.txt
```

Then open `http://localhost:8501`. To run the API instead, use `uvicorn src.api:app --reload --port 8000` and visit `http://127.0.0.1:8000/docs`.

The first semantic scoring request downloads `all-MiniLM-L6-v2` from Hugging Face. For offline demos, run the Phase 1 training and use the included artifacts, or make the model available in the Hugging Face cache before deployment.

To use spaCy lemmatization during Phase 1 training, install its English model and set the opt-in flag. The app deliberately falls back to lightweight normalization otherwise, so it can start on restricted hosts.

```bash
python -m spacy download en_core_web_sm
$env:RESUMEIQ_USE_SPACY = "1"
python src/baseline_model.py
```

## Data and training

The source dataset and generated training pairs are intentionally excluded from Git to keep the repository lightweight. Place a compatible `Resume.csv` in the project root, then run `python src/prepare_data.py` to create labelled resume/JD pairs. The expected training columns are `resume_text`, `jd_text`, and `label` (1 for match, 0 for no match). Results are saved under `data/processed/`; the best classical model and feature builder are saved under `models/`.

Run Phase 2 after Phase 1:

```bash
python src/embeddings.py
```

This writes a side-by-side F1 comparison to `data/processed/phase1_vs_phase2.csv`.

The checked-in Phase 1 run produced the following held-out metrics on the generated resume/JD-pair dataset:

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7565 | 0.7865 | 0.7042 | 0.7431 |
| Random Forest | 0.8179 | 0.8319 | 0.7968 | 0.8140 |
| XGBoost | 0.8159 | 0.8369 | 0.7847 | 0.8100 |

## Run the API

```bash
uvicorn src.api:app --reload --port 8000
```

- `GET /health` verifies the service.
- `POST /score` scores one JSON resume/JD pair.
- `POST /rank` ranks a JSON list of candidates against one JD.
- `POST /score-pdf` accepts a PDF resume and form fields.

Interactive endpoint documentation is available at `http://127.0.0.1:8000/docs`.

## Verify

```bash
python -m pytest tests -v -p no:cacheprovider
```

Continuous integration runs this same command on every push and pull request to `main`.

## Live demo

- **Live Streamlit App:** [https://resumeiq-partner.streamlit.app/](https://resumeiq-partner.streamlit.app/)

Deployed with [Streamlit Community Cloud](https://share.streamlit.io): branch `main`, entrypoint `app.py`. The live app uses the reliable TF-IDF similarity fallback if the optional sentence-transformer model is not installed. GitHub pushes automatically redeploy the app.

## Project structure

```
app/                 Streamlit recruiter dashboard
src/                 scoring engine, API, NLP, and training modules
tests/               pipeline regression tests
data/processed/      generated evaluation outputs and training pairs
models/              serialized Phase 1 and embedding configuration artifacts
```

## Scoring model

```
resume + job description
        |
normalization -> dense embeddings / skill gap / entity extraction
        |
weighted skills + role experience + hierarchical education
        |
explainable ATS score + AI diagnosis (gaps & suggestions) + tailored template
```

The dashboard requires the three recruiter weights to total 1.0. ResumeIQ is a decision-support tool: reviewers should validate results and avoid using it as the sole basis for employment decisions.
