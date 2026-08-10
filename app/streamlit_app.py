"""
streamlit_app.py — ResumeIQ recruiter dashboard.

Two modes:
  1. Single Resume — paste/upload one resume, get a detailed score breakdown
  2. Batch Ranking — upload multiple resumes, get a ranked leaderboard

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring_engine import ScoringEngine, ScoringWeights, parse_resume

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeIQ — ATS Scoring Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal clean look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .score-box {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 2px solid #4f46e5;
    }
    .score-number {
        font-size: 64px;
        font-weight: 800;
        color: #4f46e5;
    }
    .skill-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 13px;
    }
    .matched { background: #d1fae5; color: #065f46; }
    .missing { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — weights + mode selection
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Scoring Configuration")
st.sidebar.markdown("Adjust how much each factor contributes to the final score.")

w_skills = st.sidebar.slider("Skills Match Weight", 0.0, 1.0, 0.50, 0.05)
w_exp = st.sidebar.slider("Experience Match Weight", 0.0, 1.0, 0.30, 0.05)
w_edu = st.sidebar.slider("Education Match Weight", 0.0, 1.0, 0.20, 0.05)

total_w = round(w_skills + w_exp + w_edu, 2)
if abs(total_w - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Weights sum to {total_w:.2f} — they must sum to 1.0")
    st.sidebar.stop()

mode = st.sidebar.radio("Mode", ["Single Resume", "Batch Ranking"])

# ---------------------------------------------------------------------------
# Engine — cached so the embedding model only loads once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models...")
def load_engine() -> ScoringEngine:
    return ScoringEngine()


engine = load_engine()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_upload(uploaded_file) -> str:
    """Read a Streamlit UploadedFile → plain text."""
    if uploaded_file.name.endswith(".pdf"):
        return parse_resume(uploaded_file.read())
    return uploaded_file.read().decode("utf-8", errors="ignore")


def _score_color(score: float) -> str:
    if score >= 75:
        return "#065f46"
    if score >= 50:
        return "#92400e"
    return "#991b1b"


def _render_skill_chips(skills: list[str], chip_class: str) -> None:
    chips = "".join(
        f'<span class="skill-chip {chip_class}">{s}</span>'
        for s in skills
    )
    st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Single Resume mode
# ---------------------------------------------------------------------------

def single_resume_mode():
    st.title("📄 ResumeIQ — Resume Scorer")
    st.markdown("Paste or upload a resume and a job description to get an ATS score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        jd_upload = st.file_uploader("Upload JD (.txt)", type=["txt"], key="jd_file")
        jd_text = st.text_area(
            "Or paste JD text here",
            value=jd_upload and _read_upload(jd_upload) or "",
            height=300,
            key="jd_text",
        )

    with col2:
        st.subheader("Resume")
        resume_upload = st.file_uploader("Upload Resume (.pdf or .txt)", type=["pdf", "txt"], key="resume_file")
        resume_text = st.text_area(
            "Or paste resume text here",
            value=resume_upload and _read_upload(resume_upload) or "",
            height=300,
            key="resume_text",
        )

    candidate_name = st.text_input("Candidate name (optional)", value="Candidate")

    if st.button("🚀 Score Resume", type="primary", use_container_width=True):
        if not jd_text.strip() or not resume_text.strip():
            st.error("Please provide both a resume and a job description.")
            return

        with st.spinner("Scoring..."):
            try:
                weights = ScoringWeights(w_skills, w_exp, w_edu)
                result = engine.score(resume_text, jd_text, candidate_name, weights)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                return

        # Results
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)

        color = _score_color(result.final_score)
        c1.markdown(
            f'<div class="score-box"><div class="score-number" style="color:{color}">'
            f'{result.final_score:.0f}</div><div>ATS Score / 100</div></div>',
            unsafe_allow_html=True,
        )
        c2.metric("Semantic Similarity", f"{result.embedding_similarity:.2%}")
        c3.metric("Skill Coverage", f"{result.keyword_coverage:.2%}")
        c4.metric("Entity Match", f"{result.entity_match_score:.2%}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader(f"✅ Matched Skills ({len(result.matched_skills)})")
            if result.matched_skills:
                _render_skill_chips(result.matched_skills, "matched")
            else:
                st.info("No matching skills detected.")

        with col_b:
            st.subheader(f"❌ Missing Skills ({len(result.missing_skills)})")
            if result.missing_skills:
                _render_skill_chips(result.missing_skills, "missing")
            else:
                st.success("No skill gaps detected.")

        with st.expander("🔍 Extracted Entities (NER)"):
            for etype, items in result.extracted_entities.items():
                if items:
                    st.write(f"**{etype}:** {', '.join(items)}")

        with st.expander("⚖️ Score Breakdown"):
            st.json({
                "final_score": result.final_score,
                "embedding_similarity": result.embedding_similarity,
                "keyword_coverage": result.keyword_coverage,
                "entity_match_score": result.entity_match_score,
                "weights": result.weights_used,
            })


# ---------------------------------------------------------------------------
# Batch ranking mode
# ---------------------------------------------------------------------------

def batch_ranking_mode():
    st.title("📊 ResumeIQ — Batch Ranker")
    st.markdown("Upload one JD and multiple resumes to rank candidates.")

    st.subheader("Job Description")
    jd_upload = st.file_uploader("Upload JD (.txt)", type=["txt"], key="batch_jd")
    jd_text = st.text_area(
        "Or paste JD text",
        value=jd_upload and _read_upload(jd_upload) or "",
        height=200,
        key="batch_jd_text",
    )

    st.subheader("Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF or TXT — multiple allowed)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="batch_resumes",
    )

    if st.button("🏆 Rank All Candidates", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Please provide a job description.")
            return
        if not resume_files:
            st.error("Please upload at least one resume.")
            return

        with st.spinner(f"Scoring {len(resume_files)} resume(s)..."):
            resumes = []
            for f in resume_files:
                name = os.path.splitext(f.name)[0]
                text = _read_upload(f)
                resumes.append((name, text))

            try:
                weights = ScoringWeights(w_skills, w_exp, w_edu)
                ranked = engine.rank(resumes, jd_text, weights)
            except Exception as e:
                st.error(f"Ranking failed: {e}")
                return

        st.markdown("---")
        st.subheader("🏆 Candidate Rankings")

        # Leaderboard table
        table_data = []
        for i, r in enumerate(ranked):
            table_data.append({
                "Rank": i + 1,
                "Candidate": r.candidate_name,
                "Score": f"{r.final_score:.1f}",
                "Semantic Sim": f"{r.embedding_similarity:.2%}",
                "Skill Coverage": f"{r.keyword_coverage:.2%}",
                "Matched Skills": len(r.matched_skills),
                "Missing Skills": len(r.missing_skills),
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detailed breakdown per candidate
        st.markdown("---")
        st.subheader("📋 Detailed Breakdown")
        for r in ranked:
            with st.expander(f"#{ranked.index(r)+1} — {r.candidate_name} ({r.final_score:.1f}/100)"):
                ca, cb = st.columns(2)
                with ca:
                    st.write("**✅ Matched Skills**")
                    _render_skill_chips(r.matched_skills or ["(none)"], "matched")
                with cb:
                    st.write("**❌ Missing Skills**")
                    _render_skill_chips(r.missing_skills or ["(none)"], "missing")

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            file_name="resumeiq_rankings.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if mode == "Single Resume":
    single_resume_mode()
else:
    batch_ranking_mode()
