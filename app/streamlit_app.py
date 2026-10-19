"""
ResumeIQ — lavender document-intelligence workspace.
The scoring and parsing work is delegated to the existing ScoringEngine.
"""
import html
import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scoring_engine import ScoringEngine, ScoringWeights, parse_resume

st.set_page_config(
    page_title="ResumeIQ — Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #a99af1; }
[data-testid="stAppViewContainer"] > .main { background: #f7f6fc; }
[data-testid="stHeader"] { background: rgba(247,246,252,.92); }
.block-container { padding: 1.15rem 1.55rem 1.6rem; max-width: 1600px; }
section[data-testid="stSidebar"] { background: #fbfafe; border-right: 1px solid #eae6f5; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
h1,h2,h3,h4 { color: #302b45 !important; letter-spacing: -.025em; }
p, label, [data-testid="stMarkdownContainer"] { color: #77728d; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #e8e3f4 !important; border-radius: 16px !important; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e8e3f4; border-radius: 14px; padding: 15px 17px; }
div[data-testid="stMetricLabel"] { color: #8b85a1; }
div[data-testid="stMetricValue"] { color: #302b45; }
.stButton > button, .stDownloadButton > button { border-radius: 11px; border: 1px solid #ded7f6; color: #5e4cc1; background: #f5f2ff; font-weight: 600; min-height: 2.5rem; }
.stButton > button[kind="primary"], .stFormSubmitButton > button { background: #6b58ca; border: 1px solid #6b58ca; color: white; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #9889e9; color: #4d3bb0; }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { border-color: #e5dff5; border-radius: 10px; background: #fbfafe; }
hr { border-color: #eae6f5; }
.rr-brand { display:flex; align-items:center; gap:11px; padding: 8px 0 22px; }
.rr-mark { width:39px;height:39px;border-radius:14px;background:#e8e2ff;color:#6754c8;display:flex;align-items:center;justify-content:center;font-size:21px;font-weight:800; }
.rr-brand-name { color:#28253a;font-family:Georgia,serif;font-size:23px;font-weight:700;line-height:1.1; }
.rr-brand-sub { color:#8b86a3;font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-top:4px; }
.rr-eyebrow { color:#928aa9;font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase; }
.rr-title { color:#302b45;font-size:27px;font-weight:750;letter-spacing:-.04em;margin:3px 0 5px; }
.rr-subtitle { color:#89839f;font-size:13px;line-height:1.55; }
.rr-card { background:#fff;border:1px solid #e8e3f4;border-radius:16px;padding:19px 20px;box-shadow:0 5px 20px rgba(67,54,120,.035);margin-bottom:13px; }
.rr-card-title { color:#7e7797;font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px; }
.rr-pill { display:inline-block;border-radius:999px;padding:4px 9px;background:#eaf7f0;color:#4d9a78;font-size:10px;font-weight:700;margin:2px 4px 2px 0; }
.rr-pill-warn { background:#fff2e5;color:#b77b38; }
.rr-chat { background:#fbfafe;border:1px solid #ece7f7;border-radius:15px;padding:15px;margin-bottom:10px; }
.rr-chat-label { color:#7664c8;font-size:10px;font-weight:800;margin-bottom:7px; }
.rr-muted { color:#9992ae;font-size:11px; }
.rr-score { color:#6b58ca;font-size:45px;font-weight:800;line-height:1.05;letter-spacing:-.05em; }
.rr-divider { height:1px;background:#eeeaf7;margin:13px 0; }
@media (max-width: 900px) { .block-container { padding: .8rem .75rem 1.2rem; } .rr-title {font-size:23px;} }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading ResumeIQ scoring engine…")
def load_engine():
    return ScoringEngine()

engine = load_engine()

def safe(value):
    return html.escape(str(value or ""))

def read_upload(uploaded_file):
    try:
        payload = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".pdf"):
            return parse_resume(payload)
        return payload.decode("utf-8", errors="ignore")
    except Exception as exc:
        st.warning(f"Could not read {uploaded_file.name}: {exc}")
        return ""

def score_color(score):
    if score >= 75: return "#4d9a78"
    if score >= 50: return "#b77b38"
    return "#c56e77"

def skill_pills(skills, warning=False):
    if not skills:
        st.caption("None detected.")
        return
    cls = "rr-pill rr-pill-warn" if warning else "rr-pill"
    st.markdown("".join(f'<span class="{cls}">{safe(skill)}</span>' for skill in skills), unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="rr-brand"><div class="rr-mark">✳</div><div><div class="rr-brand-name">ResumeIQ</div><div class="rr-brand-sub">Resume intelligence</div></div></div>', unsafe_allow_html=True)
        st.caption("WORKSPACE")
        page = st.radio(
            "Workspace navigation",
            ["Dashboard", "Applications", "Jobs", "Documents", "Analytics", "Settings"],
            label_visibility="collapsed",
            key="rr_nav",
        )
        st.markdown("---")
        st.markdown("**Scoring configuration**")
        st.caption("Adjust the contribution of each scoring signal. Weights must total 100%.")
        skills = st.slider("Skills match", 0.0, 1.0, 0.50, 0.05, key="rr_w_skills")
        exp = st.slider("Experience match", 0.0, 1.0, 0.30, 0.05, key="rr_w_exp")
        edu = st.slider("Education match", 0.0, 1.0, 0.20, 0.05, key="rr_w_edu")
        total = round(skills + exp + edu, 2)
        if abs(total - 1.0) > .001:
            st.warning(f"Weights total {total:.0%}. Adjust them to 100%.")
        st.markdown("---")
        st.markdown("🌐 [Live application](https://resumeiq-partner.streamlit.app/)")
        st.markdown("💻 [GitHub repository](https://github.com/Ankitkumar06102005/ResumeIQ)")
        st.markdown('<div class="rr-muted" style="padding-top:14px">AI-assisted decision support · Verify evidence before making hiring decisions.</div>', unsafe_allow_html=True)
    return page, skills, exp, edu, abs(total - 1.0) <= .001

def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="rr-eyebrow">{safe(eyebrow)}</div><div class="rr-title">{safe(title)}</div><div class="rr-subtitle">{safe(subtitle)}</div>', unsafe_allow_html=True)

def render_result(result, candidate_name):
    insights = getattr(result, "insights", {}) or {}
    score = float(result.final_score)
    left, right = st.columns([1.12, .88], gap="medium")
    with left:
        st.markdown('<div class="rr-card"><div class="rr-card-title">Resume match</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([.7, 1.3])
        with c1:
            st.markdown(f'<div class="rr-score" style="color:{score_color(score)}">{score:.0f}<span style="font-size:17px">/100</span></div><div class="rr-muted">Weighted match score</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="font-weight:750;color:#4d9a78;margin-bottom:5px">{"Strong alignment" if score >= 75 else "Promising match" if score >= 50 else "Room to improve"}</div><div class="rr-subtitle">The score summarizes the current resume and job description using ResumeIQ’s existing scoring engine.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Semantic similarity", f"{result.embedding_similarity:.0%}")
        m2.metric("Skill coverage", f"{result.keyword_coverage:.0%}")
        m3.metric("Entity match", f"{result.entity_match_score:.0%}")
        st.markdown('<div class="rr-card"><div class="rr-card-title">Strongest evidence · Matched skills</div>', unsafe_allow_html=True)
        skill_pills(result.matched_skills)
        st.markdown('<div class="rr-divider"></div><div class="rr-card-title">Priority gaps · Missing skills</div>', unsafe_allow_html=True)
        skill_pills(result.missing_skills, warning=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("View score breakdown and extracted entities"):
            st.json({"weights": result.weights_used, "extracted_entities": result.extracted_entities})
    with right:
        st.markdown('<div class="rr-card"><div style="display:flex;align-items:center;gap:10px;margin-bottom:14px"><div class="rr-mark" style="width:33px;height:33px;font-size:17px">✳</div><div><div style="font-weight:750;color:#37314d">ResumeIQ Assistant</div><div class="rr-muted">Resume review and tailoring</div></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rr-chat"><div class="rr-chat-label">YOUR REQUEST</div><div style="font-size:12px;color:#51496d">Review my resume against the selected job description.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="rr-subtitle" style="margin:8px 0 12px">Here is how your resume aligns with the role and where to improve it.</div>', unsafe_allow_html=True)
        st.markdown('<div class="rr-chat"><div class="rr-chat-label">PRIORITY GAPS</div>', unsafe_allow_html=True)
        gaps = insights.get("gaps", []) or []
        if gaps:
            for gap in gaps[:4]:
                st.markdown(f'<div style="font-size:12px;color:#5c5478;padding:5px 0"><span style="color:#d0a166">△</span> <b>{safe(gap.get("title","Review item"))}</b><div style="font-size:11px;color:#8b85a1;padding-left:18px">{safe(gap.get("detail",""))}</div></div>', unsafe_allow_html=True)
        elif result.missing_skills:
            for skill in result.missing_skills[:4]:
                st.markdown(f'<div style="font-size:12px;color:#5c5478;padding:4px 0">△ {safe(skill)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rr-subtitle">No major gaps were returned by the current analysis.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="rr-chat"><div class="rr-chat-label">SUGGESTED IMPROVEMENTS</div>', unsafe_allow_html=True)
        suggestions = insights.get("suggestions", []) or []
        if suggestions:
            for suggestion in suggestions[:3]:
                st.markdown(f'<div style="font-size:12px;color:#5c5478;padding:5px 0">✧ <b>{safe(suggestion.get("title","Improvement"))}</b><div style="font-size:11px;color:#8b85a1;padding-left:18px">{safe(suggestion.get("action",""))}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rr-subtitle">Run the analysis to get suggestions from the existing ResumeIQ advisor.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        sample = insights.get("sample_tailored_resume", "")
        if sample:
            st.download_button("✧ Download tailored sample", sample, file_name=f"resumeiq_{safe(candidate_name).replace(' ','_')}_sample.md", mime="text/markdown", use_container_width=True)
        st.markdown('<div class="rr-muted" style="padding:4px 0 0">Suggestions are drafts. Keep only claims that accurately reflect your experience.</div></div>', unsafe_allow_html=True)

def single_resume_mode():
    header("RESUME WORKSPACE", "Applications", "Compare your resume with a job description and see the evidence behind the match.")
    result_area = st.container()
    with st.form("resumeiq_score_form", clear_on_submit=False):
        main, side = st.columns([1.02, .98], gap="medium")
        with main:
            st.markdown('<div class="rr-card-title">01 · Job description</div>', unsafe_allow_html=True)
            jd_upload = st.file_uploader("Upload job description (.txt)", type=["txt"], key="rr_jd_upload")
            jd_text = st.text_area("Paste the job description", value=read_upload(jd_upload) if jd_upload else "", height=205, placeholder="Paste the role requirements, responsibilities, and preferred skills…", key="rr_jd_text")
        with side:
            st.markdown('<div class="rr-card-title">02 · Resume document</div>', unsafe_allow_html=True)
            resume_upload = st.file_uploader("Upload resume (.pdf or .txt)", type=["pdf", "txt"], key="rr_resume_upload")
            resume_text = st.text_area("Or paste resume text", value=read_upload(resume_upload) if resume_upload else "", height=205, placeholder="Paste your resume text or upload a PDF/TXT file…", key="rr_resume_text")
        candidate = st.text_input("Candidate name (optional)", value="Candidate", key="rr_candidate")
        submitted = st.form_submit_button("✧ Analyze resume", type="primary", use_container_width=True)
    if submitted:
        if not jd_text.strip() or not resume_text.strip():
            st.error("Please provide both a resume and a job description.")
        elif not st.session_state.get("rr_weights_valid", True):
            st.error("Adjust the scoring weights in the sidebar so they total 100%.")
        else:
            try:
                weights = ScoringWeights(st.session_state["rr_w_skills"], st.session_state["rr_w_exp"], st.session_state["rr_w_edu"])
                with st.spinner("Extracting signals and scoring the resume…"):
                    result = engine.score(resume_text, jd_text, candidate.strip() or "Candidate", weights)
                st.session_state["rr_latest_result"] = result
                st.session_state["rr_latest_candidate"] = candidate.strip() or "Candidate"
            except Exception as exc:
                st.error(f"Resume analysis failed: {exc}")
    with result_area:
        result = st.session_state.get("rr_latest_result")
        if result:
            st.markdown("---")
            header("ANALYSIS RESULTS", "Your match, explained.", "Evidence and suggested improvements returned by the existing scoring engine.")
            render_result(result, st.session_state.get("rr_latest_candidate", "Candidate"))
        else:
            st.markdown('<div class="rr-card"><div class="rr-card-title">YOUR ANALYSIS WILL APPEAR HERE</div><div style="font-size:18px;font-weight:750;color:#403953">Upload a resume and add a job description.</div><div class="rr-subtitle" style="margin-top:7px">ResumeIQ will extract text, calculate the match score, identify matched and missing skills, and show the existing advisor’s suggestions.</div></div>', unsafe_allow_html=True)

def batch_ranking_mode():
    header("CANDIDATE WORKSPACE", "Batch ranking", "Rank multiple resumes against the same role using ResumeIQ’s existing scoring engine.")
    jd_upload = st.file_uploader("Upload job description (.txt)", type=["txt"], key="rr_batch_jd_upload")
    jd_text = st.text_area("Job description", value=read_upload(jd_upload) if jd_upload else "", height=170, key="rr_batch_jd")
    files = st.file_uploader("Upload resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True, key="rr_batch_files")
    if st.button("Rank candidates", type="primary", use_container_width=True):
        if not jd_text.strip() or not files:
            st.error("Add a job description and at least one resume.")
        elif not st.session_state.get("rr_weights_valid", True):
            st.error("Adjust the scoring weights in the sidebar so they total 100%.")
        else:
            try:
                candidates = []
                for file in files:
                    text = read_upload(file)
                    if text.strip():
                        candidates.append((os.path.splitext(file.name)[0], text))
                if not candidates:
                    st.error("No readable resume text was found.")
                else:
                    weights = ScoringWeights(st.session_state["rr_w_skills"], st.session_state["rr_w_exp"], st.session_state["rr_w_edu"])
                    with st.spinner(f"Ranking {len(candidates)} resumes…"):
                        ranked = engine.rank(candidates, jd_text, weights)
                    df = pd.DataFrame([{"Rank": i + 1, "Candidate": r.candidate_name, "Match score": round(r.final_score, 1), "Semantic similarity": f"{r.embedding_similarity:.0%}", "Skill coverage": f"{r.keyword_coverage:.0%}", "Matched skills": len(r.matched_skills), "Missing skills": len(r.missing_skills)} for i, r in enumerate(ranked)])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button("Download ranking CSV", df.to_csv(index=False).encode("utf-8"), file_name="resumeiq_rankings.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Ranking failed: {exc}")

def main():
    page, skills, exp, edu, valid = show_sidebar()
    st.session_state["rr_weights_valid"] = valid
    if page == "Settings":
        header("PREFERENCES", "Scoring settings", "Tune the weighting used by the existing resume matching engine.")
        st.info(f"Current weights — Skills {skills:.0%}, Experience {exp:.0%}, Education {edu:.0%}. Total: {skills + exp + edu:.0%}.")
        st.markdown("The score is a decision-support signal. It should not be the sole basis for hiring decisions.")
    elif page == "Analytics":
        header("INSIGHTS", "Model evaluation", "Reference benchmark values documented in the project repository.")
        st.caption("These are Phase 1 benchmark values from the repository README, not live measurements from this session.")
        st.dataframe(pd.DataFrame([
            {"Model": "Logistic Regression", "Accuracy": 0.7565, "Precision": 0.7865, "Recall": 0.7042, "F1": 0.7431},
            {"Model": "Random Forest", "Accuracy": 0.8179, "Precision": 0.8319, "Recall": 0.7968, "F1": 0.8140},
            {"Model": "XGBoost", "Accuracy": 0.8159, "Precision": 0.8369, "Recall": 0.7847, "F1": 0.8100},
        ]), use_container_width=True, hide_index=True)
    elif page == "Documents":
        header("DOCUMENTS", "Resume documents", "Upload PDF/TXT resumes from the Applications workspace to extract and compare text.")
        st.info("Resume files are parsed during analysis. Use Applications to score a single resume, or Jobs to rank multiple resumes.")
    elif page == "Jobs":
        batch_ranking_mode()
    elif page == "Applications":
        single_resume_mode()
    else:
        header("OVERVIEW", "Your resume workspace.", "Tailor your resume to a role with evidence-based scoring and clear improvement suggestions.")
        a, b = st.columns([1.25, .75], gap="medium")
        with a:
            st.markdown('<div class="rr-card"><div class="rr-card-title">GET STARTED</div><div style="font-size:20px;font-weight:750;color:#403953">Find the strongest version of your resume.</div><div class="rr-subtitle" style="margin:8px 0 16px">Start with a job description and your current resume. ResumeIQ compares skills, semantic relevance, and extracted entities.</div></div>', unsafe_allow_html=True)
            st.button("Start resume analysis →", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({"rr_nav": "Applications"}))
        with b:
            st.markdown('<div class="rr-card"><div class="rr-card-title">WHAT RESUMEIQ CHECKS</div><div style="padding:7px 0;color:#5c5478;font-size:13px">✓ Semantic relevance</div><div style="padding:7px 0;color:#5c5478;font-size:13px">✓ Skill coverage and gaps</div><div style="padding:7px 0;color:#5c5478;font-size:13px">✓ Entity and education signals</div><div style="padding:7px 0;color:#5c5478;font-size:13px">✓ Advisor suggestions and rewrite examples</div></div>', unsafe_allow_html=True)
        result = st.session_state.get("rr_latest_result")
        if result:
            st.markdown("---")
            header("LATEST ANALYSIS", "Continue where you left off.", "Latest result retained in this Streamlit session.")
            render_result(result, st.session_state.get("rr_latest_candidate", "Candidate"))

if __name__ == "__main__":
    main()
