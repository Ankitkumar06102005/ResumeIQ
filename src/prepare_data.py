"""
prepare_data.py — Convert the Kaggle Resume Dataset into resume-JD pairs.

Strategy:
  - For each category, we have a template JD.
  - Positive pair: resume from category X  +  JD for category X  → label=1
  - Negative pair: resume from category X  +  JD for category Y  → label=0
  - We balance classes so each category contributes equally.

Output: data/processed/resume_jd_pairs.csv (resume_text, jd_text, label)

Run:
    python src/prepare_data.py
"""

import os
import sys
import re
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

INPUT_PATH = "Resume.csv"
OUTPUT_PATH = "data/processed/resume_jd_pairs.csv"

# ---------------------------------------------------------------------------
# Template JDs — one realistic JD per category
# Intentionally concise and keyword-rich so TF-IDF has clean signal.
# ---------------------------------------------------------------------------
CATEGORY_JDS = {
    "INFORMATION-TECHNOLOGY": """
        We are looking for a Software Engineer / IT professional with strong experience in Python,
        Java, or C++. Required skills: SQL databases, REST APIs, cloud platforms (AWS or Azure),
        Linux, Git, Docker, Agile development. Nice to have: Kubernetes, CI/CD pipelines, microservices
        architecture, data structures, algorithms. BS/MS in Computer Science or related field.
    """,
    "BUSINESS-DEVELOPMENT": """
        Seeking a Business Development Manager to drive revenue growth. Must have experience in
        sales strategy, lead generation, CRM software, market research, partnership development,
        and contract negotiation. Strong communication and presentation skills required.
        MBA preferred. Experience with Salesforce and Excel a plus.
    """,
    "FINANCE": """
        Financial Analyst position open. CPA or CFA preferred. Required: financial modeling,
        Excel, SQL, budgeting, forecasting, P&L analysis, GAAP accounting, ERP systems (SAP, Oracle).
        Experience in investment analysis, portfolio management, or corporate finance. MBA a plus.
    """,
    "ADVOCATE": """
        Seeking a licensed Attorney / Legal Advocate. Juris Doctor (JD) required. Experience in
        litigation, legal research, drafting contracts, client counseling. Proficiency with legal
        databases (Westlaw, LexisNexis). Specialization in corporate, criminal, or civil law preferred.
    """,
    "ACCOUNTANT": """
        Looking for a Certified Public Accountant (CPA). Skills required: bookkeeping, tax preparation,
        auditing, QuickBooks, SAP, financial reporting, accounts payable/receivable, GAAP compliance,
        Excel, payroll processing. Bachelor's degree in Accounting required.
    """,
    "ENGINEERING": """
        Mechanical / Civil Engineer needed. Required: AutoCAD, SolidWorks, project management,
        structural analysis, technical drawings, MATLAB, quality control. PE license preferred.
        Experience in construction, manufacturing, or infrastructure. BS in Engineering required.
    """,
    "CHEF": """
        Head Chef / Culinary professional. Must have experience in menu planning, food preparation,
        kitchen management, inventory control, HACCP food safety standards, team leadership.
        Culinary degree or equivalent experience. Fine dining or restaurant chain experience preferred.
    """,
    "AVIATION": """
        Aviation professional / Pilot. FAA Airline Transport Pilot (ATP) certificate required.
        Experience flying commercial aircraft, instrument rating, multi-engine rating, crew resource
        management (CRM), flight planning, safety management systems. Hours requirement: 1500+.
    """,
    "FITNESS": """
        Certified Personal Trainer / Fitness Coach. NASM, ACE, or NSCA certification required.
        Experience designing fitness programs, nutrition coaching, group fitness instruction,
        strength and conditioning, injury prevention, client assessment and progress tracking.
    """,
    "SALES": """
        Sales Representative / Account Executive. Proven track record in B2B or B2C sales,
        cold calling, pipeline management, CRM (Salesforce), quota attainment, negotiation,
        product demos. Experience in SaaS, retail, or pharmaceutical sales a plus.
    """,
    "BANKING": """
        Retail / Commercial Banking professional. Experience with loan origination, credit analysis,
        risk management, KYC/AML compliance, financial products, Bloomberg terminal, regulatory
        reporting. CFA or FRM preferred. Strong analytical and client relationship skills.
    """,
    "HEALTHCARE": """
        Healthcare professional (RN, NP, or Physician). Required: patient care, clinical documentation,
        EMR systems (Epic, Cerner), HIPAA compliance, medical procedures, diagnostics.
        Board certification preferred. Experience in hospital, clinic, or telehealth setting.
    """,
    "CONSULTANT": """
        Management Consultant / Strategy Consultant. MBA preferred. Experience in process improvement,
        stakeholder management, data analysis, PowerPoint presentations, project management (PMP),
        change management, Six Sigma or Lean methodology. Big 4 or MBB experience a plus.
    """,
    "CONSTRUCTION": """
        Construction Project Manager. Required: project scheduling (MS Project, Primavera),
        cost estimation, blueprint reading, OSHA safety certification, subcontractor management,
        AutoCAD. PMP or CCM certification a plus. Experience with commercial or residential construction.
    """,
    "PUBLIC-RELATIONS": """
        PR Manager / Communications Specialist. Required: media relations, press release writing,
        crisis communications, social media strategy, brand management, event planning,
        Google Analytics, Adobe Creative Suite. Bachelor's degree in Communications or PR required.
    """,
    "HR": """
        Human Resources Manager / HR Business Partner. Required: talent acquisition, HRIS systems
        (Workday, BambooHR), employee relations, performance management, compensation and benefits,
        labor law compliance, onboarding, ADP payroll. PHR or SHRM certification preferred.
    """,
    "DESIGNER": """
        UX/UI Designer or Graphic Designer. Required: Figma, Adobe XD, Photoshop, Illustrator,
        user research, wireframing, prototyping, design systems, typography, color theory.
        Portfolio required. Experience with web or mobile product design preferred.
    """,
    "ARTS": """
        Creative Arts professional. Experience in visual arts, photography, video production,
        Adobe Premiere, Final Cut Pro, After Effects, illustration, painting, or sculpture.
        Strong portfolio required. Experience with galleries, media production, or art education.
    """,
    "TEACHER": """
        K-12 Teacher or Educator. State teaching certification required. Experience in curriculum
        development, classroom management, lesson planning, student assessment, differentiated
        instruction. Proficiency with Google Classroom, LMS platforms. Master's degree preferred.
    """,
    "APPAREL": """
        Fashion / Apparel professional. Experience in garment construction, textile sourcing,
        trend forecasting, Adobe Illustrator for tech packs, fashion merchandising, supply chain
        management, retail buying or brand management. Degree in Fashion Design or Merchandising.
    """,
    "DIGITAL-MEDIA": """
        Digital Marketing / Content Creator. Required: SEO, Google Analytics, social media
        management (Facebook Ads, Instagram, TikTok), email marketing, content strategy,
        HubSpot or Marketo, video editing, copywriting. Data-driven mindset preferred.
    """,
    "AGRICULTURE": """
        Agricultural Scientist / Farm Manager. Experience with crop management, soil science,
        irrigation systems, precision agriculture (GIS, GPS), pesticide application, USDA
        regulations, livestock management. BS in Agriculture or related field required.
    """,
    "AUTOMOBILE": """
        Automotive Engineer / Technician. ASE certification preferred. Experience with vehicle
        diagnostics, engine repair, CAD software for automotive design, EV systems, ADAS
        technologies, manufacturing QC. BS in Mechanical or Automotive Engineering preferred.
    """,
    "BPO": """
        BPO / Call Center professional. Experience in customer service operations, quality
        assurance, workforce management, CRM tools (Zendesk, Salesforce), data entry, KPI
        reporting, SLA management. Team lead or supervisor experience a plus.
    """,
}


def clean_resume(text: str) -> str:
    """Strip HTML tags and excessive whitespace from resume text."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pairs(df: pd.DataFrame, neg_ratio: float = 1.0) -> pd.DataFrame:
    """
    For every resume, create one positive pair and `neg_ratio` negative pairs.
    """
    categories = df["Category"].unique().tolist()
    records = []

    for _, row in df.iterrows():
        cat = row["Category"]
        resume = clean_resume(row["Resume_str"])

        if not resume or cat not in CATEGORY_JDS:
            continue

        # Positive pair
        records.append({
            "resume_text": resume,
            "jd_text": CATEGORY_JDS[cat].strip(),
            "label": 1,
            "resume_category": cat,
            "jd_category": cat,
        })

        # Negative pair(s) — random different category
        neg_cats = [c for c in categories if c != cat and c in CATEGORY_JDS]
        for neg_cat in random.sample(neg_cats, min(int(neg_ratio), len(neg_cats))):
            records.append({
                "resume_text": resume,
                "jd_text": CATEGORY_JDS[neg_cat].strip(),
                "label": 0,
                "resume_category": cat,
                "jd_category": neg_cat,
            })

    return pd.DataFrame(records)


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} resumes across {df['Category'].nunique()} categories")

    pairs = build_pairs(df, neg_ratio=1.0)

    pos = (pairs["label"] == 1).sum()
    neg = (pairs["label"] == 0).sum()
    print(f"  Pairs created: {len(pairs)} total | {pos} positive | {neg} negative")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pairs.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved → {OUTPUT_PATH}")

    # Quick sanity check
    sample = pairs.sample(3, random_state=1)
    for _, r in sample.iterrows():
        print(f"\n  [{r['label']}] Resume: {r['resume_category']} | JD: {r['jd_category']}")
        print(f"  Resume (first 80 chars): {r['resume_text'][:80]}...")


if __name__ == "__main__":
    main()
