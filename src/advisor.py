"""
advisor.py — AI Resume Diagnosis, Gap Analysis, and Actionable Optimizer.

Analyzes what a resume lacks, detects weak phrasing and unquantified duties,
provides step-by-step recommendations with before/after examples, and generates
a tailored sample resume designed for a 95+ ATS score on the target job.
"""

import re
from typing import Optional

# Strong proactive action verbs for tech and professional resumes
POWER_VERBS = {
    "architected", "engineered", "spearheaded", "orchestrated", "streamlined",
    "optimized", "deployed", "scaled", "automated", "designed", "implemented",
    "led", "overhauled", "developed", "built", "accelerated", "reduced",
    "increased", "boosted", "delivered", "mentored", "championed", "integrated",
    "refactored", "migrated", "configured", "published", "pioneered", "resolved",
}

# Weak, passive, or vague phrases that weaken a resume
WEAK_PHRASES = [
    r"\bresponsible for\b",
    r"\bduties included\b",
    r"\bworked on\b",
    r"\bassisted with\b",
    r"\bhelped with\b",
    r"\bparticipated in\b",
    r"\btasked with\b",
    r"\bhandled\b",
    r"\binvolved in\b",
]

# Patterns detecting quantified metrics (numbers, percentages, currency, multipliers, scale)
METRIC_PATTERNS = re.compile(
    r"(\b\d+(\.\d+)?%|"                     # 45%, 99.9%
    r"\$\d+([,\.]\d+)?\s*(k|m|b|million|billion)?|"  # $500k, $2M
    r"\b\d+\s*x\b|"                         # 2x, 10x
    r"\b\d+\+?\s*(users|clients|customers|requests|transactions|rps|qps|tb|gb|queries|endpoints|services|microservices|engineers|developers|members)\b|"
    r"\b(reduced|increased|boosted|cut|saved|accelerated|improved)\b[^\.\n]*?\b\d+)",
    re.IGNORECASE,
)


class ResumeAdvisor:
    """
    Evaluates resumes beyond raw keyword counting:
      1. What it lacks (Hard skills, quantified achievements, active language, degree/seniority)
      2. Concrete suggested changes (Google X-Y-Z formula before/after fixes)
      3. A tailored 95+ sample resume template for the given JD
    """

    def analyze(
        self,
        resume_text: str,
        jd_text: str,
        matched_skills: list[str],
        missing_skills: list[str],
        resume_entities: dict,
        jd_entities: dict,
    ) -> dict:
        """Complete diagnosis of gaps, quality metrics, suggestions, and tailored sample."""
        # 1. Bullet point and metric analysis
        lines = [line.strip("- *•\t ") for line in resume_text.splitlines() if len(line.strip()) > 20]
        total_bullets = len(lines)
        quantified_bullets = [line for line in lines if METRIC_PATTERNS.search(line)]
        metric_ratio = (len(quantified_bullets) / total_bullets) if total_bullets > 0 else 0.0

        # 2. Action verb analysis
        resume_lower = resume_text.lower()
        power_verbs_found = sorted({v for v in POWER_VERBS if re.search(r"\b" + v + r"\b", resume_lower)})
        weak_phrases_found = sorted({
            re.search(pat, resume_lower).group()
            for pat in WEAK_PHRASES
            if re.search(pat, resume_lower)
        })

        # 3. Contextual skill placement check (anti-keyword stuffing)
        # Check if skills appear in descriptive sentences vs just in a flat list
        stuffed_skills = []
        for s in matched_skills:
            # Look for skill followed or preceded by action context
            if not re.search(r"(developed|built|using|with|via|in|engineered|deployed|designed)[^\.\n]*?" + re.escape(s), resume_lower):
                stuffed_skills.append(s)

        # 4. Target role / title
        jd_titles = jd_entities.get("JOB_TITLE", [])
        target_role = jd_titles[0].title() if jd_titles else "Target Role"

        # 5. Compile gaps ("What It Lacks")
        gaps = self._compile_gaps(
            missing_skills=missing_skills,
            metric_ratio=metric_ratio,
            weak_phrases_found=weak_phrases_found,
            resume_entities=resume_entities,
            jd_entities=jd_entities,
        )

        # 6. Actionable suggestions
        suggestions = self._generate_suggestions(
            missing_skills=missing_skills,
            weak_phrases_found=weak_phrases_found,
            metric_ratio=metric_ratio,
            target_role=target_role,
        )

        # 7. Sample tailored resume template
        sample_resume = self._generate_sample_resume(
            jd_text=jd_text,
            target_role=target_role,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )

        return {
            "gaps": gaps,
            "suggestions": suggestions,
            "impact_metrics": {
                "total_experience_lines": total_bullets,
                "quantified_lines_count": len(quantified_bullets),
                "metric_percentage": round(metric_ratio * 100, 1),
                "metric_grade": "Strong" if metric_ratio >= 0.5 else ("Moderate" if metric_ratio >= 0.25 else "Low"),
                "power_verbs_count": len(power_verbs_found),
                "power_verbs_sample": power_verbs_found[:6],
                "weak_phrases_detected": weak_phrases_found,
            },
            "sample_tailored_resume": sample_resume,
        }

    def _compile_gaps(
        self,
        missing_skills: list[str],
        metric_ratio: float,
        weak_phrases_found: list[str],
        resume_entities: dict,
        jd_entities: dict,
    ) -> list[dict]:
        """Categorize what the resume lacks with severity levels."""
        gaps = []

        # Skill gaps
        if missing_skills:
            gaps.append({
                "category": "Missing Core Skills",
                "severity": "High" if len(missing_skills) >= 3 else "Medium",
                "title": f"{len(missing_skills)} Required Skills Not Found",
                "detail": f"The job description highlights {', '.join(missing_skills[:6])}, which are not detected in your resume.",
            })

        # Metric & impact gaps
        if metric_ratio < 0.25:
            gaps.append({
                "category": "Quantified Impact Deficiency",
                "severity": "High",
                "title": "Low Measurable Impact (< 25% quantified bullets)",
                "detail": "Most bullet points describe daily duties rather than measurable results. Modern ATS and recruiters favor resumes that quantify scale, latency reductions, cost savings, or user impact.",
            })
        elif metric_ratio < 0.50:
            gaps.append({
                "category": "Quantified Impact Opportunity",
                "severity": "Medium",
                "title": "Moderate Metric Coverage",
                "detail": f"Only {metric_ratio:.0%} of your bullet points contain numerical results. Adding numbers, percentages, or team sizes will elevate your ranking.",
            })

        # Passive / weak phrasing
        if weak_phrases_found:
            gaps.append({
                "category": "Passive Phrasing Detected",
                "severity": "Medium",
                "title": f"Found Passive Duty Statements ({', '.join(weak_phrases_found[:3])})",
                "detail": "Phrases like 'responsible for' or 'worked on' signal passive participation. Top-scoring resumes use proactive power verbs (e.g. 'Architected', 'Streamlined', 'Deployed').",
            })

        # Degree mismatch
        r_deg = resume_entities.get("DEGREE", [])
        j_deg = jd_entities.get("DEGREE", [])
        if j_deg and not r_deg:
            gaps.append({
                "category": "Education Requirement",
                "severity": "High",
                "title": f"Required Degree ({', '.join(j_deg).title()}) Not Detected",
                "detail": "The JD requests a specific degree credential that was not detected in your education section.",
            })

        return gaps

    def _generate_suggestions(
        self,
        missing_skills: list[str],
        weak_phrases_found: list[str],
        metric_ratio: float,
        target_role: str,
    ) -> list[dict]:
        """Produce concrete, actionable before/after improvement recommendations."""
        suggestions = []

        # Suggestion 1: Integrating missing skills
        if missing_skills:
            skills_sample = missing_skills[:4]
            suggestions.append({
                "title": f"Incorporate Missing Skills into Project Context",
                "action": f"Add {', '.join(skills_sample)} into real accomplishment bullets rather than just listing them in a skills block.",
                "before": "Skills: Python, SQL.",
                "after": f"Engineered scalable microservices using Python and {skills_sample[0]}, integrating with {skills_sample[1] if len(skills_sample) > 1 else 'cloud APIs'} to process 50k+ daily events.",
            })

        # Suggestion 2: Applying the Google X-Y-Z formula for metrics
        if metric_ratio < 0.5:
            suggestions.append({
                "title": "Apply the Google X-Y-Z Impact Formula",
                "action": "Rewrite duty statements to follow: 'Accomplished [X], as measured by [Y], by doing [Z]'.",
                "before": "Responsible for optimizing SQL queries and database caching.",
                "after": "Reduced API p99 latency by 42% across 10M monthly requests by optimizing SQL queries and implementing Redis caching.",
            })

        # Suggestion 3: Replacing passive verbs with power verbs
        if weak_phrases_found:
            sample_weak = weak_phrases_found[0]
            suggestions.append({
                "title": f"Swap Passive '{sample_weak.title()}' with Power Action Verbs",
                "action": "Begin every single experience bullet with a past-tense power verb (e.g., Architected, Spearheaded, Overhauled).",
                "before": f"{sample_weak.capitalize()} developing the backend service and fixing pipeline bugs.",
                "after": "Architected resilient backend services and automated CI/CD pipelines, cutting deployment failure rates by 35%.",
            })

        return suggestions

    def _generate_sample_resume(
        self,
        jd_text: str,
        target_role: str,
        matched_skills: list[str],
        missing_skills: list[str],
    ) -> str:
        """Create a tailored, high-scoring sample resume in clean Markdown."""
        all_skills = sorted(set(matched_skills + missing_skills))
        top_skills = all_skills[:10] if all_skills else ["Python", "Docker", "Kubernetes", "AWS", "REST APIs", "PostgreSQL"]

        skills_formatted = ", ".join(s.title() for s in top_skills)
        primary_skill = top_skills[0].title() if top_skills else "Python"
        secondary_skill = top_skills[1].title() if len(top_skills) > 1 else "Cloud"
        database_skill = next((s.title() for s in top_skills if s in {"Postgresql", "Mysql", "Mongodb", "Redis"}), "PostgreSQL")
        cloud_skill = next((s.title() for s in top_skills if s in {"Aws", "Gcp", "Azure", "Docker", "Kubernetes"}), "AWS")

        template = f"""# ALEX MORGAN
**{target_role.upper()} | CLOUD & BACKEND SYSTEMS**  
San Francisco, CA • alex.morgan@email.com • (555) 019-2834 • [linkedin.com/in/alexmorgan](https://linkedin.com) • [github.com/alexmorgan](https://github.com)

---

### PROFESSIONAL SUMMARY
High-impact **{target_role}** with 5+ years of experience designing, scaling, and deploying mission-critical distributed systems. Proven track record leveraging **{skills_formatted}** to accelerate throughput by 3x and cut cloud infrastructure costs by 30%. Expert in building resilient microservices and CI/CD automation with 99.99% availability.

---

### CORE COMPETENCIES
- **Languages & Frameworks:** {skills_formatted}
- **Architecture & Practices:** Microservices, System Design, REST APIs, GraphQL, TDD, Agile/Scrum
- **Cloud & DevOps:** {cloud_skill}, Docker, Kubernetes, Terraform, CI/CD Pipelines, Linux
- **Databases & Storage:** {database_skill}, Redis Caching, High-Volume Data Pipelines

---

### PROFESSIONAL EXPERIENCE

**SENIOR {target_role.upper()}** | TechCorp Solutions Inc.  
*2022 – Present | San Francisco, CA*
- **Architected and scaled** high-throughput backend services using **{primary_skill}** and **{secondary_skill}**, processing over **15M daily requests** with sub-50ms latency.
- **Spearheaded migration** to containerized architecture with **Docker** and **Kubernetes** on **{cloud_skill}**, boosting deployment velocity by **45%** and reducing downtime to zero.
- **Optimized data access layers** with **{database_skill}** and distributed caching, slashing p95 response times by **40%** across 20+ microservice endpoints.
- **Automated end-to-end CI/CD pipelines**, reducing release turnaround from 2 days to **15 minutes** while upholding rigorous test coverage (>90%).

**BACKEND SOFTWARE ENGINEER** | Nexus Data Systems  
*2019 – 2022 | Austin, TX*
- **Engineered secure RESTful APIs** utilizing **{primary_skill}**, serving **2.5M active mobile and web users**.
- **Refactored legacy database queries** in **{database_skill}**, cutting query execution overhead by **35%** and saving **$120,000 annually** in compute costs.
- **Mentored team of 4 junior developers** in clean architecture, code reviews, and automated integration testing.

---

### EDUCATION
**Bachelor of Science in Computer Science**  
*State University of Technology — Magna Cum Laude*
"""
        return template.strip()
