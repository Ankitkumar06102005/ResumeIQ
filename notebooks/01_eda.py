"""
01_eda.py — Exploratory Data Analysis for the resume-JD dataset.

Run cell-by-cell in VS Code (Python Interactive) or convert to .ipynb with:
    jupytext --to notebook notebooks/01_eda.py
"""

# %%
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.preprocessing import preprocess
from src.features import SKILLS_VOCAB

# %%
df = pd.read_csv("data/processed/resume_jd_pairs.csv")
print(df.shape)
df.head()

# %%
# Class balance
df["label"].value_counts().plot(kind="bar", title="Label Distribution")
plt.tight_layout(); plt.show()

# %%
# Text length distributions
df["resume_len"] = df["resume_text"].str.split().str.len()
df["jd_len"] = df["jd_text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["resume_len"].hist(bins=40, ax=axes[0], title="Resume Word Count")
df["jd_len"].hist(bins=40, ax=axes[1], title="JD Word Count")
plt.tight_layout(); plt.show()

# %%
# Most common skills in resumes
all_resume_text = " ".join(df["resume_text"].fillna(""))
skill_counts = Counter({s: all_resume_text.lower().count(s) for s in SKILLS_VOCAB})
top_skills = pd.DataFrame(skill_counts.most_common(20), columns=["skill", "count"])
top_skills.plot(kind="barh", x="skill", y="count", title="Top 20 Skills in Resumes")
plt.tight_layout(); plt.show()
