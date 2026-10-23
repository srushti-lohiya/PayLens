"""Step 2: turn raw columns into model features.

build_features() works on the ORIGINAL Glassdoor column names, so the same function
is used for training, the API and the Streamlit app (no train/serve skew).
Nothing here looks at the target or at other rows, so it is safe before CV.
"""
import re
import numpy as np
import pandas as pd

SIZE_MAP = {"1 to 50 employees": 1, "51 to 200 employees": 2, "201 to 500 employees": 3,
            "501 to 1000 employees": 4, "1001 to 5000 employees": 5,
            "5001 to 10000 employees": 6, "10000+ employees": 7}
REVENUE_MAP = {"Less than $1 million (USD)": 1, "$1 to $5 million (USD)": 2,
               "$5 to $10 million (USD)": 3, "$10 to $25 million (USD)": 4,
               "$25 to $50 million (USD)": 5, "$50 to $100 million (USD)": 6,
               "$100 to $500 million (USD)": 7, "$500 million to $1 billion (USD)": 8,
               "$1 to $2 billion (USD)": 9, "$2 to $5 billion (USD)": 10,
               "$5 to $10 billion (USD)": 11, "$10+ billion (USD)": 12}
SKILLS = {"python": r"\bpython\b", "r_lang": r"\br\b(?![&\-])", "sql": r"\bsql\b",
          "spark": r"\bspark\b", "aws": r"\baws\b", "excel": r"\bexcel\b",
          "tableau": r"\btableau\b", "deep_learning": r"tensorflow|pytorch|keras|deep learning",
          "phd": r"\bph\.?d\b", "masters": r"master'?s|\bm\.s\.", "clearance": r"clearance"}

NUMERIC = ["company_rating", "company_age", "company_size", "revenue", "n_competitors",
           "desc_len", "job_in_hq_state", "seniority"] + [f"skill_{k}" for k in SKILLS]
CATEGORICAL = ["title_simple", "state", "sector", "ownership", "industry"]
TEXT = "description"
CURRENT_YEAR = 2020  # data was scraped around 2019/2020


def simplify_title(t: str) -> str:
    t = t.lower()
    if "director" in t: return "director"
    if "manager" in t: return "manager"
    if "machine learning" in t or "ml engineer" in t: return "mle"
    if "data engineer" in t: return "data engineer"
    if "analyst" in t: return "analyst"
    if "scientist" in t: return "data scientist"
    return "other"


def seniority(t: str) -> int:
    t = t.lower()
    if re.search(r"\b(sr|senior|lead|principal|staff|director|head)\b", t): return 2
    if re.search(r"\b(jr|junior|associate|entry|intern)\b", t): return 0
    return 1


_MISSING = {"-1", "-1.0", "Unknown", "Unknown / Non-Applicable", "", "nan", "None"}


def _na(x):
    return np.nan if str(x).strip() in _MISSING else x


def state_of(loc) -> str:
    loc = str(loc)
    return loc.split(",")[-1].strip() if "," in loc else loc.strip()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    title = raw["Job Title"].fillna("").astype(str)
    desc = raw["Job Description"].fillna("").astype(str)
    desc_l = desc.str.lower()

    df["title_simple"] = title.apply(simplify_title)
    df["seniority"] = title.apply(seniority)
    df["company_rating"] = pd.to_numeric(raw["Rating"].map(_na), errors="coerce")
    founded = pd.to_numeric(raw["Founded"].map(_na), errors="coerce")
    df["company_age"] = CURRENT_YEAR - founded
    df["company_size"] = raw["Size"].map(SIZE_MAP)
    df["revenue"] = raw["Revenue"].map(REVENUE_MAP)
    comp = raw["Competitors"].fillna("-1").astype(str)
    df["n_competitors"] = comp.apply(lambda c: 0 if c.strip() == "-1" else len(c.split(",")))
    df["state"] = raw["Location"].apply(state_of)
    df["job_in_hq_state"] = (raw["Location"].apply(state_of)
                             == raw["Headquarters"].apply(state_of)).astype(int)
    for col, src in [("sector", "Sector"), ("ownership", "Type of ownership"), ("industry", "Industry")]:
        df[col] = raw[src].map(_na).astype("object").fillna("missing").astype(str)
    df["desc_len"] = desc.str.len()
    for k, pat in SKILLS.items():
        df[f"skill_{k}"] = desc_l.str.contains(pat, regex=True).astype(int)
    df[TEXT] = title + ". " + desc
    return df
