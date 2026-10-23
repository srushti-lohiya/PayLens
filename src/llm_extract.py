"""Step 6: LLM feature extraction from job descriptions.

The LLM reads each description and returns validated JSON (Pydantic). Results are
cached in data/processed/llm_features.jsonl so you pay for each posting once.

Run:
  python -m src.llm_extract --limit 20        # try on 20 postings first
  python -m src.llm_extract                   # all postings
  python -m src.llm_extract --evaluate        # compare: keywords vs +LLM features vs LLM-only guess
"""
import argparse, hashlib, json
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import LLM_CACHE, N_SPLITS
from src.features import NUMERIC
from src.llm import ask_json, llm_available
from src.models import text_preprocessor
from src.train import load_xy


class JobSignals(BaseModel):
    min_years_experience: Optional[float] = Field(None, ge=0, le=30)
    degree_required: Literal["none", "bachelors", "masters", "phd", "unspecified"] = "unspecified"
    seniority: Literal["intern", "junior", "mid", "senior", "lead", "manager", "unspecified"] = "unspecified"
    remote: Literal["remote", "hybrid", "onsite", "unspecified"] = "unspecified"
    security_clearance: bool = False
    people_management: bool = False
    domain: str = "other"
    skills: List[str] = []
    llm_salary_guess_k: Optional[float] = Field(None, ge=10, le=500)


SYSTEM = ("You extract structured facts from job postings. Use 'unspecified' or null when the "
          "posting does not say. Do not guess skills that are not mentioned.")
PROMPT = """Extract these fields from the job posting:
min_years_experience (number or null), degree_required (none|bachelors|masters|phd|unspecified),
seniority (intern|junior|mid|senior|lead|manager|unspecified), remote (remote|hybrid|onsite|unspecified),
security_clearance (bool), people_management (bool), domain (one or two words, e.g. healthcare, finance),
skills (list of lowercase tool/skill names, max 15),
llm_salary_guess_k (your best estimate of the annual US salary in thousands of USD, around 2019).

JOB TITLE: {title}
LOCATION: {location}
DESCRIPTION:
{description}"""


def key_of(row) -> str:
    return hashlib.md5((str(row["Job Title"]) + str(row["Job Description"])).encode()).hexdigest()


def load_cache() -> dict:
    if not LLM_CACHE.exists():
        return {}
    return {json.loads(l)["key"]: json.loads(l) for l in LLM_CACHE.read_text().splitlines() if l.strip()}


def extract_one(title, location, description) -> JobSignals:
    raw = ask_json(PROMPT.format(title=title, location=location, description=description[:6000]), SYSTEM)
    return JobSignals.model_validate(raw)


def run_extraction(df, limit=None):
    if not llm_available():
        raise SystemExit("Set ANTHROPIC_API_KEY first (export ANTHROPIC_API_KEY=...).")
    cache = load_cache()
    todo = [r for _, r in df.iterrows() if key_of(r) not in cache][:limit]
    print(f"{len(cache)} cached, extracting {len(todo)} ...")
    with LLM_CACHE.open("a") as f:
        for i, r in enumerate(todo, 1):
            try:
                sig = extract_one(r["Job Title"], r["Location"], r["Job Description"])
                f.write(json.dumps({"key": key_of(r), **sig.model_dump()}) + "\n"); f.flush()
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                print(f"  skipped row ({type(e).__name__}): {str(e)[:80]}")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")


LLM_NUMERIC = ["llm_years", "llm_degree", "llm_seniority", "llm_clearance", "llm_manages", "llm_n_skills"]
DEGREE = {"none": 0, "bachelors": 1, "masters": 2, "phd": 3}
SEN = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "manager": 4}


def add_llm_features(df_raw, X):
    cache = load_cache()
    rows = [cache.get(key_of(r), {}) for _, r in df_raw.iterrows()]
    X = X.copy()
    X["llm_years"] = [r.get("min_years_experience") for r in rows]
    X["llm_degree"] = [DEGREE.get(r.get("degree_required")) for r in rows]
    X["llm_seniority"] = [SEN.get(r.get("seniority")) for r in rows]
    X["llm_clearance"] = [float(r["security_clearance"]) if "security_clearance" in r else np.nan for r in rows]
    X["llm_manages"] = [float(r["people_management"]) if "people_management" in r else np.nan for r in rows]
    X["llm_n_skills"] = [len(r["skills"]) if "skills" in r else np.nan for r in rows]
    X["llm_guess"] = [r.get("llm_salary_guess_k") for r in rows]
    for c in LLM_NUMERIC + ["llm_guess"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def evaluate():
    df, X, y, groups = load_xy()
    X = add_llm_features(df, X)
    have = X["llm_guess"].notna().values
    if have.sum() < 50:
        raise SystemExit(f"Only {have.sum()} rows extracted; run extraction on more rows first.")
    X, y, groups = X[have].reset_index(drop=True), y[have], groups[have]
    cv = GroupKFold(N_SPLITS)

    def rmse(numeric):
        pipe = Pipeline([("prep", text_preprocessor(numeric=numeric)), ("model", Ridge(alpha=3))])
        return -cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="neg_root_mean_squared_error").mean()

    llm_only = float(np.sqrt(np.mean((X["llm_guess"].values - y) ** 2)))
    res = {"rows": int(len(y)), "llm_only_rmse": llm_only,
           "ml_keywords_rmse": rmse(NUMERIC), "ml_keywords+llm_rmse": rmse(NUMERIC + LLM_NUMERIC)}
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--evaluate", action="store_true")
    a = p.parse_args()
    if a.evaluate:
        evaluate()
    else:
        df, _, _, _ = load_xy()
        run_extraction(df, a.limit)
