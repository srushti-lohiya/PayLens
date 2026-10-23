
"""Step 6: LLM feature extraction from job descriptions.

Uses OpenAI to extract structured job signals validated by Pydantic.
Results are cached locally to avoid repeated API charges.

Run:
    python -m src.llm_extract --limit 1
    python -m src.llm_extract --limit 20
    python -m src.llm_extract
    python -m src.llm_extract --evaluate
"""

import argparse
import hashlib
import json
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from openai import OpenAIError
from pydantic import BaseModel, Field, ValidationError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import LLM_CACHE, N_SPLITS, LLM_MODEL
from src.features import NUMERIC
from src.llm import ask_json, llm_available
from src.models import text_preprocessor
from src.train import load_xy


# --------------------------------------------------
# 1. Structured output schema
# --------------------------------------------------

class JobSignals(BaseModel):
    min_years_experience: Optional[float] = Field(
        None, ge=0, le=30
    )

    degree_required: Literal[
        "none", "bachelors", "masters", "phd", "unspecified"
    ] = "unspecified"

    seniority: Literal[
        "intern", "junior", "mid", "senior",
        "lead", "manager", "unspecified"
    ] = "unspecified"

    remote: Literal[
        "remote", "hybrid", "onsite", "unspecified"
    ] = "unspecified"

    security_clearance: bool = False
    people_management: bool = False
    domain: str = "other"
    skills: List[str] = Field(default_factory=list)

    llm_salary_guess_k: Optional[float] = Field(
        None, ge=10, le=500
    )


# --------------------------------------------------
# 2. Extraction instructions
# --------------------------------------------------

SYSTEM = (
    "You extract structured facts from job postings. "
    "Use 'unspecified' or null when the posting does not say. "
    "Do not invent qualifications, skills, or experience. "
    "Return a single valid JSON object."
)

PROMPT = """Extract these fields from the job posting:

min_years_experience: number or null

degree_required:
none | bachelors | masters | phd | unspecified

seniority:
intern | junior | mid | senior | lead | manager | unspecified

remote:
remote | hybrid | onsite | unspecified

security_clearance: boolean
people_management: boolean

domain:
One or two words, for example healthcare or finance.

skills:
List of lowercase tools and skills explicitly mentioned.
Maximum 15 skills.

llm_salary_guess_k:
Estimate annual US salary around 2019.
Return salary in THOUSANDS of USD.

Examples:
$80,000 = 80
$120,000 = 120
$175,000 = 175

Return a number between 10 and 500, or null.
Never return the full dollar amount.

JOB TITLE:
{title}

LOCATION:
{location}

DESCRIPTION:
{description}
"""


# --------------------------------------------------
# 3. Cache management
# --------------------------------------------------

def key_of(row) -> str:
    content = (
        str(row["Job Title"])
        + str(row["Job Description"])
    )

    return hashlib.md5(
        content.encode("utf-8")
    ).hexdigest()


def load_cache() -> dict:
    if not LLM_CACHE.exists():
        return {}

    cache = {}

    with LLM_CACHE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            cache[record["key"]] = record

    return cache


# --------------------------------------------------
# 4. Extract one job posting
# --------------------------------------------------


def extract_one(title, location, description) -> JobSignals:
    prompt = PROMPT.format(
        title=title,
        location=location,
        description=str(description)[:6000]
    )

    raw = ask_json(
        prompt=prompt,
        system=SYSTEM
    )

    salary = raw.get("llm_salary_guess_k")

    if salary is not None:
        try:
            salary = float(salary)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid salary prediction: {salary!r}"
            )

        # Convert USD to thousands only when the value
        # clearly looks like a full annual salary.
        if 10000 <= salary <= 500000:
            salary = salary / 1000

        raw["llm_salary_guess_k"] = salary

    return JobSignals.model_validate(raw)



# --------------------------------------------------
# 5. Extract and cache job signals
# --------------------------------------------------

def run_extraction(df, limit=None):

    if not llm_available():
        raise SystemExit(
            "OPENAI_API_KEY is missing. "
            "Add it to your local .env file."
        )

    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1.")

    cache = load_cache()

    # Exclude postings already processed or repeated in this batch.
    todo = []
    seen = set(cache)

    for _, row in df.iterrows():
        key = key_of(row)

        if key in seen:
            continue

        seen.add(key)
        todo.append(row)

        if limit is not None and len(todo) >= limit:
            break

    print(f"OpenAI model: {LLM_MODEL}")
    print(f"Cached records: {len(cache)}")
    print(f"New records to extract: {len(todo)}")

    success = 0
    skipped = 0

    with LLM_CACHE.open("a", encoding="utf-8") as file:

        for index, row in enumerate(todo, start=1):

            try:
                signals = extract_one(
                    title=row["Job Title"],
                    location=row["Location"],
                    description=row["Job Description"]
                )

                record = {
                    "key": key_of(row),
                    **signals.model_dump()
                }

                file.write(
                    json.dumps(record, ensure_ascii=False)
                    + "\n"
                )

                file.flush()
                success += 1

            except OpenAIError as error:
                raise SystemExit(
                    f"OpenAI API request failed "
                    f"({type(error).__name__}): {error}"
                ) from error

            except (ValidationError, ValueError) as error:
                skipped += 1
                print(
                    f"Skipped posting {index}: "
                    f"{type(error).__name__}: "
                    f"{str(error)[:120]}"
                )

            if index % 25 == 0:
                print(
                    f"Progress: {index}/{len(todo)}"
                )

    print("\nExtraction completed.")
    print(f"Successful: {success}")
    print(f"Skipped: {skipped}")
    print(f"Cache saved to: {LLM_CACHE}")


# --------------------------------------------------
# 6. Convert LLM outputs into ML features
# --------------------------------------------------

LLM_NUMERIC = [
    "llm_years",
    "llm_degree",
    "llm_seniority",
    "llm_clearance",
    "llm_manages",
    "llm_n_skills"
]

DEGREE = {
    "none": 0,
    "bachelors": 1,
    "masters": 2,
    "phd": 3
}

SEN = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
    "manager": 4
}


def add_llm_features(df_raw, X):

    cache = load_cache()

    rows = [
        cache.get(key_of(row), {})
        for _, row in df_raw.iterrows()
    ]

    X = X.copy()

    X["llm_years"] = [
        row.get("min_years_experience")
        for row in rows
    ]

    X["llm_degree"] = [
        DEGREE.get(row.get("degree_required"))
        for row in rows
    ]

    X["llm_seniority"] = [
        SEN.get(row.get("seniority"))
        for row in rows
    ]

    X["llm_clearance"] = [
        float(row["security_clearance"])
        if "security_clearance" in row
        else np.nan
        for row in rows
    ]

    X["llm_manages"] = [
        float(row["people_management"])
        if "people_management" in row
        else np.nan
        for row in rows
    ]

    X["llm_n_skills"] = [
        len(row["skills"])
        if "skills" in row
        else np.nan
        for row in rows
    ]

    X["llm_guess"] = [
        row.get("llm_salary_guess_k")
        for row in rows
    ]

    for column in LLM_NUMERIC + ["llm_guess"]:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X


# --------------------------------------------------
# 7. Compare LLM and ML performance
# --------------------------------------------------

def evaluate():

    df, X, y, groups = load_xy()

    X = add_llm_features(df, X)

    have = X["llm_guess"].notna().values

    if have.sum() < 50:
        raise SystemExit(
            f"Only {have.sum()} rows have LLM salary guesses. "
            "Extract at least 50 records first."
        )

    X = X[have].reset_index(drop=True)
    y = y[have]
    groups = groups[have]

    cv = GroupKFold(n_splits=N_SPLITS)

    def rmse(numeric):

        pipeline = Pipeline([
            (
                "prep",
                text_preprocessor(numeric=numeric)
            ),
            (
                "model",
                Ridge(alpha=3)
            )
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="neg_root_mean_squared_error"
        )

        return float(-scores.mean())

    llm_only = float(
        np.sqrt(
            np.mean(
                (X["llm_guess"].values - y) ** 2
            )
        )
    )

    results = {
        "rows": int(len(y)),
        "llm_only_rmse": llm_only,
        "ml_keywords_rmse": rmse(NUMERIC),
        "ml_keywords+llm_rmse": rmse(
            NUMERIC + LLM_NUMERIC
        )
    }

    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of new uncached postings to extract"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate ML and LLM predictions"
    )

    args = parser.parse_args()

    if args.evaluate:
        evaluate()

    else:
        df, _, _, _ = load_xy()

        run_extraction(
            df=df,
            limit=args.limit
        )
