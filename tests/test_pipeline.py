"""Step 10: tests.  Run:  pytest -q"""
import numpy as np
import pandas as pd
import pytest
from src.data import parse_salary, clean, load_raw
from src.features import build_features, NUMERIC, CATEGORICAL, simplify_title, seniority


@pytest.mark.parametrize("text,expected", [
    ("$53K-$91K (Glassdoor est.)", 72.0),
    ("Employer Provided Salary:$120K-$160K", 140.0),
    ("$17-$24 Per Hour(Glassdoor est.)", round((17 + 24) / 2 * 2080 / 1000, 2)),
    ("-1", np.nan),
])
def test_parse_salary(text, expected):
    out = parse_salary(text)
    assert (np.isnan(out) and np.isnan(expected)) or out == pytest.approx(expected)


def test_clean_has_no_missing_target_or_duplicates():
    df = clean(load_raw(), verbose=False)
    assert df["salary"].notna().all()
    assert not df.duplicated(subset=["Job Title", "company", "Location", "Job Description"]).any()


def test_features_single_row_matches_training_columns():
    raw = pd.DataFrame([{"Job Title": "Sr. Data Scientist", "Job Description": "Python and SQL",
                         "Rating": -1, "Founded": -1, "Size": "Unknown", "Revenue": "Unknown / Non-Applicable",
                         "Competitors": "-1", "Location": "Austin, TX", "Headquarters": "Austin, TX",
                         "Sector": "-1", "Type of ownership": "-1", "Industry": "-1"}])
    X = build_features(raw)
    assert set(NUMERIC + CATEGORICAL) <= set(X.columns)
    assert X.loc[0, "skill_python"] == 1 and X.loc[0, "job_in_hq_state"] == 1
    assert np.isnan(X.loc[0, "company_rating"])


def test_title_rules():
    assert simplify_title("Senior Machine Learning Engineer") == "mle"
    assert seniority("Sr. Data Analyst") == 2 and seniority("Junior Analyst") == 0
