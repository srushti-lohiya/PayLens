"""Step 1: load raw Glassdoor data, parse salary, remove leakage sources.

Fixes vs. the original notebook:
  * rows with no salary are DROPPED (never impute the target)
  * duplicate postings are DROPPED (they leak between CV folds)
"""
import re
import numpy as np
import pandas as pd
from src.config import RAW_CSV, CLEAN_CSV

HOURS_PER_YEAR = 40 * 52  # standard full-time year


def parse_salary(text: str) -> float:
    """'$53K-$91K (Glassdoor est.)' -> 72.0 (thousand USD / year). NaN if missing."""
    if not isinstance(text, str) or text.strip() == "-1":
        return np.nan
    s = text.lower().split("(")[0]
    hourly = "per hour" in s
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", s)]
    if len(nums) < 2:
        return np.nan
    low, high = nums[0], nums[1]
    if hourly:
        low, high = low * HOURS_PER_YEAR / 1000, high * HOURS_PER_YEAR / 1000
    return round((low + high) / 2, 2)


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    df["salary"] = df["Salary Estimate"].apply(parse_salary)
    df["company"] = df["Company Name"].astype(str).str.split("\n").str[0].str.strip()

    missing = int(df["salary"].isna().sum())
    df = df[df["salary"].notna()]

    key = ["Job Title", "company", "Location", "Job Description"]
    dupes = int(df.duplicated(subset=key).sum())
    df = df.drop_duplicates(subset=key).reset_index(drop=True)

    if verbose:
        print(f"raw rows:            {n0}")
        print(f"dropped (no salary): {missing}")
        print(f"dropped (duplicate): {dupes}")
        print(f"clean rows:          {len(df)}")
    return df


if __name__ == "__main__":
    df = clean(load_raw())
    df.to_csv(CLEAN_CSV, index=False)
    print(f"saved -> {CLEAN_CSV}")
    print(df["salary"].describe().round(1))
