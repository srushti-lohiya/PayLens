import json
from functools import lru_cache
import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.features import build_features
from src import rag


@lru_cache
def _models():
    point = joblib.load(MODELS_DIR / "salary_model.joblib")
    interval = joblib.load(MODELS_DIR / "interval_model.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return point, interval, meta


def to_raw(job: dict) -> pd.DataFrame:
    """Map friendly input names to the original Glassdoor columns."""
    return pd.DataFrame(
        [
            {
                "Job Title": job["job_title"],
                "Job Description": job.get("job_description", ""),
                "Rating": job.get("company_rating", -1),
                "Founded": job.get("company_founded", -1),
                "Size": job.get("company_size", "Unknown"),
                "Revenue": job.get("revenue", "Unknown / Non-Applicable"),
                "Competitors": job.get("competitors", "-1"),
                "Location": job.get("location", ""),
                "Headquarters": job.get("headquarters", job.get("location", "")),
                "Sector": job.get("sector", "-1"),
                "Type of ownership": job.get("ownership", "-1"),
                "Industry": job.get("industry", "-1"),
            }
        ]
    )


def predict_job(job: dict, k: int = 5, with_explanation: bool = False) -> dict:
    point_model, interval_model, meta = _models()
    X = build_features(to_raw(job))
    point = float(point_model.predict(X)[0])
    lo, hi = interval_model.predict(X)
    low, high = float(min(lo[0], point)), float(max(hi[0], point))
    text = f"{job['job_title']}. {job.get('job_description', '')}"
    nn = rag.search(text, k=k)
    out = {
        "salary_k": round(point, 1),
        "low_k": round(max(low, 0), 1),
        "high_k": round(high, 1),
        "interval": "80% conformal",
        "model": meta["model_name"],
        "similar_jobs": nn[["Job Title", "company", "Location", "salary", "similarity"]]
        .round(3)
        .to_dict(orient="records"),
    }
    if with_explanation:
        out["explanation"] = rag.explain(text, point, low, high, nn)
    return out
