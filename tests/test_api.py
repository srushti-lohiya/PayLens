import pytest
from src.config import MODELS_DIR

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "salary_model.joblib").exists(),
                                reason="train models first (make train)")


def test_predict_endpoint():
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json={"job_title": "Senior Data Scientist",
                                     "job_description": "Python, Spark, AWS, PhD", "location": "Boston, MA"})
        body = r.json()
        assert r.status_code == 200
        assert body["low_k"] <= body["salary_k"] <= body["high_k"]
        assert len(body["similar_jobs"]) == 5
