"""Step 8: REST API.  Run:  uvicorn src.api:app --reload   ->  http://127.0.0.1:8000/docs"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.service import predict_job, _models
from src import rag


class JobInput(BaseModel):
    job_title: str = Field(..., examples=["Senior Data Scientist"])
    job_description: str = Field(
        "", examples=["Python, SQL, Spark, AWS. 5+ years. PhD preferred."]
    )
    location: str = Field("", examples=["San Francisco, CA"])
    headquarters: Optional[str] = None
    company_rating: float = Field(-1, ge=-1, le=5)
    company_founded: int = Field(-1, ge=-1, le=2030)
    company_size: str = "Unknown"
    revenue: str = "Unknown / Non-Applicable"
    sector: str = "-1"
    ownership: str = "-1"
    industry: str = "-1"
    competitors: str = "-1"


@asynccontextmanager
async def lifespan(app):
    _models()
    rag.load_index()  # load models once at startup
    yield


app = FastAPI(title="Data Science Salary AI", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": _models()[2]["model_name"]}


@app.post("/predict")
def predict(job: JobInput, explain: bool = False, k: int = 5):
    data = job.model_dump()
    if data["headquarters"] is None:
        data.pop("headquarters")
    return predict_job(data, k=k, with_explanation=explain)


@app.get("/similar")
def similar(q: str, k: int = 5):
    return (
        rag.search(q, k=k)[["Job Title", "company", "salary", "similarity"]]
        .round(3)
        .to_dict(orient="records")
    )
