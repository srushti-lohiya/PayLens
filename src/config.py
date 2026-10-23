"""Step 0: one place for paths and settings."""

from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "glassdoor_jobs.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "clean.csv"
LLM_CACHE = PROCESSED_DIR / "llm_features.jsonl"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "salary"
GROUP_COL = "company"
RANDOM_STATE = 42
N_SPLITS = 5

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

for d in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
