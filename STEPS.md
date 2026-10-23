# Step-by-step guide

Do these in order. Each step has: what to run, what you should see, and why it matters.
Commit to Git after every step (`git add . && git commit -m "step N: ..."`).

---
## Step 0 — Set up
```bash
unzip salary-ai.zip && cd salary-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git init && git add . && git commit -m "step 0: project skeleton"
```
Files: `src/config.py` holds every path and setting in one place.

---
## Step 1 — Clean data without leakage  (`src/data.py`)
```bash
python -m src.data
```
Expected:
```
raw rows:            956
dropped (no salary): 214
dropped (duplicate): 275
clean rows:          467
```
Why: the target is never imputed, and duplicate postings can't leak across CV folds.

---
## Step 2 — Features  (`src/features.py`)
Nothing to run on its own. `build_features()` creates title family, seniority, company age,
size/revenue ranks, state, HQ-in-same-state, 11 skill flags and the full description text.
The SAME function is used by training, the API and the app, so there is no train/serve skew.

---
## Step 3 — Honest evaluation + training  (`src/models.py`, `src/train.py`)
```bash
python -m src.train          # optional: pip install mlflow && python -m src.train --mlflow && mlflow ui
```
Expected (5-fold GroupKFold by company):
```
baseline_mean                    RMSE  36.06
baseline_title_state_median      RMSE  32.42
ridge_tabular                    RMSE  27.54
ridge_tabular+tfidf              RMSE  27.09   <- best
random_forest_tabular            RMSE  29.10
lightgbm_tabular                 RMSE  29.55
lightgbm_tabular+text_svd        RMSE  27.88
```
Saves `models/salary_model.joblib`, `models/metadata.json`, `reports/cv_results.csv`.
Why: all preprocessing is inside a `Pipeline`, no company is in both train and test,
and every model is compared to two baselines.

---
## Step 4 — Salary range instead of a fake ±$9K  (`src/intervals.py`)
```bash
python -m src.intervals
```
Expected: `target_coverage 0.8`, `cv_coverage ≈ 0.84`, `mean_width_k ≈ 75`.
Why: conformal calibration means the 80% range really contains ~80% of real salaries.
The range is wide because the data is small; saying that honestly is a strength.

---
## Step 5 — Explainability  (`src/explain.py`)
```bash
python -m src.explain
```
Creates `reports/shap_summary.png` and `reports/top_words.csv`.
Top drivers you should see: seniority, California, analyst title (negative), R/Python skills.
Put the SHAP plot in your README.

---
## Step 6 — LLM feature extraction  (`src/llm.py`, `src/llm_extract.py`)
```bash
export ANTHROPIC_API_KEY=sk-ant-...          # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
python -m src.llm_extract --limit 20         # test on 20 postings, check data/processed/llm_features.jsonl
python -m src.llm_extract                    # all 467 (cached, safe to re-run)
python -m src.llm_extract --evaluate
```
The LLM returns JSON validated by a Pydantic schema (`JobSignals`): years of experience,
degree, seniority, remote, clearance, people management, skills, and its own salary guess.
`--evaluate` prints three numbers. Put them in the README as a table:
- `llm_only_rmse`       — the LLM guessing salary directly
- `ml_keywords_rmse`    — your model with keyword features
- `ml_keywords+llm_rmse`— your model + LLM-extracted features
Whatever the result, report it honestly. "The LLM alone was worse than a trained model"
is a great interview talking point.
Extra credit: hand-label 50 postings and measure extraction accuracy per field.
Change the model with `export LLM_MODEL=...` (default in `src/config.py`).

---
## Step 7 — RAG: similar postings + grounded explanation  (`src/embedder.py`, `src/rag.py`)
```bash
pip install sentence-transformers       # recommended; without it a TF-IDF fallback is used
python -m src.rag build
python -m src.rag eval
python -m src.rag search "senior data scientist python spark healthcare"
```
`eval` is leave-company-out: `title_match@5` (≈0.63 with TF-IDF) and `knn_salary_rmse` (≈33).
Run it once with TF-IDF and once with sentence-transformers and report both — that's
a real retrieval comparison. With an API key, `rag.explain()` writes an explanation that may
only use the retrieved jobs and the model's numbers.
Upgrade path: swap the numpy index for Chroma / Qdrant / pgvector when data grows.

---
## Step 8 — API  (`src/service.py`, `src/api.py`)
```bash
uvicorn src.api:app --reload
```
Open http://127.0.0.1:8000/docs and try `POST /predict` (add `?explain=true` for the LLM text).
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"job_title":"Senior Data Scientist","job_description":"Python, Spark, AWS, PhD","location":"Boston, MA"}'
```

---
## Step 9 — Demo app  (`app.py`)
```bash
streamlit run app.py
```
Deploy free: push to GitHub → Hugging Face Spaces (Streamlit SDK) or Streamlit Community Cloud.
Add `ANTHROPIC_API_KEY` as a secret there. Put the live link at the top of the README.

---
## Step 10 — Tests  (`tests/`)
```bash
pytest -q          # expect: 8 passed
```

---
## Step 11 — Docker
```bash
docker build -t salary-ai .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY salary-ai
```

---
## Step 12 — CI  (`.github/workflows/ci.yml`)
Push to GitHub. Every push lints, retrains from scratch and runs the tests.

---
## Step 13 — Make the README sell it
1. Live demo link + screenshot at the top
2. "Key finding: the original model was inflated by leakage" with the before/after numbers
3. The results table, the interval coverage, the SHAP plot
4. The LLM comparison table from step 6 and the RAG comparison from step 7
5. Architecture diagram and limitations

## Next ideas after this
- Add a newer salary dataset (2023–2025) and compare drift
- Optuna tuning for LightGBM; CatBoost with text features
- Agent with tools: `predict_salary`, `find_similar_jobs`, `skill_gap(resume, jd)`
- Evidently report for data drift between old and new data
