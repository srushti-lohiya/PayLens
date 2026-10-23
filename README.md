# 💼 Data Science Salary AI

Predicts data-science salaries from a job posting, gives an **80% salary range with a
coverage guarantee**, retrieves **similar real postings** (RAG), and can **explain the
estimate with an LLM** grounded only in those postings. Served with FastAPI + Streamlit.

## Key finding: the original model was inflated by leakage
The original notebook reported RMSE ≈ 17.6K for Random Forest. That number was driven by:
1. **275 duplicate postings** shared between CV folds (the model memorised them),
2. **214 missing salaries filled with the median** (the model was graded on invented targets),
3. scaling and feature selection fit on the full dataset before CV.

After fixing all three (dedup, drop missing target, everything inside a `Pipeline`,
`GroupKFold` by company), the honest results on 467 postings / 343 companies are:

| Model | RMSE ($K) | MAE ($K) | MAPE | R² |
|---|---|---|---|---|
| Predict the mean (baseline) | 36.1 | 28.3 | 33.6% | -0.01 |
| Median by title + state (baseline) | 32.4 | 24.8 | 28.3% | 0.18 |
| Random Forest, tabular | 29.1 | 21.7 | 24.3% | 0.34 |
| LightGBM, tabular | 29.6 | 22.3 | 24.7% | 0.32 |
| LightGBM, tabular + description (SVD) | 27.9 | 21.2 | 23.1% | 0.39 |
| **Ridge, tabular + description (TF-IDF)** | **27.1** | **20.9** | 23.8% | **0.42** |

Using the job description text is the biggest single improvement.
Salary range (conformalized quantile LightGBM): target 80% coverage, measured **83.5%**, mean width ≈ $75K.

## Architecture
```
job posting ──► build_features() ──► Ridge+TF-IDF ──────────► point estimate
                    │                 Quantile LGBM + conformal ► 80% range
                    └► Embedder ──► vector search ──► top-5 similar postings
                                                        │
                                   LLM (optional) ◄─────┘  grounded explanation
```

## Run it
```bash
pip install -r requirements.txt
make all            # data → train → intervals → explain → rag → tests
make api            # http://127.0.0.1:8000/docs
make app            # Streamlit demo
export ANTHROPIC_API_KEY=...   # optional: LLM features + explanations
make llm && make llm-eval
```

## Limitations
2019 US Glassdoor *estimates* (not offers), only 467 unique postings, mostly US-coastal
companies. Treat outputs as rough guidance, not compensation advice.
