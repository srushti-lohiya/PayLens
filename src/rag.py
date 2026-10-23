"""Step 7: RAG — retrieve similar real postings and explain the estimate.

Index  : one embedding per posting (sentence-transformers if installed,
         otherwise TF-IDF + SVD so it still works offline).
Search : cosine similarity, top-k.
Explain: an LLM writes a short explanation that may ONLY use the retrieved jobs
         and the model's numbers (grounded). Without an API key a template is used.

Run:
  python -m src.rag build       # build models/rag_index.joblib
  python -m src.rag eval        # retrieval quality (leave-company-out)
  python -m src.rag search "senior data scientist python spark healthcare"
"""
import sys
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, CLEAN_CSV
from src.data import clean, load_raw
from src.features import simplify_title
from src.llm import ask, llm_available
from src.embedder import Embedder

INDEX_PATH = MODELS_DIR / "rag_index.joblib"


def _doc(df):
    return (df["Job Title"] + ". " + df["Job Description"].str[:2000]).tolist()


def build():
    df = pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else clean(load_raw())
    emb = Embedder().fit(_doc(df))
    meta = df[["Job Title", "company", "Location", "salary", "Sector"]].copy()
    meta["snippet"] = df["Job Description"].str.replace(r"\s+", " ", regex=True).str[:300]
    index = {"embedder": emb, "vectors": emb.encode(_doc(df)), "meta": meta.reset_index(drop=True)}
    joblib.dump(index, INDEX_PATH)
    print(f"indexed {len(meta)} postings with {emb.kind} -> {INDEX_PATH}")
    return index


_INDEX = None


def load_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = joblib.load(INDEX_PATH) if INDEX_PATH.exists() else build()
    return _INDEX


def search(text, k=5, exclude_company=None):
    idx = load_index()
    q = idx["embedder"].encode([text])[0]
    sims = idx["vectors"] @ q
    meta = idx["meta"].assign(similarity=sims.astype(float))
    if exclude_company is not None:
        meta = meta[meta["company"] != exclude_company]
    return meta.nlargest(k, "similarity")


def evaluate(k=5):
    """Leave-company-out: for each posting, retrieve k neighbours from OTHER companies.
       title@k  = share of neighbours with the same job family
       knn_rmse = RMSE if we predicted salary as the neighbours' mean salary"""
    idx = load_index()
    V, meta = idx["vectors"], idx["meta"]
    fam = meta["Job Title"].apply(simplify_title).values
    comp, sal = meta["company"].values, meta["salary"].values
    S = V @ V.T
    title_hits, errs = [], []
    for i in range(len(meta)):
        s = S[i].copy(); s[comp == comp[i]] = -np.inf
        nn = np.argpartition(-s, k)[:k]
        title_hits.append(np.mean(fam[nn] == fam[i]))
        errs.append(sal[nn].mean() - sal[i])
    res = {"embedder": idx["embedder"].kind, "k": k, "title_match@k": float(np.mean(title_hits)),
           "knn_salary_rmse": float(np.sqrt(np.mean(np.square(errs))))}
    print(res)
    return res


def explain(job_text, point, low, high, neighbours):
    rows = "\n".join(f"- {r['Job Title']} at {r['company']} ({r['Location']}): ${r['salary']:.0f}K"
                     for _, r in neighbours.iterrows())
    if not llm_available():
        return (f"Model estimate: ${point:.0f}K (80% range ${low:.0f}K–${high:.0f}K). "
                f"Most similar real postings:\n{rows}")
    prompt = f"""A salary model estimated ${point:.0f}K per year (80% interval ${low:.0f}K–${high:.0f}K)
for the job below. Similar real postings and their Glassdoor salaries:
{rows}

JOB:
{job_text[:3000]}

In 4-6 sentences explain the estimate to a job seeker. Rules: use ONLY the numbers above,
cite the similar postings by company name, mention what in the job description likely pushes
pay up or down, and note the data is 2019 US Glassdoor estimates."""
    return ask(prompt, system="You are a careful compensation analyst. Never invent numbers.", max_tokens=400)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build": build()
    elif cmd == "eval": evaluate()
    elif cmd == "search": print(search(" ".join(sys.argv[2:]))[["Job Title", "company", "salary", "similarity"]])
