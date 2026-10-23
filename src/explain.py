import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import REPORTS_DIR
from src.models import get_models
from src.train import load_xy


def shap_report(X, y):
    pipe = get_models()["lightgbm_tabular"].fit(X, y)
    Xt = pipe.named_steps["prep"].transform(X)
    names = [
        n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()
    ]
    Xt = pd.DataFrame(Xt, columns=names)
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    shap.summary_plot(sv, Xt, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150)
    plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    print("Top SHAP features (mean |impact| in $K):")
    print(imp.head(10).round(2))
    return imp


def top_words(X, y, k=20):
    pipe = get_models()["ridge_tabular+tfidf"].fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(pipe.named_steps["model"].coef_, index=names)
    words = coef[coef.index.str.startswith("txt__")].rename(lambda s: s[5:])
    out = pd.DataFrame(
        {
            "raises_salary": words.nlargest(k).index,
            "lowers_salary": words.nsmallest(k).index,
        }
    )
    out.to_csv(REPORTS_DIR / "top_words.csv", index=False)
    print("\nWords that push salary up:  ", ", ".join(out.raises_salary[:10]))
    print("Words that push salary down:", ", ".join(out.lowers_salary[:10]))


if __name__ == "__main__":
    _, X, y, _ = load_xy()
    shap_report(X, y)
    top_words(X, y)
