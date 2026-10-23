"""Step 4: predict a salary RANGE with a coverage guarantee (Conformalized Quantile Regression).

1. Train LightGBM quantile models for the 10th and 90th percentile.
2. On a held-out calibration set (by company), measure how often reality falls outside.
3. Widen the band by exactly enough so ~80% of real salaries fall inside.
This replaces the notebook's hard-coded "±$9K".
Run:  python -m src.intervals
"""
import json
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from src.config import MODELS_DIR, REPORTS_DIR, N_SPLITS
from src.models import ConformalInterval
from src.train import load_xy

ALPHA = 0.2  # 80% interval (ConformalInterval lives in src/models.py so it can be pickled)


def main():
    _, X, y, groups = load_xy()
    cover, width = [], []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        ci = ConformalInterval().fit(X.iloc[tr], y[tr], groups[tr])
        lo, hi = ci.predict(X.iloc[te])
        cover.append(np.mean((y[te] >= lo) & (y[te] <= hi)))
        width.append(np.mean(hi - lo))
    report = {"target_coverage": 1 - ALPHA, "cv_coverage": float(np.mean(cover)),
              "mean_width_k": float(np.mean(width))}
    print(json.dumps(report, indent=2))
    (REPORTS_DIR / "interval_report.json").write_text(json.dumps(report, indent=2))

    final = ConformalInterval().fit(X, y, groups)
    joblib.dump(final, MODELS_DIR / "interval_model.joblib")
    print(f"saved -> {MODELS_DIR / 'interval_model.joblib'}")


if __name__ == "__main__":
    main()
