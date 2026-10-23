import argparse
import json
import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate

from src.config import CLEAN_CSV, MODELS_DIR, REPORTS_DIR, TARGET, GROUP_COL, N_SPLITS
from src.data import clean, load_raw
from src.features import build_features
from src.models import get_models

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "r2": "r2",
}


def load_xy():
    df = pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else clean(load_raw())
    X = build_features(df)
    return df, X, df[TARGET].values, df[GROUP_COL].values


def evaluate(models, X, y, groups):
    cv = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=1)
        row = {
            "model": name,
            "rmse": -s["test_rmse"].mean(),
            "rmse_std": s["test_rmse"].std(),
            "mae": -s["test_mae"].mean(),
            "mape_%": -100 * s["test_mape"].mean(),
            "r2": s["test_r2"].mean(),
        }
        rows.append(row)
        print(
            f"{name:32s} RMSE {row['rmse']:6.2f} ± {row['rmse_std']:4.2f} | "
            f"MAE {row['mae']:6.2f} | MAPE {row['mape_%']:5.1f}% | R² {row['r2']:5.2f}"
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def main(use_mlflow=False):
    df, X, y, groups = load_xy()
    print(f"rows={len(X)}  companies={len(set(groups))}  salary std={y.std():.2f}K\n")
    models = get_models()
    results = evaluate(models, X, y, groups)

    results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    best_name = results.loc[~results.model.str.startswith("baseline"), "model"].iloc[0]
    print(f"\nbest model: {best_name}")

    best = models[best_name].fit(X, y)  # refit on ALL clean data for serving
    joblib.dump(best, MODELS_DIR / "salary_model.joblib")
    meta = {
        "model_name": best_name,
        "n_rows": int(len(X)),
        "cv": results.round(3).to_dict(orient="records"),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {MODELS_DIR / 'salary_model.joblib'}")

    if use_mlflow:
        import mlflow

        mlflow.set_experiment("salary-prediction")
        for r in results.to_dict(orient="records"):
            with mlflow.start_run(run_name=r["model"]):
                mlflow.log_metrics({k: v for k, v in r.items() if k != "model"})
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlflow", action="store_true")
    main(p.parse_args().mlflow)
