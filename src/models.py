"""Step 3a: model definitions. Every preprocessing step lives INSIDE a Pipeline,
so scalers / encoders / TF-IDF are fit only on the training fold (no leakage)."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupShuffleSplit

from src.config import RANDOM_STATE
from src.features import NUMERIC, CATEGORICAL, TEXT


class GroupMedianBaseline(BaseEstimator, RegressorMixin):
    """'Median salary for this title in this state' -> falls back to title, then global.
    This is the baseline a recruiter would use; every ML model must beat it."""

    def fit(self, X, y):
        d = X[["title_simple", "state"]].copy()
        d["y"] = np.asarray(y)
        self.global_ = float(d["y"].median())
        self.title_ = d.groupby("title_simple")["y"].median().to_dict()
        g = d.groupby(["title_simple", "state"])["y"]
        med, cnt = g.median(), g.size()
        self.pair_ = med[cnt >= 3].to_dict()   # only trust groups with >= 3 jobs
        return self

    def predict(self, X):
        out = []
        for t, s in zip(X["title_simple"], X["state"]):
            out.append(self.pair_.get((t, s), self.title_.get(t, self.global_)))
        return np.array(out)


def _num():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _cat():
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)


def _tfidf(max_features=3000):
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3,
                           stop_words="english", sublinear_tf=True)


def tabular_preprocessor(numeric=NUMERIC, categorical=CATEGORICAL):
    return ColumnTransformer([("num", _num(), list(numeric)), ("cat", _cat(), list(categorical))])


def text_preprocessor(svd_components=None, numeric=NUMERIC, categorical=CATEGORICAL):
    text = _tfidf() if svd_components is None else Pipeline(
        [("tfidf", _tfidf()), ("svd", TruncatedSVD(svd_components, random_state=RANDOM_STATE))])
    return ColumnTransformer([("num", _num(), list(numeric)), ("cat", _cat(), list(categorical)),
                              ("txt", text, TEXT)])


def get_models():
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "baseline_title_state_median": GroupMedianBaseline(),
        "ridge_tabular": Pipeline([("prep", tabular_preprocessor()), ("model", Ridge(alpha=10))]),
        "ridge_tabular+tfidf": Pipeline([("prep", text_preprocessor()), ("model", Ridge(alpha=3))]),
        "random_forest_tabular": Pipeline([("prep", tabular_preprocessor()),
                                           ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                                           n_jobs=-1, random_state=RANDOM_STATE))]),
        "lightgbm_tabular": Pipeline([("prep", tabular_preprocessor()), ("model", lgbm())]),
        "lightgbm_tabular+text_svd": Pipeline([("prep", text_preprocessor(svd_components=40)),
                                               ("model", lgbm())]),
    }


def lgbm(**kw):
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=RANDOM_STATE, verbose=-1)
    params.update(kw)
    return LGBMRegressor(**params)


def quantile_pipe(q):
    return Pipeline([("prep", text_preprocessor(svd_components=40)),
                     ("model", lgbm(objective="quantile", alpha=q, n_estimators=300))])


class ConformalInterval:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, X, y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        fit_idx, cal_idx = next(gss.split(X, y, groups))
        self.lo_ = quantile_pipe(self.alpha / 2).fit(X.iloc[fit_idx], y[fit_idx])
        self.hi_ = quantile_pipe(1 - self.alpha / 2).fit(X.iloc[fit_idx], y[fit_idx])
        lo, hi = self.lo_.predict(X.iloc[cal_idx]), self.hi_.predict(X.iloc[cal_idx])
        scores = np.maximum(lo - y[cal_idx], y[cal_idx] - hi)
        n = len(cal_idx)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, X):
        lo, hi = self.lo_.predict(X) - self.q_, self.hi_.predict(X) + self.q_
        return np.minimum(lo, hi), np.maximum(lo, hi)


