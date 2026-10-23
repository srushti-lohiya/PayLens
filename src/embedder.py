"""Text embedder for RAG: sentence-transformers if available, else TF-IDF+SVD (offline)."""
import numpy as np
from src.config import RANDOM_STATE

EMB_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(EMB_MODEL)
            self.kind = EMB_MODEL
        except Exception:                      # not installed / offline -> TF-IDF fallback
            self.st, self.kind = None, "tfidf+svd"

    def fit(self, texts):
        if self.st is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.pipeline import make_pipeline
            self.tf = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                                    stop_words="english", sublinear_tf=True, min_df=2),
                                    TruncatedSVD(128, random_state=RANDOM_STATE)).fit(texts)
        return self

    def encode(self, texts):
        v = self.st.encode(list(texts), batch_size=32, show_progress_bar=False) if self.st else self.tf.transform(texts)
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def __getstate__(self):                    # don't pickle the big model, reload it instead
        d = self.__dict__.copy(); d["st"] = None; return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.kind != "tfidf+svd":
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(self.kind)
