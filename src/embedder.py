import os
import numpy as np

from src.config import RANDOM_STATE

EMB_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self):
        backend = os.getenv("EMBEDDER_BACKEND", "auto").lower()

        if backend not in ("auto", "tfidf", "sentence-transformers"):
            raise ValueError(f"Unsupported EMBEDDER_BACKEND: {backend}")

        self.st = None
        self.tf = None
        self.kind = "tfidf+svd"

        if backend == "tfidf":
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.st = SentenceTransformer(EMB_MODEL)
            self.kind = EMB_MODEL

        except Exception:
            if backend == "sentence-transformers":
                raise

            print("Sentence-transformers unavailable. Using TF-IDF + SVD.")

    def fit(self, texts):
        if self.kind == "tfidf+svd":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.pipeline import make_pipeline

            self.tf = make_pipeline(
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    min_df=2,
                ),
                TruncatedSVD(n_components=128, random_state=RANDOM_STATE),
            ).fit(texts)

        return self

    def encode(self, texts):
        if self.kind == "tfidf+svd":
            if self.tf is None:
                raise RuntimeError("TF-IDF embedder must be fitted before encoding.")
            vectors = self.tf.transform(texts)
        else:
            vectors = self.st.encode(
                list(texts), batch_size=32, show_progress_bar=False
            )

        vectors = np.asarray(vectors, dtype=np.float32)

        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["st"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.kind != "tfidf+svd":
            from sentence_transformers import SentenceTransformer

            self.st = SentenceTransformer(self.kind)
