"""
=============================================================
  src/features.py — TF-IDF Vectorization & Feature Combination
=============================================================
  Handles:
    - TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
    - Combining TF-IDF sparse matrix with structured features (hstack)
    - Fit/transform/save/load of the vectorizer
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"


# ---------------------------------------------------------------------------
# TF-IDF Wrapper
# ---------------------------------------------------------------------------
class TextFeatureExtractor:
    """
    Wraps sklearn TfidfVectorizer with save/load support.
    Uses bigrams and up to 5000 features to capture phrase-level patterns.
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b|[₹$!/-]",
            min_df=2,
        )
        self.is_fitted = False

    def fit_transform(self, texts: pd.Series) -> csr_matrix:
        """Fit vectorizer on training texts and return sparse matrix."""
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix

    def transform(self, texts: pd.Series) -> csr_matrix:
        """Transform new texts using the already-fitted vectorizer."""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return all TF-IDF feature names (vocabulary terms)."""
        return list(self.vectorizer.get_feature_names_out())

    def save(self, model_dir: str) -> None:
        """Persist the fitted vectorizer to disk."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, VECTORIZER_FILENAME)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✔  Vectorizer saved → {path}")

    @staticmethod
    def load(model_dir: str) -> "TextFeatureExtractor":
        """Load a previously saved vectorizer from disk."""
        path = os.path.join(model_dir, VECTORIZER_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vectorizer not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✔  Vectorizer loaded ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Feature Combination (TF-IDF + Structured)
# ---------------------------------------------------------------------------
def combine_features(
    tfidf_matrix: csr_matrix,
    struct_df: pd.DataFrame,
) -> csr_matrix:
    """
    Horizontally stack TF-IDF sparse matrix and structured numeric features
    into a single sparse feature matrix using scipy.sparse.hstack.

    Args:
        tfidf_matrix : scipy sparse matrix (n_samples × n_tfidf_features)
        struct_df    : DataFrame of numeric structured features (n_samples × k)

    Returns:
        Combined sparse matrix (n_samples × (n_tfidf_features + k))
    """
    # Convert structured DataFrame to sparse matrix
    struct_sparse = csr_matrix(struct_df.values.astype(np.float32))

    # Horizontally stack: [TF-IDF | structured]
    combined = hstack([tfidf_matrix, struct_sparse], format="csr")
    return combined


# ---------------------------------------------------------------------------
# Feature Name Builder (for interpretability)
# ---------------------------------------------------------------------------
def build_feature_names(
    tfidf_extractor: TextFeatureExtractor,
    struct_cols: list[str],
) -> list[str]:
    """
    Returns a flat list of all feature names in the same order as
    the combined feature matrix — useful for XGBoost feature importance.
    """
    return tfidf_extractor.get_feature_names() + struct_cols
