"""
=============================================================
  src/model.py — XGBoost Classifier Wrapper
=============================================================
  Handles:
    - XGBoost model definition (with class-imbalance handling)
    - Training, evaluation, and metrics reporting
    - Confusion matrix display
    - Feature importance extraction
    - Save / load of trained model
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILENAME = "xgb_model.pkl"

# XGBoost default hyperparameters
DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",      # fast histogram method
)


# ---------------------------------------------------------------------------
# Fake Job Classifier
# ---------------------------------------------------------------------------
class FakeJobClassifier:
    """
    XGBoost-based binary classifier for detecting fraudulent job postings.
    Automatically computes scale_pos_weight to handle class imbalance.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.model: XGBClassifier | None = None
        self.scale_pos_weight: float = 1.0

    # ------------------------------------------------------------------
    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        XGBoost's built-in imbalance handler:
          scale_pos_weight = count(negative_class) / count(positive_class)
        """
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        ratio = neg / max(pos, 1)
        print(f"  ℹ  Class distribution → Real: {neg}, Fake: {pos}")
        print(f"  ℹ  scale_pos_weight   = {ratio:.2f}")
        return float(ratio)

    # ------------------------------------------------------------------
    def train(self, X_train: csr_matrix, y_train: np.ndarray) -> None:
        """
        Fit the XGBoost model on training data.
        Computes scale_pos_weight automatically.
        """
        self.scale_pos_weight = self._compute_scale_pos_weight(y_train)
        self.params["scale_pos_weight"] = self.scale_pos_weight

        self.model = XGBClassifier(**self.params)

        print("\n  ⚙  Training XGBoost …")
        self.model.fit(
            X_train,
            y_train,
            verbose=False,
        )
        print("  ✔  Training complete.")

    # ------------------------------------------------------------------
    def predict(self, X: csr_matrix) -> np.ndarray:
        """Return binary predictions (0 = Real, 1 = Fake)."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Return probability estimates for both classes."""
        self._check_fitted()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: csr_matrix,
        y_test: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict:
        """
        Evaluate model on test set and print a full report.

        Returns:
            dict with accuracy, precision, recall, f1
        """
        self._check_fitted()
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # ── Print Summary ──────────────────────────────────────────────
        sep = "─" * 52
        print(f"\n{sep}")
        print("  📊  EVALUATION RESULTS")
        print(sep)
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"\n{sep}")
        print("  Confusion Matrix  (rows=Actual, cols=Predicted)")
        print(f"  {'':12s}  Pred:Real  Pred:Fake")
        print(f"  Actual:Real  {cm[0,0]:>9}  {cm[0,1]:>9}")
        print(f"  Actual:Fake  {cm[1,0]:>9}  {cm[1,1]:>9}")
        print(f"\n{sep}")
        print("  Full Classification Report:")
        print(
            classification_report(
                y_test, y_pred,
                target_names=["Real (0)", "Fake (1)"],
            )
        )

        # ── Top Feature Importances ────────────────────────────────────
        if feature_names:
            self._print_top_features(feature_names, top_n=20)

        return {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(),
        }

    # ------------------------------------------------------------------
    def get_feature_importances(
        self,
        feature_names: list[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of the top-N most important features
        ranked by XGBoost's gain-based importance.
        """
        self._check_fitted()
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def _print_top_features(self, feature_names: list[str], top_n: int = 20) -> None:
        df = self.get_feature_importances(feature_names, top_n)
        print(f"  🔍  Top {top_n} Most Important Features:")
        print(f"  {'Rank':>4}  {'Feature':<35}  Importance")
        print(f"  {'─'*4}  {'─'*35}  {'─'*10}")
        for i, row in df.iterrows():
            print(f"  {i+1:>4}  {row['feature']:<35}  {row['importance']:.6f}")
        print()

    # ------------------------------------------------------------------
    def save(self, model_dir: str) -> None:
        """Persist the trained classifier to disk."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, MODEL_FILENAME)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✔  Model saved → {path}")

    @staticmethod
    def load(model_dir: str) -> "FakeJobClassifier":
        """Load a trained classifier from disk."""
        path = os.path.join(model_dir, MODEL_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✔  Model loaded ← {path}")
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
