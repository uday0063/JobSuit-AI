"""
=============================================================
  src/pipeline.py — End-to-End Training & Inference Pipeline
=============================================================
  Orchestrates:
    1. Load & validate CSV dataset
    2. Preprocess (text clean + structured feature extraction)
    3. TF-IDF vectorization
    4. Feature combination (hstack)
    5. Train/test split
    6. XGBoost training
    7. Evaluation & saving artifacts
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessor import preprocess, STRUCTURED_COLS
from src.features import TextFeatureExtractor, combine_features, build_feature_names
from src.model import FakeJobClassifier


# ---------------------------------------------------------------------------
# Metadata filename — saved alongside the model for version tracking
# ---------------------------------------------------------------------------
METADATA_FILENAME = "pipeline_metadata.json"


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_pipeline(
    data_path: str,
    model_dir: str = "models/",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full training pipeline from raw CSV to saved model artifacts.

    Args:
        data_path    : path to the labeled CSV file
        model_dir    : directory to save all model artifacts
        test_size    : fraction of data held out for evaluation
        random_state : reproducibility seed

    Returns:
        evaluation metrics dict
    """
    sep = "═" * 56

    # ── Step 1: Load Dataset ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  📂  STEP 1 — Loading Dataset")
    print(sep)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    if "fraudulent" not in df.columns:
        raise ValueError("Dataset must contain a 'fraudulent' column (0=Real, 1=Fake).")

    # Drop rows with missing target
    df = df.dropna(subset=["fraudulent"]).reset_index(drop=True)
    y = df["fraudulent"].astype(int).values
    fake_count = int(y.sum())
    print(f"  Target distribution → Real: {len(y)-fake_count}, Fake: {fake_count}")

    # ── Step 2: Preprocess ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  🧹  STEP 2 — Preprocessing")
    print(sep)
    combined_text, struct_feats = preprocess(df)
    print(f"  ✔  Combined text built    ({len(combined_text):,} samples)")
    print(f"  ✔  Structured features    ({struct_feats.shape[1]} columns)")

    # ── Step 3: TF-IDF Vectorization ─────────────────────────────────
    print(f"\n{sep}")
    print("  📐  STEP 3 — TF-IDF Vectorization")
    print(sep)
    extractor = TextFeatureExtractor()
    tfidf_matrix = extractor.fit_transform(combined_text)
    print(f"  ✔  TF-IDF matrix          {tfidf_matrix.shape}")

    # ── Step 4: Combine Features ──────────────────────────────────────
    print(f"\n{sep}")
    print("  🔗  STEP 4 — Combining Features (hstack)")
    print(sep)
    X = combine_features(tfidf_matrix, struct_feats)
    feature_names = build_feature_names(extractor, STRUCTURED_COLS)
    print(f"  ✔  Final feature matrix   {X.shape}")
    print(f"  ✔  Total features         {X.shape[1]:,}")

    # ── Step 5: Train/Test Split ──────────────────────────────────────
    print(f"\n{sep}")
    print(f"  ✂️   STEP 5 — Train/Test Split  ({int((1-test_size)*100)}/{int(test_size*100)})")
    print(sep)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class ratio in both splits
    )
    print(f"  Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

    # ── Step 6: Train Model ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  🤖  STEP 6 — Training XGBoost Classifier")
    print(sep)
    classifier = FakeJobClassifier()
    classifier.train(X_train, y_train)

    # ── Step 7: Evaluate ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  📊  STEP 7 — Evaluation on Test Set")
    print(sep)
    metrics = classifier.evaluate(X_test, y_test, feature_names=feature_names)

    # ── Step 8: Save Artifacts ────────────────────────────────────────
    print(f"\n{sep}")
    print("  💾  STEP 8 — Saving Artifacts")
    print(sep)
    extractor.save(model_dir)
    classifier.save(model_dir)

    # Save metadata
    metadata = {
        "data_path": data_path,
        "n_samples": len(df),
        "n_features": X.shape[1],
        "tfidf_vocab_size": len(extractor.get_feature_names()),
        "structured_features": STRUCTURED_COLS,
        "test_size": test_size,
        "metrics": metrics,
    }
    meta_path = os.path.join(model_dir, METADATA_FILENAME)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✔  Metadata saved         → {meta_path}")

    print(f"\n{sep}")
    print("  🎉  TRAINING COMPLETE!")
    print(sep)
    return metrics


# ---------------------------------------------------------------------------
# Inference Pipeline — predict on a single raw dict / DataFrame row
# ---------------------------------------------------------------------------
def load_inference_pipeline(model_dir: str = "models/") -> tuple:
    """
    Load the saved TF-IDF extractor and XGBoost model from disk.

    Returns:
        (extractor, classifier)
    """
    extractor = TextFeatureExtractor.load(model_dir)
    classifier = FakeJobClassifier.load(model_dir)
    return extractor, classifier


def predict_single(
    job: dict,
    extractor: TextFeatureExtractor,
    classifier: FakeJobClassifier,
) -> dict:
    """
    Predict whether a single job posting is Fake or Real.

    Args:
        job        : dict with job posting fields
        extractor  : fitted TextFeatureExtractor
        classifier : trained FakeJobClassifier

    Returns:
        dict with keys: label, label_str, probability_fake, probability_real
    """
    # Build a single-row DataFrame
    df_row = pd.DataFrame([job])

    # Preprocess
    combined_text, struct_feats = preprocess(df_row)

    # TF-IDF transform (NOT fit — use already-fitted vectorizer)
    tfidf_matrix = extractor.transform(combined_text)

    # Combine features
    X = combine_features(tfidf_matrix, struct_feats)

    # Predict
    label = int(classifier.predict(X)[0])
    proba = classifier.predict_proba(X)[0]  # [P(Real), P(Fake)]
    
    # 🛡️ HEURISTIC SHIELD: If red flag count is high, override the AI's over-cautious score.
    # Obvious micro-payment scams should never be 99.9% real.
    sus_count = int(struct_feats["suspicious_score"].iloc[0])
    if sus_count >= 3:
        label = 1
        # Boost fake probability to at least 70% if enough red flags are found
        if proba[1] < 0.7:
            proba[1] = 0.7 + (sus_count * 0.05)
            # Ensure it doesn't exceed 0.99
            proba[1] = min(proba[1], 0.99)
            proba[0] = 1.0 - proba[1]

    return {
        "label": label,
        "label_str": "🚨 FAKE" if label == 1 else "✅ REAL",
        "probability_fake": round(float(proba[1]), 4),
        "probability_real": round(float(proba[0]), 4),
    }
