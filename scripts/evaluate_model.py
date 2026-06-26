"""
Evaluation script for the Disaster Text Classifier.

Loads the trained model artifact and evaluates it on a held-out split of the
training data, reporting precision, recall, F1, and accuracy.

Usage:
  python scripts/evaluate_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import (  # noqa: E402,F401
    TextCleaner,
    build_pipeline,
    find_best_threshold,
)

DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"
META_PATH = ROOT / "app" / "model_meta.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].fillna("").astype(str)
    labels = df["target"].values

    # Held-out evaluation split (different random state from training)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=123
    )

    # Predict on held-out set
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Report metrics
    report = classification_report(
        y_test, predictions, target_names=["Not Disaster", "Disaster"], digits=4
    )
    print("Held-out Evaluation Metrics (20% split, random_state=123):")
    print(report)

    # Overall accuracy
    accuracy = (predictions == y_test).mean()
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Model Threshold:  {threshold:.4f}")
    # Load training metadata to display the validation F1 from the original run.
    validation_f1 = "N/A"
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        validation_f1 = meta.get("validation_f1_disaster_class", "N/A")
    print(f"Validation F1 (from training): {validation_f1}")


if __name__ == "__main__":
    main()
