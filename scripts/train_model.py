"""
Training script for the Disaster Text Classifier.

Expects:
  data/train.csv  — Kaggle disaster-tweets training data

Produces:
  app/model.joblib      — saved scikit-learn pipeline + threshold
  app/model_meta.json   — training metadata (F1, hyperparameters, etc.)
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Allow importing from the project root (app.model)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import build_pipeline, find_best_threshold  # noqa: E402

DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"
META_PATH = ROOT / "app" / "model_meta.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("train.csv must contain 'text' and 'target' columns")

    X = df["text"].fillna("").astype(str)
    y = df["target"].astype(int)

    # 80/20 stratified split keeps class balance consistent
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Tune decision threshold to maximise F1 on the disaster class
    probabilities = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, probabilities)

    # Apply the tuned threshold for the final validation report
    y_pred = (probabilities >= best_threshold).astype(int)

    print("Validation results")
    print("------------------")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best F1 disaster class: {best_f1:.4f}")
    print()
    print(classification_report(y_val, y_pred, digits=4))

    # Ensure output directory exists before writing
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the trained pipeline alongside the tuned threshold
    artifact = {
        "model": model,
        "threshold": best_threshold,
    }

    joblib.dump(artifact, MODEL_PATH)

    # Persist human-readable metadata for reproducibility
    metadata = {
        "model_type": "TF-IDF word+char n-grams with Logistic Regression",
        "validation_f1_disaster_class": best_f1,
        "threshold": best_threshold,
        "validation_split": "80/20 stratified, random_state=42",
        "features": {
            "word_ngrams": [1, 2],
            "char_wb_ngrams": [3, 5],
        },
    }

    META_PATH.write_text(json.dumps(metadata, indent=2))

    print()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()
