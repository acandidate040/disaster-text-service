"""
Test-set evaluator for the Disaster Text Classifier.

Expects:
  data/test.csv  — Kaggle disaster-tweets test data (no labels)
  app/model.joblib  — trained model artifact

Produces:
  predictions.csv  — id,label columns ready for Kaggle submission

Usage:
  python scripts/predict_test.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# These imports are required at module level so joblib can resolve the
# custom transformer class during model unpickling.
from app.model import (  # noqa: E402,F401
    TextCleaner,
    build_pipeline,
    find_best_threshold,
)

DATA_PATH = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"
OUTPUT_PATH = ROOT / "predictions.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing test data: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if "model" not in artifact or "threshold" not in artifact:
        raise ValueError("Model artifact must contain 'model' and 'threshold' keys")

    model = artifact["model"]
    threshold = float(artifact["threshold"])

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns:
        raise ValueError("test.csv must contain a 'text' column")

    if "id" not in df.columns:
        raise ValueError("test.csv must contain an 'id' column")

    texts = df["text"].fillna("").astype(str)
    probabilities = model.predict_proba(texts)[:, 1]
    labels = (probabilities >= threshold).astype(int)

    with OUTPUT_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label"])
        for idx, label in zip(df["id"], labels):
            writer.writerow([int(idx), int(label)])

    print(f"Wrote {len(labels)} predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
