"""
Generate a histogram of model confidence scores from a sample of the training data.

Produces a PNG histogram showing the distribution of /predict scores across
~500 held-out tweets, coloured by true label. This provides visual evidence
that the model separates disaster and non-disaster texts effectively.

Usage:
  python scripts/score_histogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Required for joblib unpickling.
from app.model import (  # noqa: E402,F401
    TextCleaner,
    build_pipeline,
    find_best_threshold,
)

DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "app" / "model.joblib"
# Save into the static directory so the UI can serve it directly.
OUTPUT_PATH = ROOT / "app" / "static" / "score_histogram.png"
SAMPLE_SIZE = 500

# Brand colour palette (matches the UI stylesheet)
BRAND_NAVY = "#003399"
BRAND_BLUE = "#0092d1"
BRAND_GOLD = "#ffd256"
BRAND_DARK = "#1a1a2e"
BRAND_GRAY = "#5a5a6e"
BRAND_LIGHT_GRAY = "#f5f7fa"
BRAND_BORDER = "#d0d7de"
DISASTER_RED = "#c62828"
SAFE_GREEN = "#2e7d32"


def main() -> None:
    # Configure matplotlib for headless environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = float(artifact["threshold"])

    df = pd.read_csv(DATA_PATH)
    df = df[["text", "target"]].dropna()

    # Stratified sample: ensure both classes are represented.
    disaster = df[df["target"] == 1]
    non_disaster = df[df["target"] == 0]

    sample_disaster = disaster.sample(n=min(len(disaster), SAMPLE_SIZE // 2), random_state=42)
    sample_non = non_disaster.sample(n=min(len(non_disaster), SAMPLE_SIZE // 2), random_state=42)
    sample = pd.concat([sample_disaster, sample_non]).sample(frac=1, random_state=42)

    texts = sample["text"].astype(str).tolist()
    labels = sample["target"].values

    probabilities = model.predict_proba(texts)[:, 1]

    disaster_scores = probabilities[labels == 1]
    non_disaster_scores = probabilities[labels == 0]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Open Sans", "Segoe UI", "Arial"]
    plt.rcParams["axes.edgecolor"] = BRAND_BORDER
    plt.rcParams["axes.labelcolor"] = BRAND_DARK
    plt.rcParams["xtick.color"] = BRAND_GRAY
    plt.rcParams["ytick.color"] = BRAND_GRAY
    plt.rcParams["figure.facecolor"] = BRAND_LIGHT_GRAY
    plt.rcParams["axes.facecolor"] = BRAND_LIGHT_GRAY

    plt.figure(figsize=(8, 5), dpi=150)

    # Compute counts per bin to set y-axis limit with headroom
    all_counts, bin_edges = np.histogram(
        np.concatenate([non_disaster_scores, disaster_scores]), bins=30
    )
    y_max = int(all_counts.max()) + 5

    plt.hist(
        non_disaster_scores,
        bins=bin_edges,
        alpha=0.65,
        label="Not Disaster",
        color=SAFE_GREEN,
        edgecolor="white",
    )
    plt.hist(
        disaster_scores,
        bins=bin_edges,
        alpha=0.65,
        label="Disaster",
        color=DISASTER_RED,
        edgecolor="white",
    )
    plt.axvline(
        threshold,
        color=BRAND_NAVY,
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.2f})",
    )
    plt.xlabel("Model Confidence (Probability of Disaster)", fontsize=11, fontweight="600")
    plt.ylabel("Tweet Count", fontsize=11, fontweight="600")
    plt.title(
        f"Score Distribution on {len(sample)} Held-Out Tweets",
        fontsize=13,
        fontweight="bold",
        color=BRAND_NAVY,
    )
    plt.legend(loc="upper right", frameon=True, fancybox=True, fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, y_max)
    plt.grid(axis="y", alpha=0.3, color=BRAND_BORDER)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"Histogram saved to: {OUTPUT_PATH}")
    d_mean = disaster_scores.mean()
    nd_mean = non_disaster_scores.mean()
    print(f"  Disaster tweets:     {len(disaster_scores)} (mean: {d_mean:.3f})")
    print(f"  Non-disaster tweets: {len(non_disaster_scores)} (mean: {nd_mean:.3f})")
    print(f"  Decision threshold:  {threshold:.3f}")


if __name__ == "__main__":
    main()
