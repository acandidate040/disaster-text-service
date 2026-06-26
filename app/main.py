"""
FastAPI application for the Disaster Text Classifier.

Serves:
  - GET  /health     → health check
  - POST /predict    → classify text (label + score)
  - GET  /           → static HTML UI
  - GET  /static/*   → static assets

The trained scikit-learn model is loaded once at startup from
app/model.joblib. Model classes (TextCleaner, etc.) are imported
at module level so joblib can resolve them during unpickling.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictStr

# Model classes must be imported here so they exist in the module
# namespace when joblib unpickles the saved pipeline.
# These imports are required at module level so joblib can resolve the
# custom transformer class during model unpickling. They appear unused
# to static analysis but are essential for runtime correctness.
from .model import TextCleaner, build_pipeline, find_best_threshold  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
META_PATH = BASE_DIR / "model_meta.json"
STATIC_DIR = BASE_DIR / "static"
DATA_PATH = BASE_DIR.parent / "data" / "train.csv"

# Configure minimal structured logging for the container
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disaster_classifier")


class PredictRequest(BaseModel):
    """Validated input for the /predict endpoint."""

    text: StrictStr = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    """Structured output from the /predict endpoint."""

    label: int
    score: float


app = FastAPI(
    title="Disaster Text Classifier",
    description="Binary classifier for short disaster-related text.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


def load_model_artifact() -> dict[str, Any]:
    """Load the saved model artifact and validate its contents."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise RuntimeError("Model artifact must be a dictionary")

    if "model" not in artifact or "threshold" not in artifact:
        raise RuntimeError("Model artifact must contain 'model' and 'threshold'")

    return artifact


artifact = load_model_artifact()
model = artifact["model"]
threshold = float(artifact["threshold"])

# Load training metadata for response enrichment.
model_version = "unknown"
if META_PATH.exists():
    try:
        model_version = json.loads(META_PATH.read_text()).get("model_type", "unknown")
    except Exception:
        logger.warning("Could not parse model_meta.json")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a health-check response that also validates model readiness."""
    if model is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service unavailable.",
        )
    return {"status": "ok"}


class BatchPredictRequest(BaseModel):
    """Validated input for the /predict_batch endpoint."""

    texts: list[StrictStr] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    """Structured output from the /predict_batch endpoint."""

    predictions: list[dict[str, int | float]]
    model_version: str


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> dict[str, Any]:
    """Classify multiple pieces of text in a single request."""
    texts = [t.strip() for t in payload.texts if t.strip()]

    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty texts provided.")

    try:
        probabilities = model.predict_proba(texts)[:, 1]
        predictions = []
        for score in probabilities:
            score = max(0.0, min(1.0, float(score)))
            label = int(score >= threshold)
            predictions.append(
                {
                    "label": label,
                    "score": round(score, 6),
                }
            )

        return {
            "predictions": predictions,
            "model_version": model_version,
        }

    except Exception as exc:
        logger.exception("Batch prediction failed for %d texts", len(texts))
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed due to an internal model error.",
        ) from exc


# Lightweight in-memory statistics for the /stats endpoint.
# Not persisted across container restarts (acceptable for a demo service).
_prediction_count = 0
_cumulative_latency_ms = 0.0


class StatsResponse(BaseModel):
    """Service-level statistics for observability."""

    total_predictions: int
    avg_latency_ms: float
    model_version: str
    model_loaded: bool
    uptime_seconds: float


@app.get("/stats")
def stats() -> dict[str, int | float | str | bool]:
    """Return lightweight service statistics for monitoring."""
    avg_latency = _cumulative_latency_ms / _prediction_count if _prediction_count > 0 else 0.0
    return {
        "total_predictions": _prediction_count,
        "avg_latency_ms": round(avg_latency, 2),
        "model_version": model_version,
        "model_loaded": model is not None and threshold is not None,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


# Record container start time for uptime calculation.
_start_time = time.time()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict[str, int | float]:
    """Classify a single piece of text and return the label + confidence."""
    text = payload.text.strip()

    start = time.perf_counter()
    try:
        score = float(model.predict_proba([text])[0, 1])
        score = max(0.0, min(1.0, score))
        label = int(score >= threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000
        global _prediction_count, _cumulative_latency_ms
        _prediction_count += 1
        _cumulative_latency_ms += elapsed_ms

        return {
            "label": label,
            "score": round(score, 6),
        }

    except Exception as exc:
        logger.exception("Prediction failed for input length %d", len(text))
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal model error.",
        ) from exc


class ExplainRequest(BaseModel):
    """Validated input for the /explain endpoint."""

    text: StrictStr = Field(..., min_length=1, max_length=2000)
    top_n: int = Field(5, ge=1, le=20)


class ExplainResponse(BaseModel):
    """Structured explanation output from the /explain endpoint."""

    text: str
    label: int
    score: float
    top_positive: list[dict[str, str | float]]
    top_negative: list[dict[str, str | float]]


# Pre-compute feature names and coefficients for explainability.
# These are extracted once at startup to avoid repeated pipeline introspection.
_explain_feature_names: list[str] | None = None
_explain_coefficients: Any = None


def _build_explain_index() -> tuple[list[str], Any]:
    """Extract feature names and coefficients from the trained pipeline."""
    feature_union = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    names = []
    for name, transformer in feature_union.transformer_list:
        vocab = transformer.vocabulary_
        # Sort vocabulary by index to align with feature matrix columns
        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
        for term, _ in sorted_vocab:
            names.append(f"{name}::{term}")

    coefficients = classifier.coef_[0]
    return names, coefficients


def _explain_text(text: str, top_n: int) -> dict[str, Any]:
    """Return the top positive and negative TF-IDF contributors for a given text."""
    global _explain_feature_names, _explain_coefficients

    if _explain_feature_names is None:
        _explain_feature_names, _explain_coefficients = _build_explain_index()

    # Compute probability for label
    score = float(model.predict_proba([text])[0, 1])
    score = max(0.0, min(1.0, score))
    label = int(score >= threshold)

    # Extract TF-IDF vector for the input
    feature_union = model.named_steps["features"]
    tfidf_vector = feature_union.transform([text]).toarray()[0]

    # Multiply TF-IDF values by coefficients to get per-feature contributions
    contributions = tfidf_vector * _explain_coefficients

    # Pair with feature names and filter out zero contributions
    named_contributions = [
        (name, float(contrib))
        for name, contrib in zip(_explain_feature_names, contributions)
        if abs(contrib) > 1e-9
    ]

    # Sort by contribution magnitude
    named_contributions.sort(key=lambda item: item[1], reverse=True)

    positive = [
        {"feature": name.split("::")[-1], "contribution": round(c, 4)}
        for name, c in named_contributions[:top_n]
        if c > 0
    ]
    negative = [
        {"feature": name.split("::")[-1], "contribution": round(abs(c), 4)}
        for name, c in named_contributions[-top_n:]
        if c < 0
    ][::-1]

    return {
        "text": text,
        "label": label,
        "score": round(score, 6),
        "top_positive": positive,
        "top_negative": negative,
    }


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest) -> dict[str, Any]:
    """Explain which features most strongly influenced the prediction."""
    text = payload.text.strip()
    try:
        return _explain_text(text, payload.top_n)
    except Exception as exc:
        logger.exception("Explanation failed for input length %d", len(text))
        raise HTTPException(
            status_code=500,
            detail="Explanation failed due to an internal model error.",
        ) from exc


class SampleResponse(BaseModel):
    """Structured output from the /sample endpoint."""

    text: str
    label: int


@app.get("/sample")
def sample() -> dict[str, str | int]:
    """Return a random labelled tweet from the training dataset."""
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Training data not available in the deployed container.",
        )
    try:
        df = pd.read_csv(DATA_PATH)
        df = df[["text", "target"]].dropna()
        row = df.sample(n=1, random_state=random.randint(0, 9999)).iloc[0]
        return {
            "text": str(row["text"]),
            "label": int(row["target"]),
        }
    except Exception as exc:
        logger.exception("Failed to load random sample")
        raise HTTPException(
            status_code=500,
            detail="Failed to load a random sample from the dataset.",
        ) from exc


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    """Serve the static HTML frontend."""
    return FileResponse(STATIC_DIR / "index.html")
