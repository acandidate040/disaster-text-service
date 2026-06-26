"""
Integration tests for the FastAPI endpoints.

Requires the model artifact to be present at app/model.joblib.
Run with:
  python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

# Model classes must be importable before importing main, because main loads
# the joblib artifact at module level.
from app.model import (  # noqa: E402,F401
    TextCleaner,
    build_pipeline,
    find_best_threshold,
)

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_get_only(self):
        # POST to /health should not be allowed by FastAPI routing.
        response = client.post("/health")
        assert response.status_code == 405


class TestPredictEndpoint:
    def test_predict_disaster_text(self):
        response = client.post(
            "/predict",
            json={"text": "Forest fire near La Ronge Sask. Canada"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert "model_version" not in data
        assert data["label"] == 1
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_non_disaster_text(self):
        response = client.post(
            "/predict",
            json={"text": "I love eating pizza on weekends"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == 0
        assert 0.0 <= data["score"] <= 1.0

    def test_predict_empty_text(self):
        # Pydantic validation rejects empty strings.
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_field(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_too_long(self):
        response = client.post("/predict", json={"text": "x" * 2001})
        assert response.status_code == 422

    def test_predict_score_format(self):
        response = client.post(
            "/predict",
            json={"text": "Earthquake hit the city"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["label"], int)
        assert isinstance(data["score"], float)
        assert data["score"] == round(data["score"], 6)


class TestBatchPredictEndpoint:
    def test_batch_predict_basic(self):
        response = client.post(
            "/predict_batch",
            json={
                "texts": [
                    "Forest fire near La Ronge Sask. Canada",
                    "I love eating pizza on weekends",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "model_version" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["label"] == 1
        assert data["predictions"][1]["label"] == 0

    def test_batch_predict_empty_list(self):
        response = client.post("/predict_batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_predict_all_empty_strings(self):
        response = client.post("/predict_batch", json={"texts": ["", " ", "   "]})
        assert response.status_code == 422

    def test_batch_predict_too_many(self):
        response = client.post("/predict_batch", json={"texts": ["x"] * 101})
        assert response.status_code == 422


class TestStatsEndpoint:
    def test_stats_returns_data(self):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "avg_latency_ms" in data
        assert "model_version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert data["model_loaded"] is True
        assert isinstance(data["uptime_seconds"], float)


class TestSampleEndpoint:
    def test_sample_returns_text_and_label(self):
        response = client.get("/sample")
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "label" in data
        assert isinstance(data["text"], str)
        assert isinstance(data["label"], int)
        assert data["label"] in (0, 1)
        assert len(data["text"]) > 0

    def test_sample_varies_between_calls(self):
        texts = set()
        for _ in range(5):
            response = client.get("/sample")
            assert response.status_code == 200
            texts.add(response.json()["text"])
        assert len(texts) > 1


class TestStaticEndpoint:
    def test_index_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Disaster Text Classifier" in response.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
