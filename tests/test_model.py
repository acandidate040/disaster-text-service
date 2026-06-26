"""
Unit tests for the Disaster Text Classifier model components.

Run with:
  python -m pytest tests/test_model.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import TextCleaner, find_best_threshold  # noqa: E402


class TestTextCleaner:
    def test_basic_cleaning(self):
        cleaner = TextCleaner()
        raw = "  Forest Fire Near LA!!!  #wildfire @user http://t.co/abc  "
        result = cleaner.clean_text(raw)
        assert "forest fire near la" in result
        assert "http" not in result
        assert "@user" not in result
        assert "#" not in result
        assert result == result.strip()

    def test_url_replacement(self):
        assert "URL" in TextCleaner.clean_text("Check out https://example.com/news")
        assert "URL" in TextCleaner.clean_text("Visit www.example.org for more")

    def test_mention_replacement(self):
        assert "USER" in TextCleaner.clean_text("@example is responding to the crisis")

    def test_non_string_input(self):
        assert TextCleaner.clean_text(None) == ""
        assert TextCleaner.clean_text(123) == ""

    def test_sklearn_compatibility(self):
        cleaner = TextCleaner()
        X = ["Hello world", "Test tweet"]
        y = [0, 1]
        cleaner.fit(X, y)
        out = cleaner.transform(X)
        assert len(out) == 2
        assert isinstance(out, list)


class TestFindBestThreshold:
    def test_basic_search(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.95, 0.6])
        threshold, f1 = find_best_threshold(y_true, probs)
        assert 0.25 <= threshold <= 0.75
        assert 0.0 <= f1 <= 1.0

    def test_all_zeros(self):
        y_true = np.zeros(10, dtype=int)
        probs = np.random.rand(10)
        threshold, f1 = find_best_threshold(y_true, probs)
        assert 0.25 <= threshold <= 0.75
        assert f1 == 0.0

    def test_all_ones(self):
        y_true = np.ones(10, dtype=int)
        # All probabilities must be above the max threshold (0.75)
        # so every prediction is classified as 1.
        probs = np.full(10, 0.95)
        threshold, f1 = find_best_threshold(y_true, probs)
        assert 0.25 <= threshold <= 0.75
        assert f1 == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
