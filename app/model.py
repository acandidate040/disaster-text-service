"""
Model definition for the Disaster Text Classifier.

Contains:
  - TextCleaner: scikit-learn compatible text-preprocessing transformer
  - build_pipeline(): constructs the TF-IDF + LogisticRegression pipeline
  - find_best_threshold(): selects the decision threshold that maximises F1
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


class TextCleaner(BaseEstimator, TransformerMixin):
    """Lightweight text cleaner compatible with scikit-learn Pipelines."""

    def fit(self, X, y=None) -> "TextCleaner":
        # Stateless transformer; nothing to learn from the data.
        return self

    def transform(self, X) -> list[str]:
        return [self.clean_text(x) for x in X]

    @staticmethod
    def clean_text(text) -> str:
        """Normalise and sanitise raw tweet-like text."""
        if not isinstance(text, str):
            text = ""
        text = text.lower()
        # Replace URLs and user mentions with neutral tokens
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)
        text = re.sub(r"@\w+", " USER ", text)
        text = text.replace("#", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text


# Hyperparameters chosen to balance vocabulary coverage and overfitting risk
# on a ~7,600-tweet training corpus.
MAX_WORD_FEATURES = 20000
MAX_CHAR_FEATURES = 30000
LOGREG_C = 1.5  # moderate regularisation; tuned informally on validation F1
LOGREG_MAX_ITER = 1000

# Threshold search bounds for F1 optimisation.
THRESHOLD_MIN = 0.25
THRESHOLD_MAX = 0.76
THRESHOLD_STEP = 0.01


def build_pipeline() -> Pipeline:
    """Build the full classification pipeline."""

    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=MAX_WORD_FEATURES,
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=MAX_CHAR_FEATURES,
    )

    features = FeatureUnion(
        [
            ("word_tfidf", word_features),
            ("char_tfidf", char_features),
        ]
    )

    classifier = LogisticRegression(
        C=LOGREG_C,
        max_iter=LOGREG_MAX_ITER,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        [
            ("cleaner", TextCleaner()),
            ("features", features),
            ("classifier", classifier),
        ]
    )


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Search for the probability threshold that maximises F1 on the disaster class."""
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)
    scores = []

    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, y_pred, pos_label=1)
        scores.append((threshold, score))

    best_threshold, best_f1 = max(scores, key=lambda item: item[1])
    return float(best_threshold), float(best_f1)
