"""Multiclass incident classifier persisted as a fitted scikit-learn model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import joblib

from trace_core.config import SEQUENCE_FEATURE_NAMES

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


class IncidentClassifier:
    def __init__(self, classes: np.ndarray, coefficients: np.ndarray, intercept: np.ndarray,
                 estimator: LogisticRegression | None = None) -> None:
        self.classes = classes.astype(str)
        self.coefficients = coefficients.astype(np.float64)
        self.intercept = intercept.astype(np.float64)
        self.estimator = estimator

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, *, seed: int = 31,
            class_weight: dict[str, float] | None = None) -> tuple["IncidentClassifier", LogisticRegression]:
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression(max_iter=1500, class_weight=class_weight, random_state=seed, solver="lbfgs")
        estimator.fit(features, labels)
        return cls(estimator.classes_, estimator.coef_, estimator.intercept_, estimator), estimator

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.asarray(features, dtype=np.float64) @ self.coefficients.T + self.intercept
        if logits.shape[1] == 1 and len(self.classes) == 2:
            logits = np.column_stack((np.zeros(len(logits), dtype=np.float64), logits[:, 0]))
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(np.clip(logits, -60, 60))
        return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.estimator is None:
            raise ValueError("Only a fitted classifier can be saved.")
        joblib.dump({"model": self.estimator, "feature_names": SEQUENCE_FEATURE_NAMES}, target)

    @classmethod
    def load(cls, path: str | Path) -> "IncidentClassifier":
        artifact = joblib.load(path)
        if tuple(artifact["feature_names"]) != SEQUENCE_FEATURE_NAMES:
            raise ValueError("Classifier artifact feature schema is missing or incompatible. Retrain the models.")
        estimator = artifact["model"]
        if estimator.n_features_in_ != len(SEQUENCE_FEATURE_NAMES):
            raise ValueError("Classifier dimensions are incompatible. Retrain the models.")
        return cls(estimator.classes_, estimator.coef_, estimator.intercept_, estimator)
