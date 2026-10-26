"""Bernoulli Restricted Boltzmann Machine trained with CD-k / Gibbs sampling."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from trace_core.config import ARTIFACT_FILES, RBM_FEATURE_NAMES


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -25.0, 25.0)))


class RestrictedBoltzmannMachine:
    """A Bernoulli RBM whose negative phase uses k Gibbs transitions."""

    def __init__(self, n_visible: int, n_hidden: int = 12, seed: int = 23) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        scale = 0.01
        self.weights = self.rng.normal(0.0, scale, (n_visible, n_hidden)).astype(np.float32)
        self.visible_bias = np.zeros(n_visible, dtype=np.float32)
        self.hidden_bias = np.zeros(n_hidden, dtype=np.float32)

    def hidden_probabilities(self, visible: np.ndarray) -> np.ndarray:
        return _sigmoid(visible @ self.weights + self.hidden_bias)

    def visible_probabilities(self, hidden: np.ndarray) -> np.ndarray:
        return _sigmoid(hidden @ self.weights.T + self.visible_bias)

    def gibbs(self, visible: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample hidden and visible units through k alternating Gibbs steps."""
        visible_state = visible.astype(np.float32, copy=True)
        hidden_probability = self.hidden_probabilities(visible_state)
        hidden_state = (self.rng.random(hidden_probability.shape) < hidden_probability).astype(np.float32)
        for _ in range(max(k, 1)):
            visible_probability = self.visible_probabilities(hidden_state)
            visible_state = (self.rng.random(visible_probability.shape) < visible_probability).astype(np.float32)
            hidden_probability = self.hidden_probabilities(visible_state)
            hidden_state = (self.rng.random(hidden_probability.shape) < hidden_probability).astype(np.float32)
        return visible_state, hidden_state

    def fit(self, values: np.ndarray, *, epochs: int = 35, batch_size: int = 64, learning_rate: float = 0.03, k: int = 1) -> list[float]:
        if values.ndim != 2 or values.shape[1] != self.n_visible or len(values) < 16:
            raise ValueError("RBM training data must be a 2D binary feature matrix with at least 16 rows.")
        data = (values > 0.5).astype(np.float32)
        history: list[float] = []
        for _ in range(epochs):
            order = self.rng.permutation(len(data))
            batch_losses: list[float] = []
            for start in range(0, len(data), batch_size):
                positive = data[order[start : start + batch_size]]
                positive_hidden_probability = self.hidden_probabilities(positive)
                negative_visible, _ = self.gibbs(positive, k=k)
                negative_hidden_probability = self.hidden_probabilities(negative_visible)
                scale = max(len(positive), 1)
                self.weights += learning_rate * ((positive.T @ positive_hidden_probability - negative_visible.T @ negative_hidden_probability) / scale)
                self.visible_bias += learning_rate * np.mean(positive - negative_visible, axis=0)
                self.hidden_bias += learning_rate * np.mean(positive_hidden_probability - negative_hidden_probability, axis=0)
                batch_losses.append(float(np.mean((positive - self.visible_probabilities(self.hidden_probabilities(positive))) ** 2)))
            history.append(float(np.mean(batch_losses)))
        return history

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.hidden_probabilities(values.astype(np.float32))

    def reconstruction_errors(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        reconstructed = self.visible_probabilities(self.hidden_probabilities(values))
        return np.mean((values - reconstructed) ** 2, axis=1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        import tensorflow as tf

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        visible = tf.keras.Input(shape=(self.n_visible,), name="visible")
        hidden_layer = tf.keras.layers.Dense(self.n_hidden, activation="sigmoid", name="hidden")
        reconstruction_layer = tf.keras.layers.Dense(self.n_visible, activation="sigmoid", name="reconstruction")
        reconstruction = reconstruction_layer(hidden_layer(visible))
        keras_model = tf.keras.Model(visible, reconstruction, name="trace_rbm_reconstruction")
        hidden_layer.set_weights([self.weights, self.hidden_bias])
        reconstruction_layer.set_weights([self.weights.T, self.visible_bias])
        keras_model.save(target)
        (target.parent / ARTIFACT_FILES["rbm_metadata"]).write_text(
            json.dumps({"format_version": 1, "feature_names": RBM_FEATURE_NAMES,
                        "visible_units": self.n_visible, "hidden_units": self.n_hidden}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RestrictedBoltzmannMachine":
        import tensorflow as tf

        target = Path(path)
        metadata = json.loads((target.parent / ARTIFACT_FILES["rbm_metadata"]).read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or tuple(metadata.get("feature_names", ())) != RBM_FEATURE_NAMES:
            raise ValueError("RBM artifact feature schema is missing or incompatible. Retrain the models.")
        keras_model = tf.keras.models.load_model(target, compile=False)
        weights, hidden_bias = keras_model.get_layer("hidden").get_weights()
        decoder_weights, visible_bias = keras_model.get_layer("reconstruction").get_weights()
        if weights.shape != (metadata["visible_units"], metadata["hidden_units"]) or not np.allclose(decoder_weights, weights.T):
            raise ValueError("RBM model weights are incompatible. Retrain the models.")
        model = cls(weights.shape[0], weights.shape[1])
        model.weights = weights.astype(np.float32)
        model.visible_bias = visible_bias.astype(np.float32)
        model.hidden_bias = hidden_bias.astype(np.float32)
        return model
