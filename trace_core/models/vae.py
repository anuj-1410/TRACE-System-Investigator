"""TensorFlow/Keras variational autoencoder for ordered event windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from trace_core.config import ARTIFACT_FILES, VAE_FEATURE_NAMES, VAE_WINDOW_SIZE


class VariationalAutoencoder:
    """An encoder and decoder with stochastic training and stable inference."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 12,
        *,
        encoder: tf.keras.Model | None = None,
        decoder: tf.keras.Model | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        if encoder is None:
            inputs = tf.keras.Input(shape=(input_dim,), name="event_window")
            hidden = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
            hidden = tf.keras.layers.Dense(hidden_dim // 2, activation="relu")(hidden)
            mean = tf.keras.layers.Dense(latent_dim, name="latent_mean")(hidden)
            log_variance = tf.keras.layers.Dense(latent_dim, name="latent_log_variance")(hidden)
            encoder = tf.keras.Model(inputs, [mean, log_variance], name="trace_vae_encoder")
        if decoder is None:
            latent_inputs = tf.keras.Input(shape=(latent_dim,), name="latent_vector")
            hidden = tf.keras.layers.Dense(hidden_dim // 2, activation="relu")(latent_inputs)
            hidden = tf.keras.layers.Dense(hidden_dim, activation="relu")(hidden)
            reconstruction = tf.keras.layers.Dense(input_dim, activation="sigmoid")(hidden)
            decoder = tf.keras.Model(latent_inputs, reconstruction, name="trace_vae_decoder")
        self.encoder = encoder
        self.decoder = decoder

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def reconstruct(self, values: tf.Tensor) -> tf.Tensor:
        """Decode the posterior mean for repeatable anomaly scores."""
        mean, _ = self.encoder(values, training=False)
        return self.decoder(mean, training=False)


def reconstruction_errors(model: VariationalAutoencoder, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    errors: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = tf.convert_to_tensor(values[start : start + batch_size], dtype=tf.float32)
        reconstructed = model.reconstruct(batch)
        error = tf.reduce_mean(tf.square(reconstructed - batch), axis=1)
        errors.append(error.numpy())
    return np.concatenate(errors).astype(np.float32)


def train_vae(
    normal_features: np.ndarray,
    output_path: str | Path,
    *,
    calibration_sequences: list[np.ndarray] | None = None,
    epochs: int = 45,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    latent_dim: int = 12,
    seed: int = 17,
) -> dict[str, float | int | str | list[float]]:
    if normal_features.ndim != 2 or len(normal_features) < 16:
        raise ValueError("VAE training needs at least 16 normal event windows.")
    if normal_features.shape[1] != len(VAE_FEATURE_NAMES):
        raise ValueError("VAE feature columns do not match the current event-window schema.")
    if not np.isfinite(normal_features).all():
        raise ValueError("VAE training features must be finite.")
    if epochs < 1 or batch_size < 1 or latent_dim < 1 or learning_rate <= 0:
        raise ValueError("VAE epochs, batch size, latent dimension, and learning rate must be positive.")

    tf.keras.utils.set_random_seed(seed)
    model = VariationalAutoencoder(normal_features.shape[1], latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    data = tf.data.Dataset.from_tensor_slices(normal_features.astype(np.float32))
    data = data.shuffle(min(len(normal_features), 10_000), seed=seed).batch(batch_size)
    final_loss = 0.0
    epoch_history: list[float] = []
    for _ in range(epochs):
        losses: list[float] = []
        for batch in data:
            with tf.GradientTape() as tape:
                mean, log_variance = model.encoder(batch, training=True)
                log_variance = tf.clip_by_value(log_variance, -10.0, 8.0)
                sampled = mean + tf.exp(0.5 * log_variance) * tf.random.normal(tf.shape(mean))
                reconstructed = model.decoder(sampled, training=True)
                reconstruction_loss = tf.reduce_mean(tf.square(reconstructed - batch))
                kl_loss = -0.5 * tf.reduce_mean(1.0 + log_variance - tf.square(mean) - tf.exp(log_variance))
                loss = reconstruction_loss + 0.01 * kl_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            )
            losses.append(float(loss.numpy()))
        final_loss = float(np.mean(losses))
        epoch_history.append(final_loss)

    if calibration_sequences:
        case_max_errors = [float(np.max(reconstruction_errors(model, case))) for case in calibration_sequences if len(case)]
        if not case_max_errors:
            raise ValueError("VAE calibration needs nonempty normal incident sequences.")
        threshold = float(max(np.quantile(case_max_errors, 0.95), 1e-7))
        threshold_source = "95th percentile of held-out normal case maximum errors"
    else:
        errors = reconstruction_errors(model, normal_features)
        threshold = float(max(np.quantile(errors, 0.99), np.quantile(errors, 0.50) + 1e-7))
        threshold_source = "99th percentile of normal training window errors (no calibration cases supplied)"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.encoder.save(target.parent / ARTIFACT_FILES["vae_encoder"])
    model.decoder.save(target.parent / ARTIFACT_FILES["vae_decoder"])
    metadata = {
        "format_version": 4,
        "framework": "TensorFlow/Keras",
        "tensorflow_version": tf.__version__,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "window_size": VAE_WINDOW_SIZE,
        "feature_names": list(VAE_FEATURE_NAMES),
        "threshold": threshold,
        "threshold_source": threshold_source,
    }
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"rows": int(len(normal_features)), "epochs": int(epochs), "final_loss": final_loss, "epoch_loss": epoch_history, "threshold": threshold, "threshold_source": threshold_source}


def load_vae(path: str | Path) -> tuple[VariationalAutoencoder, float]:
    target = Path(path)
    metadata = json.loads(target.read_text(encoding="utf-8"))
    if (
        metadata.get("format_version") != 4
        or metadata.get("window_size") != VAE_WINDOW_SIZE
        or tuple(metadata.get("feature_names", ())) != VAE_FEATURE_NAMES
        or metadata.get("input_dim") != len(VAE_FEATURE_NAMES)
    ):
        raise ValueError("VAE artifact feature schema is incompatible. Retrain the models.")
    encoder = tf.keras.models.load_model(target.parent / ARTIFACT_FILES["vae_encoder"], compile=False)
    decoder = tf.keras.models.load_model(target.parent / ARTIFACT_FILES["vae_decoder"], compile=False)
    if encoder.input_shape[-1] != int(metadata["input_dim"]) or decoder.input_shape[-1] != int(metadata["latent_dim"]):
        raise ValueError("VAE model dimensions do not match metadata. Retrain the models.")
    model = VariationalAutoencoder(
        int(metadata["input_dim"]), hidden_dim=int(metadata["hidden_dim"]),
        latent_dim=int(metadata["latent_dim"]), encoder=encoder, decoder=decoder,
    )
    return model, float(metadata["threshold"])
