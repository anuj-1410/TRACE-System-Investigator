"""Train TRACE artifacts from controlled scenarios or investigator data.

Default data are synthetic and support a reproducible classroom demonstration.
Use --normal-csv and --labelled-csv to substitute curated project data.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_core.config import ARTIFACT_FILES, ARTIFACT_DIR, MARKOV_STATES
from trace_core.markov import MarkovTimeline, collapsed_stages
from trace_core.models.classifier import IncidentClassifier
from trace_core.models.rbm import RestrictedBoltzmannMachine
from trace_core.models.vae import reconstruction_errors, train_vae
from trace_core.preprocessing import (
    labeled_incident_groups,
    normalize_events,
    rbm_feature_matrix,
    sequence_feature_vector,
    vae_feature_matrix,
)
from trace_core.training_data import create_training_cases


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _load_normal_data(path: str | None, training_groups: list[tuple[str, pd.DataFrame]]) -> tuple[np.ndarray, int, str]:
    if path:
        events = normalize_events(pd.read_csv(path))
        if "incident_type" in events and events["incident_type"].notna().any():
            normal = events[events["incident_type"].eq("Normal Activity")]
            if normal.empty:
                raise ValueError("--normal-csv contains incident labels but no Normal Activity rows.")
            events = normal
        if len(events) < 16:
            raise ValueError("The normal CSV must provide at least 16 usable event rows.")
        return vae_feature_matrix(events), len(events), str(Path(path).resolve())
    normal_events = [sequence for label, sequence in training_groups if label == "Normal Activity"]
    if not normal_events:
        raise ValueError("VAE training requires normal rows. Supply --normal-csv or include the Normal Activity class.")
    return np.concatenate([vae_feature_matrix(events) for events in normal_events]), sum(map(len, normal_events)), "generated or labeled Normal Activity sequences"


def _anomaly_metrics(test_groups: list[tuple[str, pd.DataFrame]], model, threshold: float) -> dict[str, object]:
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    for label, events in test_groups:
        errors = reconstruction_errors(model, vae_feature_matrix(events))
        score = float(np.max(errors)) if len(errors) else 0.0
        labels.append(int(label != "Normal Activity"))
        scores.append(score)
        predictions.append(int(score > threshold))
    result: dict[str, object] = {
        "evaluation_source": "held-out synthetic/labelled incident sequences; prototype estimate",
        "sequence_count": len(labels),
        "threshold": float(threshold),
        "precision": float(metrics.precision_score(labels, predictions, zero_division=0)),
        "recall": float(metrics.recall_score(labels, predictions, zero_division=0)),
        "f1": float(metrics.f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(metrics.average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        "incident_prevalence": float(np.mean(labels)),
        "confusion_matrix_labels": ["normal", "incident"],
        "confusion_matrix": metrics.confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if len(set(labels)) == 2:
        result["roc_auc"] = float(metrics.roc_auc_score(labels, scores))
    else:
        result["roc_auc"] = None
    return result


def _markov_metrics(test_groups: list[tuple[str, pd.DataFrame]], model: MarkovTimeline) -> dict[str, object]:
    state_index = {state: index for index, state in enumerate(MARKOV_STATES)}
    correct = 0
    count = 0
    losses: list[float] = []
    for _, events in test_groups:
        stages = collapsed_stages(events)
        for source, target in zip(stages, stages[1:]):
            probabilities = model.transition[state_index[source]]
            correct += int(MARKOV_STATES[int(np.argmax(probabilities))] == target)
            count += 1
            losses.append(-float(np.log(max(probabilities[state_index[target]], 1e-12))))
    return {"held_out_transitions": count, "next_stage_top1_accuracy": float(correct / count) if count else None,
            "mean_negative_log_likelihood": float(np.mean(losses)) if losses else None}


def train(args: argparse.Namespace) -> dict[str, object]:
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    if args.labelled_csv:
        normalized = normalize_events(pd.read_csv(args.labelled_csv))
        if normalized["incident_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Every labeled row needs an incident_id so cases can be split without leakage.")
        if normalized["incident_type"].isna().any():
            raise ValueError("Every labeled row needs one of the seven supported incident_type labels.")
        conflicting = normalized.groupby("incident_id")["incident_type"].nunique()
        if (conflicting > 1).any():
            raise ValueError("Each incident_id must have one consistent incident_type label.")
        groups = labeled_incident_groups(normalized)
        if len(groups) < 8:
            raise ValueError("The labeled CSV needs at least 4 distinct incident sequences in each of at least 2 supported classes.")
        source = str(Path(args.labelled_csv).resolve())
    else:
        groups = create_training_cases(seed=args.seed)
        source = "varied, imbalanced synthetic audit-log simulation"

    label_counts = pd.Series([label for label, _ in groups]).value_counts()
    if len(label_counts) < 2 or int(label_counts.min()) < 8:
        raise ValueError("Provide at least 8 incident sequences for each of at least 2 supported classes so training, calibration, and test cases remain separate.")
    indices = np.arange(len(groups))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.20, random_state=args.seed, stratify=[groups[index][0] for index in indices])
    train_indices, calibration_indices = train_test_split(train_val_indices, test_size=0.25, random_state=args.seed + 1, stratify=[groups[index][0] for index in train_val_indices])
    training_groups = [groups[index] for index in train_indices]
    calibration_groups = [groups[index] for index in calibration_indices]
    test_groups = [groups[index] for index in test_indices]

    normal_features, normal_rows, normal_source = _load_normal_data(args.normal_csv, training_groups)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "seed": args.seed,
        "source": source,
        "train_incident_ids": [str(events.iloc[0]["incident_id"]) for _, events in training_groups],
        "calibration_incident_ids": [str(events.iloc[0]["incident_id"]) for _, events in calibration_groups],
        "test_incident_ids": [str(events.iloc[0]["incident_id"]) for _, events in test_groups],
    }
    vae_result = train_vae(
        normal_features,
        artifact_dir / ARTIFACT_FILES["vae"],
        calibration_sequences=[vae_feature_matrix(events) for label, events in calibration_groups if label == "Normal Activity"],
        epochs=args.vae_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    from trace_core.models.vae import load_vae
    vae_model, vae_threshold = load_vae(artifact_dir / ARTIFACT_FILES["vae"])

    training_events = pd.concat([sequence for _, sequence in training_groups], ignore_index=True)
    rbm_visible = rbm_feature_matrix(training_events)
    rbm = RestrictedBoltzmannMachine(rbm_visible.shape[1], n_hidden=args.rbm_hidden, seed=args.seed + 1)
    rbm_history = rbm.fit(rbm_visible, epochs=args.rbm_epochs, batch_size=args.batch_size, k=args.cd_k)
    rbm.save(artifact_dir / ARTIFACT_FILES["rbm"])

    x_train = np.stack([sequence_feature_vector(sequence) for _, sequence in training_groups])
    y_train = np.asarray([label for label, _ in training_groups], dtype=str)
    x_calibration = np.stack([sequence_feature_vector(sequence) for _, sequence in calibration_groups])
    y_calibration = np.asarray([label for label, _ in calibration_groups], dtype=str)
    x_test = np.stack([sequence_feature_vector(sequence) for _, sequence in test_groups])
    y_test = np.asarray([label for label, _ in test_groups], dtype=str)
    classes = np.unique(y_train)
    inverse_frequency = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    tuning_results = []
    for exponent in (0.0, 0.5, 1.0):
        weights = None if exponent == 0 else {str(name): float(weight ** exponent)
                                               for name, weight in zip(classes, inverse_frequency)}
        candidate, candidate_estimator = IncidentClassifier.fit(
            x_train, y_train, seed=args.seed, class_weight=weights)
        candidate_predicted = candidate_estimator.predict(x_calibration)
        tuning_results.append({
            "weight_exponent": exponent,
            "calibration_macro_f1": float(metrics.f1_score(y_calibration, candidate_predicted, average="macro", zero_division=0)),
            "calibration_balanced_accuracy": float(metrics.balanced_accuracy_score(y_calibration, candidate_predicted)),
            "calibration_log_loss": float(metrics.log_loss(
                y_calibration, candidate.predict_proba(x_calibration), labels=candidate.classes.tolist())),
            "model": candidate, "estimator": candidate_estimator,
        })
    chosen = max(tuning_results, key=lambda row: (row["calibration_macro_f1"], -row["calibration_log_loss"]))
    classifier, sklearn_model = chosen["model"], chosen["estimator"]
    classifier.save(artifact_dir / ARTIFACT_FILES["classifier"])
    predicted_labels = sklearn_model.predict(x_test)
    predicted_probabilities = classifier.predict_proba(x_test)
    train_predictions = sklearn_model.predict(x_train)
    classifier_metrics = {
        "evaluation_source": "stratified held-out incident sequences; synthetic results are demonstration metrics only",
        "sequence_count": int(len(y_test)),
        "accuracy": float(metrics.accuracy_score(y_test, predicted_labels)),
        "macro_precision": float(metrics.precision_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "macro_recall": float(metrics.recall_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "macro_f1": float(metrics.f1_score(y_test, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_test, predicted_labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_test, predicted_labels)),
        "matthews_corrcoef": float(metrics.matthews_corrcoef(y_test, predicted_labels)),
        "training_macro_f1": float(metrics.f1_score(y_train, train_predictions, average="macro", zero_division=0)),
        "selection_metric": "calibration_macro_f1",
        "chosen_class_weight_exponent": chosen["weight_exponent"],
        "calibration_candidates": [
            {key: value for key, value in result.items() if key not in ("model", "estimator")}
            for result in tuning_results
        ],
        "multiclass_log_loss": float(metrics.log_loss(y_test, predicted_probabilities, labels=classifier.classes.tolist())),
        "classes": sklearn_model.classes_.astype(str).tolist(),
        "classification_report": metrics.classification_report(y_test, predicted_labels, output_dict=True, zero_division=0),
        "confusion_matrix": metrics.confusion_matrix(y_test, predicted_labels, labels=sklearn_model.classes_).tolist(),
    }

    markov = MarkovTimeline.fit([sequence for _, sequence in training_groups])
    markov.save(artifact_dir / ARTIFACT_FILES["markov"])
    anomaly_metrics = _anomaly_metrics(test_groups, vae_model, vae_threshold)
    summary: dict[str, object] = {
        "project": "TRACE — Threat Reconstruction, Analysis, Causality & Explanation",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "normal_data_source": normal_source,
        "python_version": platform.python_version(),
        "package_versions": {name: _package_version(name) for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "streamlit", "plotly")},
        "dataset": {"incident_sequences": len(groups), "training_sequences": len(training_groups), "calibration_sequences": len(calibration_groups), "held_out_sequences": len(test_groups), "normal_event_rows_for_vae": normal_rows, "class_counts": {str(key): int(value) for key, value in label_counts.items()}, "split_unit": "incident_id", "training_fraction": 0.60, "calibration_fraction": 0.20, "test_fraction": 0.20},
        "vae": {**vae_result, "metrics": anomaly_metrics},
        "rbm": {"visible_units": int(rbm_visible.shape[1]), "hidden_units": int(args.rbm_hidden), "contrastive_divergence_k": int(args.cd_k), "epochs": int(args.rbm_epochs), "epoch_reconstruction_error": [float(value) for value in rbm_history], "final_reconstruction_error": float(rbm_history[-1]), "held_out_mean_reconstruction_error": float(np.mean([np.mean(rbm.reconstruction_errors(rbm_feature_matrix(events))) for _, events in test_groups]))},
        "classifier": classifier_metrics,
        "markov": {"states": list(MARKOV_STATES), "trained_sequences": len(training_groups), "method": "maximum-likelihood transition counts with additive smoothing", "metrics": _markov_metrics(test_groups, markov)},
        "evaluation_note": "Synthetic and elicited-probability results are educational prototype measurements, not evidence of production performance.",
    }
    (artifact_dir / ARTIFACT_FILES["summary"]).write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (artifact_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    if args.export_synthetic and not args.labelled_csv:
        export_rows = pd.concat([sequence for _, sequence in groups], ignore_index=True)
        export_path = Path(args.export_synthetic)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_rows.to_csv(export_path, index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-csv", help="Optional curated normal-only CSV for VAE baseline training.")
    parser.add_argument("--labelled-csv", help="Optional labeled CSV with incident_id and incident_type for classifier/Markov training.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR), help="Output folder for model artifacts.")
    parser.add_argument("--vae-epochs", type=int, default=45)
    parser.add_argument("--rbm-epochs", type=int, default=35)
    parser.add_argument("--rbm-hidden", type=int, default=12)
    parser.add_argument("--cd-k", type=int, default=1, help="Number of alternating Gibbs steps for contrastive divergence.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--export-synthetic", default=str(PROJECT_ROOT / "data" / "trace_synthetic_cases.csv"),
                        help="Path for the generated labeled cases.")
    args = parser.parse_args()
    for name in ("vae_epochs", "rbm_epochs", "rbm_hidden", "cd_k", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    options = parse_args()
    result = train(options)
    print(json.dumps({
        "artifact_directory": str(Path(options.artifact_dir).resolve()),
        "trained_models": ["VAE", "RBM (CD-k / Gibbs sampling)", "Logistic regression classifier", "Markov Chain"],
        "classifier_macro_f1": result["classifier"]["macro_f1"],
        "vae_anomaly_f1": result["vae"]["metrics"]["f1"],
        "note": result["evaluation_note"],
    }, indent=2))
