"""Audit saved TRACE artifacts on the untouched case-level test partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_core.config import ARTIFACT_DIR, INCIDENT_CLASSES
from trace_core.pipeline import analyze_events, load_models
from trace_core.preprocessing import labeled_incident_groups, normalize_events
from trace_core.synthetic import scenario_records


def evaluate(dataset_path: Path, artifact_dir: Path) -> dict[str, object]:
    groups = labeled_incident_groups(normalize_events(pd.read_csv(dataset_path)))
    if not groups:
        raise ValueError("No labeled cases were found in the dataset.")
    manifest_path = artifact_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("The recorded test split is missing: artifacts/split_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_ids = set(manifest["test_incident_ids"])
    test_groups = [(label, events) for label, events in groups if str(events.iloc[0]["incident_id"]) in test_ids]
    if len(test_groups) != len(test_ids):
        raise ValueError("Dataset case IDs do not match the recorded test split.")
    models = load_models(artifact_dir)
    if any(status != "Loaded" for status in models.statuses.values()):
        raise RuntimeError(f"All trained artifacts are required: {models.statuses}")
    truth: list[str] = []
    predicted: list[str] = []
    incident_truth: list[int] = []
    incident_predicted: list[int] = []
    errors: list[dict[str, object]] = []
    for label, events in test_groups:
        report = analyze_events(events, models)
        truth.append(label)
        predicted.append(str(report["incident_type"]))
        incident_truth.append(int(label != "Normal Activity"))
        incident_predicted.append(int(report["incident_detected"]))
        if label != report["incident_type"]:
            errors.append({"incident_id": str(events.iloc[0]["incident_id"]), "true_label": label,
                           "predicted_label": str(report["incident_type"]),
                           "classifier_top_hypothesis": str(report["classifier_top_hypothesis"]),
                           "anomaly_flagged": bool(report["anomaly_flagged"]),
                           "event_types": events["event_type"].astype(str).tolist()})
    result = {
        "source": str(dataset_path.resolve()),
        "evaluation_scope": "End-to-end pipeline on case-level test split of simulated data; no real-world validation",
        "seed": manifest["seed"],
        "case_count": len(truth),
        "class_labels": list(INCIDENT_CLASSES),
        "category_accuracy": float(metrics.accuracy_score(truth, predicted)),
        "category_balanced_accuracy": float(metrics.balanced_accuracy_score(truth, predicted)),
        "category_macro_f1": float(metrics.f1_score(truth, predicted, labels=INCIDENT_CLASSES, average="macro", zero_division=0)),
        "category_weighted_f1": float(metrics.f1_score(truth, predicted, labels=INCIDENT_CLASSES, average="weighted", zero_division=0)),
        "category_report": metrics.classification_report(truth, predicted, labels=INCIDENT_CLASSES, output_dict=True, zero_division=0),
        "category_confusion_matrix": metrics.confusion_matrix(truth, predicted, labels=INCIDENT_CLASSES).tolist(),
        "incident_precision": float(metrics.precision_score(incident_truth, incident_predicted, zero_division=0)),
        "incident_recall": float(metrics.recall_score(incident_truth, incident_predicted, zero_division=0)),
        "incident_f1": float(metrics.f1_score(incident_truth, incident_predicted, zero_division=0)),
        "incident_false_positive_rate": float(np.mean([p for t, p in zip(incident_truth, incident_predicted) if t == 0])),
        "incident_confusion_matrix_labels": ["normal", "incident"],
        "incident_confusion_matrix": metrics.confusion_matrix(incident_truth, incident_predicted, labels=[0, 1]).tolist(),
        "errors": errors,
    }
    demo_results = []
    for label in INCIDENT_CLASSES:
        report = analyze_events(scenario_records(label, f"audit-demo-{INCIDENT_CLASSES.index(label)}", seed=19), models)
        demo_results.append({"true_label": label, "predicted_label": str(report["incident_type"]),
                             "incident_detected": bool(report["incident_detected"])})
    result["separate_demo_scenarios"] = demo_results
    result["separate_demo_accuracy"] = float(np.mean([row["true_label"] == row["predicted_label"] for row in demo_results]))
    (artifact_dir / "evaluation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "trace_synthetic_cases.csv")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments.dataset, arguments.artifact_dir), indent=2))
