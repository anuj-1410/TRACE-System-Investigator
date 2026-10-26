"""End-to-end batch investigation pipeline for normalized event logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trace_core.bayesian import collect_evidence, evidence_explanations, infer_root_cause
from trace_core.config import ARTIFACT_DIR, ARTIFACT_FILES, INCIDENT_CLASSES
from trace_core.impact import assess_impact, recommendations
from trace_core.markov import MarkovTimeline
from trace_core.models.classifier import IncidentClassifier
from trace_core.models.rbm import RestrictedBoltzmannMachine
from trace_core.preprocessing import rbm_feature_matrix, sequence_feature_vector, vae_feature_matrix

if TYPE_CHECKING:
    from trace_core.models.vae import VariationalAutoencoder


@dataclass
class LoadedModels:
    vae: VariationalAutoencoder | None
    vae_threshold: float | None
    rbm: RestrictedBoltzmannMachine | None
    classifier: IncidentClassifier | None
    markov: MarkovTimeline
    statuses: dict[str, str]


def load_models(artifact_dir: str | Path = ARTIFACT_DIR) -> LoadedModels:
    """Load generated artifacts, keeping the UI usable when training is pending."""
    artifact_dir = Path(artifact_dir)
    statuses: dict[str, str] = {}
    vae = None
    vae_threshold = None
    rbm = None
    classifier = None
    markov = MarkovTimeline()
    loaders = (
        ("vae", ARTIFACT_FILES["vae"]),
        ("rbm", ARTIFACT_FILES["rbm"]),
        ("classifier", ARTIFACT_FILES["classifier"]),
        ("markov", ARTIFACT_FILES["markov"]),
    )
    for key, filename in loaders:
        path = artifact_dir / filename
        if not path.exists():
            statuses[key] = "Not trained yet"
            continue
        try:
            if key == "vae":
                from trace_core.models.vae import load_vae

                vae, vae_threshold = load_vae(path)
            elif key == "rbm":
                rbm = RestrictedBoltzmannMachine.load(path)
            elif key == "classifier":
                classifier = IncidentClassifier.load(path)
            elif key == "markov":
                markov = MarkovTimeline.load(path)
            statuses[key] = "Loaded"
        except Exception as exc:  # Keep the report available and expose artifact state to the user.
            statuses[key] = f"Could not load: {type(exc).__name__}: {exc}"
    return LoadedModels(vae, vae_threshold, rbm, classifier, markov, statuses)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    probabilities = np.exp(np.clip(shifted, -60, 60))
    return probabilities / probabilities.sum()


def _heuristic_class_probabilities(events: pd.DataFrame, evidence: dict[str, bool], anomaly_score: float) -> dict[str, float]:
    evidence_count = sum(bool(value) for value in evidence.values())
    event_diversity = int(events["event_type"].nunique())
    failures = int(events["event_type"].eq("login_failure").sum())
    external_count = int(events["event_type"].isin(["external_connection", "lateral_movement"]).sum())
    sensitive_access = int(events["event_type"].eq("file_access").sum())
    scores = {
        "Normal Activity": 2.15 - 0.95 * evidence_count - 0.75 * anomaly_score,
        "Account Compromise": -0.65 + 2.25 * evidence["unusual_authentication"] + 0.25 * evidence["external_communication"] + 0.10 * evidence["repeated_auth_failures"],
        "Privilege Escalation": -0.60 + 2.50 * evidence["privilege_change"] + 0.30 * evidence["suspicious_process"],
        "Suspicious Process Execution": -0.65 + 2.60 * evidence["suspicious_process"] + 0.15 * int(events["event_type"].eq("process_start").sum() > 1),
        "Data Exfiltration": -0.65 + 2.70 * evidence["large_transfer"] + 0.25 * evidence["external_communication"] + 0.20 * min(sensitive_access, 2),
        "Denial-of-Service-like Activity": -0.65 + 0.15 * failures + 0.75 * int(failures >= 6) + 0.15 * max(external_count - 1, 0) + 0.80 * int(events["event_type"].eq("firewall_block").any()),
        "Multi-stage Attack": -1.30 + 0.45 * evidence_count + 0.18 * min(event_diversity, 6) + 1.00 * int(evidence_count >= 4),
    }
    probabilities = _softmax(np.asarray([scores[label] for label in INCIDENT_CLASSES], dtype=np.float64) / 0.45)
    return {label: float(probability) for label, probability in zip(INCIDENT_CLASSES, probabilities)}


def _top_probability(probabilities: dict[str, float]) -> tuple[str, float]:
    return max(probabilities.items(), key=lambda item: item[1])


def _anomaly_analysis(events: pd.DataFrame, models: LoadedModels) -> tuple[float, list[dict[str, object]], str]:
    features = vae_feature_matrix(events)
    if models.vae is not None and models.vae_threshold is not None:
        from trace_core.models.vae import reconstruction_errors

        errors = reconstruction_errors(models.vae, features)
        threshold = max(models.vae_threshold, 1e-8)
        ratios = errors / threshold
        scores = 1.0 - np.exp(-ratios)
        count = min(max(1, int(np.ceil(len(scores) * 0.10))), len(scores))
        overall = float(np.mean(np.sort(scores)[-count:]))
        ranked = np.argsort(errors)[::-1][: min(12, len(errors))]
        return overall, [
            {"event_id": int(events.iloc[index]["event_id"]), "reconstruction_error": float(errors[index]), "threshold_ratio": float(ratios[index]), "anomaly_score": float(scores[index]), "flagged": bool(errors[index] > threshold)}
            for index in ranked
        ], "Trained TensorFlow/Keras sequence VAE reconstruction error"

    evidence = collect_evidence(events)
    weights = {
        "unusual_authentication": 0.19,
        "privilege_change": 0.16,
        "suspicious_process": 0.18,
        "external_communication": 0.14,
        "large_transfer": 0.20,
        "evidence_removal": 0.09,
        "repeated_auth_failures": 0.12,
    }
    score = 0.08 + sum(weights[key] for key, active in evidence.items() if active)
    if evidence["external_communication"] and (evidence["large_transfer"] or evidence["suspicious_process"]):
        score += 0.08
    score = float(np.clip(score, 0.0, 0.98))
    per_event = []
    evidence_ids = {int(event_id) for row in evidence_explanations(events, evidence) for event_id in row["event_ids"]}
    for _, event in events.iterrows():
        flagged = int(event["event_id"]) in evidence_ids
        per_event.append({"event_id": int(event["event_id"]), "reconstruction_error": None, "threshold_ratio": None, "anomaly_score": score if flagged else score * 0.35, "flagged": flagged})
    ranked = sorted(per_event, key=lambda row: row["anomaly_score"], reverse=True)[: min(12, len(per_event))]
    return score, ranked, "Transparent evidence heuristic (VAE artifact not trained)"


def _classifier_probabilities(events: pd.DataFrame, evidence: dict[str, bool], anomaly_score: float, models: LoadedModels) -> tuple[dict[str, float], str]:
    if models.classifier is None:
        return _heuristic_class_probabilities(events, evidence, anomaly_score), "Evidence-based fallback"
    vector = sequence_feature_vector(events).reshape(1, -1)
    predicted = models.classifier.predict_proba(vector)[0]
    probabilities = {label: 0.0 for label in INCIDENT_CLASSES}
    for label, probability in zip(models.classifier.classes, predicted):
        if label in probabilities:
            probabilities[label] = float(probability)
    total = sum(probabilities.values())
    if total <= 0:
        return _heuristic_class_probabilities(events, evidence, anomaly_score), "Evidence-based fallback (classifier classes did not match)"
    return {label: probability / total for label, probability in probabilities.items()}, "Trained logistic regression on incident-sequence features"


def _rbm_summary(events: pd.DataFrame, model: RestrictedBoltzmannMachine | None) -> dict[str, object]:
    if model is None:
        return {"available": False, "message": "RBM artifact not trained; hidden-pattern analysis will appear after training."}
    visible = rbm_feature_matrix(events)
    activations = model.transform(visible)
    averages = activations.mean(axis=0)
    top_units = np.argsort(averages)[::-1][: min(5, len(averages))]
    reconstruction_error = model.reconstruction_errors(visible)
    return {
        "available": True,
        "mean_reconstruction_error": float(np.mean(reconstruction_error)),
        "mean_hidden_activation": float(np.mean(averages)),
        "active_patterns": [{"unit": int(index + 1), "activation": float(averages[index])} for index in top_units],
        "description": "Latent unit activations summarize co-occurring binary event features; individual units do not have a fixed human label.",
    }


def analyze_events(events: pd.DataFrame, models: LoadedModels | None = None) -> dict[str, object]:
    """Produce the complete evidence-linked investigation report."""
    if events.empty:
        raise ValueError("At least one event is required for analysis.")
    if models is None:
        models = load_models()
    normalized = events.reset_index(drop=True).copy()
    if "incident_id" in normalized and normalized["incident_id"].fillna("").astype(str).nunique() > 1:
        raise ValueError("Select a single incident_id before analyzing a case.")
    if "event_id" not in normalized:
        normalized["event_id"] = np.arange(1, len(normalized) + 1)
    if "raw_event" not in normalized:
        normalized["raw_event"] = normalized["event_type"].astype(str).str.replace("_", " ")
    normalized = normalized.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    evidence = collect_evidence(normalized)
    evidence_rows = evidence_explanations(normalized, evidence)
    anomaly_score, anomaly_events, anomaly_method = _anomaly_analysis(normalized, models)
    class_probabilities, classifier_method = _classifier_probabilities(normalized, evidence, anomaly_score, models)
    ranked_hypotheses = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
    classifier_top_type, classifier_top_probability = ranked_hypotheses[0]
    incident_type, confidence = classifier_top_type, classifier_top_probability
    classifier_detected = incident_type != "Normal Activity" and confidence >= 0.34
    anomaly_flagged = bool(
        any(row["flagged"] for row in anomaly_events) if models.vae is not None and models.vae_threshold is not None
        else anomaly_score >= 0.62 and any(evidence.values())
    )
    non_normal_type, non_normal_probability = max(
        ((label, probability) for label, probability in ranked_hypotheses if label != "Normal Activity"),
        key=lambda item: item[1],
    )
    anomaly_supported = anomaly_flagged and any(evidence.values()) and non_normal_probability >= 0.25
    incident_detected = bool(classifier_detected or anomaly_supported)
    if anomaly_supported and not classifier_detected:
        incident_type, confidence = non_normal_type, non_normal_probability
        classifier_method += "; anomaly plus observed evidence supported a non-normal hypothesis"
    if not incident_detected:
        incident_type = "Normal Activity"
        confidence = class_probabilities[incident_type]
    root_causes = infer_root_cause(evidence)
    timeline = models.markov.annotate(normalized)
    markov_likelihood = models.markov.sequence_likelihood(normalized)
    impact = assess_impact(normalized, anomaly_score, confidence if incident_detected else 0.0, evidence)
    response = recommendations(normalized, evidence, incident_type, evidence_rows)
    top_cause = root_causes[0]
    status_rows = {key: value for key, value in models.statuses.items()}
    model_availability = {
        "vae": models.vae is not None,
        "rbm": models.rbm is not None,
        "classifier": models.classifier is not None,
        "markov": models.markov.trained,
    }
    start_time = normalized["timestamp"].min()
    end_time = normalized["timestamp"].max()
    return {
        "project": "TRACE — Threat Reconstruction, Analysis, Causality & Explanation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incident_detected": incident_detected,
        "anomaly_flagged": anomaly_flagged,
        "incident_type": incident_type,
        "incident_confidence": float(confidence if incident_detected and incident_type != "Normal Activity" else 0.0),
        "selected_hypothesis_probability": float(confidence),
        "classifier_top_hypothesis": classifier_top_type,
        "classifier_top_probability": float(classifier_top_probability),
        "incident_probabilities": [{"incident_type": label, "probability": float(probability)} for label, probability in ranked_hypotheses],
        "anomaly_score": float(anomaly_score),
        "anomaly_method": anomaly_method,
        "anomaly_events": anomaly_events,
        "classifier_method": classifier_method,
        "evidence": evidence_rows,
        "root_causes": root_causes,
        "probable_root_cause": top_cause["cause"] if incident_detected and any(evidence.values()) else "Undetermined from available evidence",
        "root_cause_probability": float(top_cause["probability"]) if incident_detected and any(evidence.values()) else 0.0,
        "root_cause_method": "Bayesian network exact enumeration over elicited conditional likelihoods; hypotheses remain provisional when no incident is established",
        "timeline": timeline,
        "markov_sequence_likelihood": markov_likelihood,
        "next_stage_forecast": models.markov.forecast_next(normalized),
        "markov_method": "Learned transition matrix" if models.markov.trained else "Curated prototype transition prior",
        "rbm_patterns": _rbm_summary(normalized, models.rbm),
        "impact": impact,
        "recommendations": response,
        "counterfactual": {"available": False, "message": "Select a response and timeline point in the What-if simulation tab to attach a scenario estimate."},
        "event_count": int(len(normalized)),
        "time_window": {"start": pd.Timestamp(start_time).isoformat(), "end": pd.Timestamp(end_time).isoformat()},
        "event_type_counts": {str(key): int(value) for key, value in normalized["event_type"].value_counts().items()},
        "model_availability": model_availability,
        "model_status": status_rows,
        "content_provenance": {
            "Anomaly signal": anomaly_method,
            "Incident hypotheses": classifier_method,
            "Timeline and forecast": "Learned Markov transition matrix" if models.markov.trained else "Curated prototype Markov transition prior",
            "Root cause": "Bayesian calculation using observed evidence and manually specified likelihoods",
            "Evidence descriptions": "Fixed explanations attached to matching event IDs",
            "Impact and severity": "Rule-based counts, thresholds, and weighted estimates from log fields",
            "Response suggestions": "Fixed guidance selected by observed evidence and incident category",
            "What-if estimate": "Fixed intervention-effect assumptions combined with Markov transitions",
            "Report text": "Templates populated with case-specific results; no external language model",
        },
        "limitations": [
            "Probabilities are prototype estimates and are not calibrated for production security decisions.",
            "The Bayesian network uses elicited conditional likelihoods; retrain or validate them against representative incident labels before research claims.",
            "A missing event in an incomplete log feed is not evidence that the event did not occur.",
            "What-if estimates are simulations and do not establish what would have happened in reality.",
        ],
    }


def read_training_summary(artifact_dir: str | Path = ARTIFACT_DIR) -> dict[str, object] | None:
    path = Path(artifact_dir) / ARTIFACT_FILES["summary"]
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
