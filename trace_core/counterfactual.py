"""Markov-informed what-if analysis; it never executes response actions."""

from __future__ import annotations

import pandas as pd

from trace_core.config import MARKOV_STATES
from trace_core.markov import MarkovTimeline, event_stage

INTERVENTIONS = {
    "Disable the account": {
        "effectiveness": 0.72,
        "anchors": {"initial_access", "privilege_change"},
        "relevance": {"initial_access": 1.0, "privilege_change": 0.96, "suspicious_execution": 0.78, "external_communication": 0.60, "data_transfer": 0.42, "evidence_removal": 0.35, "normal": 0.25},
    },
    "Isolate the affected system": {
        "effectiveness": 0.88,
        "anchors": {"suspicious_execution", "external_communication"},
        "relevance": {"initial_access": 0.55, "privilege_change": 0.74, "suspicious_execution": 1.0, "external_communication": 0.96, "data_transfer": 0.87, "evidence_removal": 0.78, "normal": 0.25},
    },
    "Block the external destination": {
        "effectiveness": 0.80,
        "anchors": {"external_communication"},
        "relevance": {"initial_access": 0.10, "privilege_change": 0.16, "suspicious_execution": 0.34, "external_communication": 1.0, "data_transfer": 0.88, "evidence_removal": 0.48, "normal": 0.08},
    },
}

_STAGE_RISK = {
    "normal": 0.10,
    "initial_access": 0.45,
    "privilege_change": 0.85,
    "suspicious_execution": 1.00,
    "external_communication": 1.05,
    "data_transfer": 1.30,
    "evidence_removal": 0.75,
}


def simulate_intervention(events: pd.DataFrame, markov: MarkovTimeline, action: str, pivot_event_id: int) -> dict[str, object]:
    if action not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {action}")
    if events.empty or pivot_event_id not in set(events["event_id"].astype(int)):
        raise ValueError("Choose an intervention point that exists in the event timeline.")
    config = INTERVENTIONS[action]
    pivot_row = int(events["event_id"].astype(int).eq(int(pivot_event_id)).to_numpy().nonzero()[0][0])
    pivot = events.iloc[pivot_row]
    pivot_stage = event_stage(pivot["event_type"])
    if pivot_stage not in config["anchors"]:
        raise ValueError(f"{action} is not modeled as an intervention at {pivot_stage.replace('_', ' ')}.")
    downstream = events.iloc[pivot_row + 1 :]
    efficacy = float(config["effectiveness"])
    state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
    previous = pivot_stage
    path_probability = 1.0
    stage_changes = 0
    baseline_risk = 0.0
    residual_risk = 0.0
    for _, row in downstream.iterrows():
        stage = event_stage(row["event_type"])
        if stage != previous:
            probability = float(markov.transition[state_index[previous], state_index[stage]])
            path_probability *= max(probability, 0.025)
            stage_changes += 1
            previous = stage
        event_risk = _STAGE_RISK[stage] * max(path_probability, 0.05)
        stage_relevance = float(config["relevance"].get(stage, 0.25))
        intervention_effect = efficacy * stage_relevance * max(0.45, 1.0 - 0.10 * max(stage_changes - 1, 0))
        baseline_risk += event_risk
        residual_risk += event_risk * (1.0 - intervention_effect)
    reduction = 0.0 if baseline_risk <= 1e-9 else 1.0 - residual_risk / baseline_risk
    return {
        "action": action,
        "pivot_event_id": int(pivot_event_id),
        "pivot_description": str(pivot["raw_event"]),
        "pivot_stage": pivot_stage,
        "downstream_event_count": int(len(downstream)),
        "baseline_expected_risk": round(baseline_risk, 3),
        "counterfactual_expected_risk": round(residual_risk, 3),
        "estimated_risk_reduction_percent": round(100.0 * reduction, 1),
        "assumed_near_term_control_effectiveness_percent": round(100.0 * efficacy, 1),
        "markov_model_trained": markov.trained,
        "limitation": "Illustrative model estimate based on fixed control-effect assumptions and learned stage transitions; it does not predict a guaranteed real-world outcome.",
    }
