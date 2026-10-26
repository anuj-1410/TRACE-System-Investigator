"""Markov event-stage model for incident progression and timeline annotation."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from trace_core.config import MARKOV_STATES

_EVENT_STAGE = {
    "login_success": "initial_access",
    "login_failure": "initial_access",
    "privilege_change": "privilege_change",
    "account_change": "privilege_change",
    "password_reset": "privilege_change",
    "process_start": "suspicious_execution",
    "service_install": "suspicious_execution",
    "scheduled_task": "suspicious_execution",
    "malware_detected": "suspicious_execution",
    "endpoint_alert": "suspicious_execution",
    "external_connection": "external_communication",
    "dns_query": "external_communication",
    "lateral_movement": "external_communication",
    "data_transfer": "data_transfer",
    "file_delete": "evidence_removal",
}

_STATE_LABELS = {
    "normal": "Routine activity",
    "initial_access": "Authentication / initial access",
    "privilege_change": "Privilege or account change",
    "suspicious_execution": "Process execution",
    "external_communication": "External communication",
    "data_transfer": "Data transfer",
    "evidence_removal": "Possible evidence removal",
}


def event_stage(event_type: object) -> str:
    return _EVENT_STAGE.get(str(event_type), "normal")


def collapsed_stages(events: pd.DataFrame) -> list[str]:
    stages = [event_stage(value) for value in events.get("event_type", [])]
    compact: list[str] = []
    for stage in stages:
        if not compact or compact[-1] != stage:
            compact.append(stage)
    return compact


class MarkovTimeline:
    def __init__(self, transition: np.ndarray | None = None, start: np.ndarray | None = None, *, trained: bool = False) -> None:
        count = len(MARKOV_STATES)
        if transition is None:
            transition = self._default_transition()
        if start is None:
            start = np.asarray([0.48, 0.34, 0.02, 0.02, 0.02, 0.02, 0.10], dtype=np.float64)
        self.transition = np.asarray(transition, dtype=np.float64).reshape(count, count)
        self.start = np.asarray(start, dtype=np.float64).reshape(count)
        self.transition = self.transition / np.maximum(self.transition.sum(axis=1, keepdims=True), 1e-12)
        self.start = self.start / self.start.sum()
        self.trained = trained

    @staticmethod
    def _default_transition() -> np.ndarray:
        state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
        matrix = np.full((len(MARKOV_STATES), len(MARKOV_STATES)), 0.015, dtype=np.float64)
        matrix[np.diag_indices_from(matrix)] = 0.48
        expected = (
            ("normal", "initial_access", 0.45),
            ("initial_access", "privilege_change", 0.52),
            ("privilege_change", "suspicious_execution", 0.63),
            ("suspicious_execution", "external_communication", 0.64),
            ("external_communication", "data_transfer", 0.67),
            ("data_transfer", "evidence_removal", 0.42),
        )
        for source, target, probability in expected:
            matrix[state_index[source], :] *= (1.0 - probability) / matrix[state_index[source], :].sum()
            matrix[state_index[source], state_index[target]] = probability
        return matrix / matrix.sum(axis=1, keepdims=True)

    @classmethod
    def fit(cls, sequences: list[pd.DataFrame], smoothing: float = 0.5) -> "MarkovTimeline":
        state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
        transitions = np.full((len(MARKOV_STATES), len(MARKOV_STATES)), smoothing, dtype=np.float64)
        starts = np.full(len(MARKOV_STATES), smoothing, dtype=np.float64)
        for sequence in sequences:
            stages = collapsed_stages(sequence)
            if not stages:
                continue
            starts[state_index[stages[0]]] += 1.0
            for source, target in zip(stages, stages[1:]):
                transitions[state_index[source], state_index[target]] += 1.0
        transitions /= transitions.sum(axis=1, keepdims=True)
        starts /= starts.sum()
        return cls(transitions, starts, trained=True)

    def annotate(self, events: pd.DataFrame) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        previous_stage: str | None = None
        state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
        for _, row in events.iterrows():
            stage = event_stage(row["event_type"])
            probability = (
                self.start[state_index[stage]] if previous_stage is None
                else 1.0 if previous_stage == stage
                else self.transition[state_index[previous_stage], state_index[stage]]
            )
            output.append({
                "event_id": int(row["event_id"]),
                "timestamp": row["timestamp"].isoformat(),
                "event_type": str(row["event_type"]),
                "raw_event": str(row["raw_event"]),
                "stage": stage,
                "stage_label": _STATE_LABELS[stage],
                "user": str(row["user"]),
                "system": str(row["system"]),
                "transition_probability": float(probability),
                "transition_from": previous_stage,
            })
            previous_stage = stage
        return output

    def sequence_likelihood(self, events: pd.DataFrame) -> float:
        stages = collapsed_stages(events)
        if not stages:
            return 0.0
        state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
        log_probability = np.log(max(self.start[state_index[stages[0]]], 1e-12))
        for source, target in zip(stages, stages[1:]):
            log_probability += np.log(max(self.transition[state_index[source], state_index[target]], 1e-12))
        return float(np.exp(log_probability / len(stages)))

    def forecast_next(self, events: pd.DataFrame) -> list[dict[str, object]]:
        """Rank the possible next incident stage after the last observed event."""
        state_index = {name: index for index, name in enumerate(MARKOV_STATES)}
        last_stage = event_stage(events.iloc[-1]["event_type"]) if not events.empty else None
        probabilities = self.start if last_stage is None else self.transition[state_index[last_stage]]
        return [
            {"stage": stage, "stage_label": _STATE_LABELS[stage], "probability": float(probabilities[index])}
            for index, stage in sorted(enumerate(MARKOV_STATES), key=lambda item: probabilities[item[0]], reverse=True)[:3]
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"states": MARKOV_STATES, "transition": self.transition.tolist(),
                                      "start": self.start.tolist()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MarkovTimeline":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(artifact["states"]) != MARKOV_STATES:
            raise ValueError("Markov artifact uses a different stage vocabulary. Retrain the models.")
        return cls(np.asarray(artifact["transition"]), np.asarray(artifact["start"]), trained=True)
