"""Conservative entity, exposure, severity, and response estimates."""

from __future__ import annotations

import pandas as pd


def assess_impact(events: pd.DataFrame, anomaly_score: float, incident_probability: float, evidence: dict[str, bool]) -> dict[str, object]:
    transferred = float(pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0).sum())
    external = events[events["event_type"].eq("external_connection")]
    destinations = events[events["event_type"].isin(("external_connection", "data_transfer"))]
    processes = events[events["process"].astype(str).str.strip().ne("")]["process"].astype(str).unique().tolist()
    suspicious = events[
        events["event_type"].isin(("malware_detected", "endpoint_alert", "service_install", "scheduled_task"))
        | events["process"].astype(str).str.contains(r"powershell|wscript|cscript|rundll32|encoded|unknown", case=False, regex=True)
        | (events["event_type"].eq("data_transfer") & pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0).ge(10_000_000))
    ]
    suspicious_processes = suspicious.loc[suspicious["process"].astype(str).str.strip().ne(""), "process"].astype(str).unique().tolist()
    resources = events[events["resource"].astype(str).str.strip().ne("")]["resource"].astype(str).unique().tolist()
    users = sorted(set(events.loc[events["user"].astype(str).str.strip().ne("unknown-user"), "user"].astype(str)))
    systems = sorted(set(events.loc[events["system"].astype(str).str.strip().ne("unknown-system"), "system"].astype(str)))
    if transferred >= 100_000_000 or (evidence.get("large_transfer") and evidence.get("external_communication")):
        exposure = "High"
    elif transferred > 0 or evidence.get("large_transfer") or (evidence.get("external_communication") and events["event_type"].eq("file_access").any()):
        exposure = "Medium"
    else:
        exposure = "Low"
    breadth = min((len(users) + len(systems) + len(processes) / 2 + len(external) / 2) / 12.0, 1.0)
    severity_score = 100.0 * (0.40 * anomaly_score + 0.30 * incident_probability + 0.15 * breadth + 0.15 * (1.0 if exposure == "High" else 0.55 if exposure == "Medium" else 0.15))
    severity = "Critical" if severity_score >= 80 else "High" if severity_score >= 62 else "Medium" if severity_score >= 38 else "Low"
    return {
        "affected_users": users,
        "affected_systems": systems,
        "suspicious_processes": suspicious_processes,
        "observed_processes": processes,
        "external_destinations": sorted(set(destinations["destination"].astype(str).str.strip()) - {""}),
        "affected_resources": resources,
        "events_with_external_communication": int(len(external)),
        "transferred_bytes": int(transferred),
        "potential_data_exposure": exposure,
        "severity": severity,
        "severity_score": round(float(severity_score), 1),
    }


def recommendations(
    events: pd.DataFrame,
    evidence: dict[str, bool],
    incident_type: str,
    evidence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select fixed investigator guidance and link each trigger to observed events."""
    actions: list[dict[str, object]] = []
    event_ids_by_key = {str(row["key"]): [int(value) for value in row["event_ids"]] for row in evidence_rows}
    evidence_labels = {
        "unusual_authentication": "an unusual sign-in",
        "privilege_change": "an account access change",
        "suspicious_process": "an unusual program or alert",
        "external_communication": "a network connection",
        "large_transfer": "a large data transfer",
        "evidence_removal": "a deleted file",
        "repeated_auth_failures": "repeated failed sign-ins",
    }

    def add(priority: str, action: str, keys: tuple[str, ...] = (), *, reason: str | None = None, event_ids: list[int] | None = None) -> None:
        active = [key for key in keys if evidence.get(key)]
        linked = event_ids if event_ids is not None else sorted({event_id for key in active for event_id in event_ids_by_key.get(key, [])})[:8]
        explanation = reason or "Suggested because these records show " + ", ".join(evidence_labels[key] for key in active) + "."
        actions.append({"priority": priority, "action": action, "reason": explanation, "event_ids": linked, "trigger_keys": active})

    if incident_type == "Normal Activity":
        add("Low", "Keep these records in case other information later points to a problem.", reason="This case did not show a clear warning.", event_ids=[])
        add("Medium", "Check nearby activity for the same people and devices if you still have concerns.", reason="Other records may add context to this case.", event_ids=[int(value) for value in events["event_id"].head(8)])
        return actions

    if evidence.get("unusual_authentication") or evidence.get("privilege_change"):
        add("High", "Check whether the account activity was expected. If it was not, ask your security team to sign out active sessions and secure the account.", ("unusual_authentication", "privilege_change"))
    if evidence.get("suspicious_process") or evidence.get("external_communication"):
        add("High", "Ask your security team to review the device and, if needed, disconnect it using your organization's response process.", ("suspicious_process", "external_communication"))
    if evidence.get("external_communication"):
        add("High", "Check whether the outside address was expected and review nearby network activity.", ("external_communication",))
    if evidence.get("large_transfer"):
        add("High", "Find out what data moved and whether it was sent to an approved destination.", ("large_transfer",))
    if evidence.get("evidence_removal"):
        add("High", "Save the available activity records before anyone cleans up or resets the device.", ("evidence_removal",))
    add("Medium", "Compare this case with nearby activity for the same people and devices.", reason="Other records may help explain what happened.", event_ids=[int(value) for value in events["event_id"].head(8)])
    if incident_type == "Denial-of-Service-like Activity":
        related = events.loc[events["event_type"].isin(("login_failure", "firewall_block", "external_connection")), "event_id"].head(8)
        add("Medium", "Check whether the service was unavailable and whether it received an unusual number of requests.", ("repeated_auth_failures", "external_communication"), reason="This case includes repeated sign-in or network activity.", event_ids=[int(value) for value in related])
    return actions
