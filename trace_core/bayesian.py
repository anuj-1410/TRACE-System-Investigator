"""Exact evidence inference in a small categorical-root Bayesian network."""

from __future__ import annotations

import numpy as np
import pandas as pd

ROOT_CAUSES = ("Compromised account", "Insider activity", "Misconfiguration", "Other / unknown")

# Each evidence flag is a child node of the mutually exclusive RootCause node.
# Values are P(evidence=True | root cause), elicited for this academic prototype.
_PRIORS = np.asarray([0.36, 0.20, 0.22, 0.22], dtype=np.float64)
_EVIDENCE_CPTS = {
    "unusual_authentication": np.asarray([0.76, 0.20, 0.13, 0.16]),
    "privilege_change": np.asarray([0.68, 0.55, 0.22, 0.20]),
    "suspicious_process": np.asarray([0.64, 0.42, 0.22, 0.25]),
    "external_communication": np.asarray([0.62, 0.37, 0.18, 0.25]),
    "large_transfer": np.asarray([0.58, 0.52, 0.18, 0.21]),
    "evidence_removal": np.asarray([0.49, 0.34, 0.12, 0.17]),
    "repeated_auth_failures": np.asarray([0.54, 0.10, 0.16, 0.21]),
}


def collect_evidence(events: pd.DataFrame) -> dict[str, bool]:
    event_types = events["event_type"].astype(str)
    raw_text = events["raw_event"].astype(str).str.lower()
    process = events["process"].astype(str).str.lower()
    severity = events["severity"].astype(str).str.lower()
    login_failures = int(event_types.eq("login_failure").sum())
    login_successes = events[event_types.eq("login_success")]
    success_at_night = False
    if not login_successes.empty:
        hours = pd.to_datetime(login_successes["timestamp"], utc=True, errors="coerce").dt.hour
        success_at_night = bool(hours.isin([0, 1, 2, 3, 4, 5, 23]).any())
    unusual_authentication = bool(
        (login_failures >= 2 and not login_successes.empty)
        or (success_at_night and severity[event_types.eq("login_success")].isin(["high", "critical"]).any())
        or raw_text.str.contains(r"unusual|new location|impossible travel|unknown device", regex=True).any()
    )
    process_indicators = process.str.contains(r"powershell|wscript|cscript|rundll32|encoded|unknown|mimikatz|temp\\", regex=True)
    suspicious_process = bool(
        event_types.isin(["malware_detected", "service_install", "scheduled_task", "endpoint_alert"]).any()
        or process_indicators.any()
        or raw_text.str.contains(r"suspicious process|malware|payload|encoded command", regex=True).any()
    )
    bytes_count = pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0)
    large_transfer = bool(
        (bytes_count >= 10_000_000).any()
        or (event_types.eq("data_transfer") & raw_text.str.contains(r"large|exfil|bulk|upload", regex=True)).any()
    )
    return {
        "unusual_authentication": unusual_authentication,
        "privilege_change": bool(event_types.isin(["privilege_change", "account_change"]).any()),
        "suspicious_process": suspicious_process,
        "external_communication": bool(event_types.isin(["external_connection", "dns_query", "lateral_movement"]).any()),
        "large_transfer": large_transfer,
        "evidence_removal": bool(event_types.eq("file_delete").any()),
        "repeated_auth_failures": login_failures >= 3,
    }


def infer_root_cause(evidence: dict[str, bool]) -> list[dict[str, object]]:
    """Return posterior root-cause probabilities by exact enumeration.

    Only observed positive flags affect the posterior. A missing event is not
    treated as proof that an activity did not occur in an incomplete log feed.
    """
    log_posterior = np.log(_PRIORS)
    observed = [key for key, present in evidence.items() if present and key in _EVIDENCE_CPTS]
    for key in observed:
        likelihood = np.clip(_EVIDENCE_CPTS[key], 1e-5, 1.0 - 1e-5)
        log_posterior += np.log(likelihood)
    log_posterior -= np.max(log_posterior)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    ranked = sorted(zip(ROOT_CAUSES, posterior), key=lambda item: item[1], reverse=True)
    return [{"cause": cause, "probability": float(probability)} for cause, probability in ranked]


def evidence_explanations(events: pd.DataFrame, evidence: dict[str, bool]) -> list[dict[str, object]]:
    descriptors = {
        "unusual_authentication": ("An unusual sign-in", "The records show failed sign-ins followed by success, a flagged sign-in, or a high-severity sign-in at an unusual hour."),
        "privilege_change": ("Account access changed", "An account or its permissions changed in the records."),
        "suspicious_process": ("An unusual program or alert", "A program, scheduled task, installed service, or security alert appeared in the records."),
        "external_communication": ("A network connection", "A device contacted an outside address or another system."),
        "large_transfer": ("A large data transfer", "The records show at least 10 MB transferred or explicitly describe a large upload."),
        "evidence_removal": ("A file was deleted", "A file deletion appears in the records. Its purpose is not known."),
        "repeated_auth_failures": ("Repeated failed sign-ins", "At least three sign-in attempts failed."),
    }
    results = []
    for key, present in evidence.items():
        if not present:
            continue
        title, detail = descriptors[key]
        matching = events
        if key in ("unusual_authentication", "repeated_auth_failures"):
            matching = events[events["event_type"].isin(["login_failure", "login_success"])]
        elif key == "privilege_change":
            matching = events[events["event_type"].isin(["privilege_change", "account_change"])]
        elif key == "suspicious_process":
            matching = events[events["event_type"].isin(["process_start", "service_install", "scheduled_task", "endpoint_alert", "malware_detected"])]
        elif key == "external_communication":
            matching = events[events["event_type"].isin(["external_connection", "dns_query", "lateral_movement"])]
        elif key == "large_transfer":
            matching = events[(events["event_type"].eq("data_transfer")) | (pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0) >= 10_000_000)]
        elif key == "evidence_removal":
            matching = events[events["event_type"].eq("file_delete")]
        results.append({
            "key": key,
            "title": title,
            "detail": detail,
            "event_ids": [int(value) for value in matching["event_id"].head(8)],
            "count": int(len(matching)),
        })
    return results
