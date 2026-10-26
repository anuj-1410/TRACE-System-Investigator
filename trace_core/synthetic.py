"""Small, labeled scenarios for a reproducible academic demonstration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

_SCENARIOS: dict[str, tuple[tuple[str, str, str, str, str, int], ...]] = {
    "Normal Activity": (
        ("login_success", "Successful sign-in", "browser.exe", "low", "success", 0),
        ("file_access", "Routine project file opened", "editor.exe", "low", "success", 0),
        ("process_start", "Approved office application started", "office.exe", "low", "success", 0),
        ("logout", "User signed out", "", "low", "success", 0),
    ),
    "Account Compromise": (
        ("login_failure", "Failed authentication attempt", "", "medium", "failure", 0),
        ("login_failure", "Failed authentication attempt", "", "medium", "failure", 0),
        ("login_success", "Unusual successful login from new location", "browser.exe", "critical", "success", 0),
        ("external_connection", "Outbound connection to unfamiliar address", "browser.exe", "high", "success", 0),
    ),
    "Privilege Escalation": (
        ("login_success", "Successful sign-in", "terminal.exe", "medium", "success", 0),
        ("privilege_change", "User privilege changed to administrator", "", "critical", "success", 0),
        ("process_start", "Elevated command shell started", "powershell.exe", "high", "success", 0),
        ("account_change", "Privileged account setting changed", "", "high", "success", 0),
    ),
    "Suspicious Process Execution": (
        ("login_success", "Successful sign-in", "browser.exe", "low", "success", 0),
        ("process_start", "Encoded PowerShell command started", "powershell.exe", "critical", "success", 0),
        ("scheduled_task", "Unrecognized scheduled task created", "taskhost.exe", "high", "success", 0),
        ("endpoint_alert", "Endpoint flagged suspicious payload", "payload.exe", "critical", "alert", 0),
    ),
    "Data Exfiltration": (
        ("login_success", "Successful sign-in", "browser.exe", "low", "success", 0),
        ("file_access", "Sensitive records opened in bulk", "archive.exe", "high", "success", 0),
        ("external_connection", "Outbound connection to unfamiliar address", "archive.exe", "high", "success", 0),
        ("data_transfer", "Large data upload to external destination", "archive.exe", "critical", "success", 52_000_000),
        ("file_delete", "Temporary archive deleted after transfer", "archive.exe", "high", "success", 0),
    ),
    "Denial-of-Service-like Activity": (
        ("login_failure", "Repeated authentication failure", "", "medium", "failure", 0),
        ("login_failure", "Repeated authentication failure", "", "medium", "failure", 0),
        ("login_failure", "Repeated authentication failure", "", "medium", "failure", 0),
        ("login_failure", "Repeated authentication failure", "", "high", "failure", 0),
        ("external_connection", "High-rate inbound connection burst", "", "high", "blocked", 0),
        ("firewall_block", "Firewall blocked connection burst", "", "high", "blocked", 0),
    ),
    "Multi-stage Attack": (
        ("login_failure", "Failed authentication attempt", "", "medium", "failure", 0),
        ("login_success", "Unusual successful login from new location", "browser.exe", "critical", "success", 0),
        ("privilege_change", "User privilege changed to administrator", "", "critical", "success", 0),
        ("process_start", "Encoded PowerShell command started", "powershell.exe", "critical", "success", 0),
        ("external_connection", "Outbound connection to unfamiliar address", "powershell.exe", "high", "success", 0),
        ("data_transfer", "Large data upload to external destination", "archive.exe", "critical", "success", 34_000_000),
        ("file_delete", "Temporary archive deleted after transfer", "archive.exe", "high", "success", 0),
    ),
}


def scenario_records(label: str, scenario_id: str, seed: int = 0) -> pd.DataFrame:
    """Build one scenario in the already-normalized internal event schema."""
    if label not in _SCENARIOS:
        raise ValueError(f"Unsupported synthetic label: {label}")
    rng = np.random.default_rng(seed)
    base = datetime(2026, 4, 15, 9, 0, tzinfo=timezone.utc)
    if label in ("Account Compromise", "Multi-stage Attack"):
        base = datetime(2026, 4, 15, 1, 30, tzinfo=timezone.utc)
    system = f"WS-{rng.integers(1, 9):02d}"
    user = f"user{rng.integers(1, 15):03d}"
    destination = f"198.51.100.{rng.integers(10, 240)}"
    records = []
    for index, (event_type, raw_event, process, severity, status, transferred) in enumerate(_SCENARIOS[label]):
        jitter = int(rng.integers(0, 2)) if index else 0
        records.append({
            "timestamp": pd.Timestamp(base + timedelta(minutes=index * 3 + jitter)),
            "user": user,
            "system": system,
            "event_type": event_type,
            "source": f"10.0.{rng.integers(1, 20)}.{rng.integers(2, 240)}",
            "destination": destination if event_type in ("external_connection", "data_transfer") else "",
            "process": process,
            "resource": "customer_records.zip" if event_type in ("file_access", "data_transfer", "file_delete") else "",
            "severity": severity,
            "status": status,
            "bytes_transferred": int(transferred * rng.uniform(0.85, 1.15)) if transferred else 0,
            "incident_id": scenario_id,
            "incident_type": label,
            "raw_event": raw_event,
            "_input_order": index,
            "event_id": index + 1,
        })
    return pd.DataFrame.from_records(records)
