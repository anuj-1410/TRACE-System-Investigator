"""CSV normalization and deterministic features shared by training and analysis."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from trace_core.config import (
    EVENT_TYPES,
    FEATURE_NAMES,
    INCIDENT_CLASSES,
    RBM_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    VAE_FEATURE_NAMES,
    VAE_WINDOW_SIZE,
)

_COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date", "event_time", "ts"),
    "user": ("user", "username", "account", "principal", "actor", "user_name"),
    "system": ("system", "host", "hostname", "device", "machine", "computer", "endpoint"),
    "event_type": ("event_type", "event", "action", "activity", "event_name", "operation", "message", "description"),
    "source": ("source", "src", "source_ip", "src_ip", "source_address", "client_ip"),
    "destination": ("destination", "dest", "dst", "destination_ip", "dst_ip", "remote_ip", "target_ip"),
    "process": ("process", "process_name", "executable", "image", "command", "application"),
    "resource": ("resource", "file", "file_path", "object", "object_name", "target", "path"),
    "severity": ("severity", "risk", "level", "priority"),
    "status": ("status", "result", "outcome", "auth_result", "success"),
    "bytes_transferred": ("bytes_transferred", "bytes", "transfer_bytes", "data_size", "size_bytes", "sent_bytes"),
    "incident_id": ("incident_id", "case_id", "scenario_id", "session_id"),
    "incident_type": ("incident_type", "label", "attack_type", "class", "category", "ground_truth"),
}

_EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("login_failure", ("failed login", "login failed", "authentication failure", "auth failure", "invalid password", "logon failure")),
    ("login_success", ("login", "logon", "authentication", "sign in", "signin")),
    ("privilege_change", ("privilege", "elevation", "sudo", "admin role", "permission changed", "access right")),
    ("process_start", ("process", "execution", "command", "powershell", "script", "executable")),
    ("external_connection", ("external connection", "outbound connection", "remote connection", "network connection", "connect to", "c2", "beacon")),
    ("data_transfer", ("data transfer", "upload", "download", "exfil", "large transfer", "bytes sent")),
    ("file_delete", ("file delete", "file removed", "deleted file", "log cleared", "evidence removed")),
    ("file_access", ("file access", "file read", "file opened", "sensitive file", "document accessed")),
    ("account_change", ("account created", "account changed", "account disabled", "account enabled", "user added")),
    ("service_install", ("service installed", "service created", "driver installed")),
    ("dns_query", ("dns query", "domain lookup", "name resolution")),
    ("firewall_block", ("firewall block", "connection blocked", "blocked by firewall")),
    ("endpoint_alert", ("endpoint alert", "security alert", "edr alert", "antivirus alert")),
    ("password_reset", ("password reset", "credential reset", "password changed")),
    ("scheduled_task", ("scheduled task", "cron job", "task created")),
    ("lateral_movement", ("lateral movement", "remote execution", "smb connection", "remote service")),
    ("malware_detected", ("malware", "ransomware", "trojan", "malicious file")),
    ("usb_mount", ("usb", "removable media", "external drive")),
    ("logout", ("logout", "logoff", "sign out")),
)

_EVENT_SYNONYMS = {
    "auth_success": "login_success",
    "successful_login": "login_success",
    "successful_logon": "login_success",
    "auth_failure": "login_failure",
    "failed_authentication": "login_failure",
    "privilege_escalation": "privilege_change",
    "privilege_changed": "privilege_change",
    "process_execution": "process_start",
    "process_created": "process_start",
    "network_connection": "external_connection",
    "outbound_connection": "external_connection",
    "file_deletion": "file_delete",
    "data_exfiltration": "data_transfer",
    "large_data_transfer": "data_transfer",
}

_LABEL_ALIASES = {
    "normal": "Normal Activity",
    "normal activity": "Normal Activity",
    "benign": "Normal Activity",
    "account compromise": "Account Compromise",
    "account compromised": "Account Compromise",
    "credential compromise": "Account Compromise",
    "credential theft": "Account Compromise",
    "privilege escalation": "Privilege Escalation",
    "privilege abuse": "Privilege Escalation",
    "suspicious process execution": "Suspicious Process Execution",
    "malware": "Suspicious Process Execution",
    "malware execution": "Suspicious Process Execution",
    "data exfiltration": "Data Exfiltration",
    "exfiltration": "Data Exfiltration",
    "denial-of-service-like activity": "Denial-of-Service-like Activity",
    "denial of service": "Denial-of-Service-like Activity",
    "dos": "Denial-of-Service-like Activity",
    "ddos": "Denial-of-Service-like Activity",
    "multi-stage attack": "Multi-stage Attack",
    "multi stage attack": "Multi-stage Attack",
    "attack": "Multi-stage Attack",
}


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _find_column(frame: pd.DataFrame, canonical: str) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in _COLUMN_ALIASES[canonical]:
        if alias in lookup:
            return lookup[alias]
    return None


def canonical_event(value: object) -> str:
    """Map common log descriptions to the supported event vocabulary."""
    slug = _slug(value)
    if slug in EVENT_TYPES:
        return slug
    if slug in _EVENT_SYNONYMS:
        return _EVENT_SYNONYMS[slug]
    description = str(value).strip().lower()
    for event_type, phrases in _EVENT_PATTERNS:
        if any(phrase in description for phrase in phrases):
            return event_type
    return "other"


def normalize_incident_label(value: object) -> str | None:
    """Map common dataset labels to TRACE's seven supported categories."""
    if pd.isna(value):
        return None
    cleaned = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
    if cleaned in _LABEL_ALIASES:
        return _LABEL_ALIASES[cleaned]
    for label in INCIDENT_CLASSES:
        if cleaned == label.lower():
            return label
    return None


def normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, de-duplicated canonical event table from a CSV frame.

    The input must contain a recognizable event/action/message column. Unknown
    event descriptions are retained in ``raw_event`` and mapped to ``other``.
    """
    if frame.empty:
        raise ValueError("The uploaded CSV contains no event rows.")
    columns = {key: _find_column(frame, key) for key in _COLUMN_ALIASES}
    if columns["event_type"] is None:
        raise ValueError("Could not find an event column. Use event_type, event, action, activity, or message.")

    output = pd.DataFrame(index=frame.index)
    defaults = {
        "user": "unknown-user",
        "system": "unknown-system",
        "source": "",
        "destination": "",
        "process": "",
        "resource": "",
        "severity": "low",
        "status": "unknown",
        "bytes_transferred": 0,
        "incident_id": "",
        "incident_type": None,
    }
    raw_values = frame[columns["event_type"]].fillna("").astype(str)
    if "raw_event" in frame.columns:
        raw_descriptions = frame["raw_event"].fillna("").astype(str).str.strip()
        output["raw_event"] = raw_descriptions.where(raw_descriptions.ne(""), raw_values.str.strip()).replace("", "Unspecified event")
    else:
        output["raw_event"] = raw_values.str.strip().replace("", "Unspecified event")
    output["event_type"] = raw_values.map(canonical_event)

    for field, default in defaults.items():
        source_column = columns[field]
        if source_column is None:
            output[field] = default
        else:
            output[field] = frame[source_column].where(frame[source_column].notna(), default)
    for field in ("user", "system", "source", "destination", "process", "resource", "severity", "status", "incident_id"):
        output[field] = output[field].astype(str).str.strip().replace({"": defaults[field]})
    output["bytes_transferred"] = pd.to_numeric(output["bytes_transferred"], errors="coerce").fillna(0).clip(lower=0)
    output["incident_type"] = output["incident_type"].map(normalize_incident_label)

    timestamp_column = columns["timestamp"]
    if timestamp_column is None:
        parsed = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    else:
        parsed = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    fallback_start = pd.Timestamp("2026-01-01T00:00:00Z")
    if parsed.notna().any():
        fallback_start = parsed.dropna().min().floor("s")
    missing_positions = np.flatnonzero(parsed.isna().to_numpy())
    for offset, position in enumerate(missing_positions):
        parsed.iloc[position] = fallback_start + pd.Timedelta(seconds=int(position + offset))
    output["timestamp"] = parsed
    output["_input_order"] = np.arange(len(output), dtype=np.int64)

    dedupe_columns = [
        "incident_id", "timestamp", "user", "system", "raw_event", "event_type",
        "source", "destination", "process", "resource", "severity", "status", "bytes_transferred", "incident_type",
    ]
    output = output.drop_duplicates(subset=dedupe_columns, keep="first")
    output = output.sort_values(["timestamp", "_input_order"], kind="stable").reset_index(drop=True)
    output["event_id"] = np.arange(1, len(output) + 1, dtype=np.int64)
    return output


def event_feature_matrix(events: pd.DataFrame) -> np.ndarray:
    """Create bounded, deterministic per-event model features."""
    if events.empty:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    matrix = np.zeros((len(events), len(FEATURE_NAMES)), dtype=np.float32)
    event_index = {name: index for index, name in enumerate(EVENT_TYPES)}
    for row_number, event in events.reset_index(drop=True).iterrows():
        matrix[row_number, event_index.get(str(event["event_type"]), event_index["other"])] = 1.0
    base = len(EVENT_TYPES)
    timestamps = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    hour = timestamps.dt.hour.fillna(0).to_numpy(dtype=np.float32) + timestamps.dt.minute.fillna(0).to_numpy(dtype=np.float32) / 60.0
    matrix[:, base] = (np.sin(2.0 * math.pi * hour / 24.0) + 1.0) / 2.0
    matrix[:, base + 1] = (np.cos(2.0 * math.pi * hour / 24.0) + 1.0) / 2.0

    severity = events["severity"].astype(str).str.lower()
    matrix[:, base + 2] = severity.map({"low": 0.15, "informational": 0.05, "info": 0.05, "medium": 0.45, "moderate": 0.45, "high": 0.75, "critical": 1.0, "fatal": 1.0}).fillna(0.3).to_numpy(dtype=np.float32)
    status = events["status"].astype(str).str.lower()
    matrix[:, base + 3] = status.str.contains(r"fail|denied|blocked|error|false", regex=True).to_numpy(dtype=np.float32)
    matrix[:, base + 4] = events["event_type"].isin(("external_connection", "dns_query", "lateral_movement")).to_numpy(dtype=np.float32)
    bytes_count = pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    matrix[:, base + 5] = (bytes_count >= 10_000_000).astype(np.float32)
    matrix[:, base + 6] = (events["event_type"].eq("privilege_change") | events["event_type"].eq("account_change")).to_numpy(dtype=np.float32)
    process_text = events["process"].astype(str).str.lower()
    matrix[:, base + 7] = (events["event_type"].isin(("malware_detected", "service_install", "scheduled_task")) | process_text.str.contains(r"powershell|wscript|cscript|rundll32|encoded|unknown", regex=True)).to_numpy(dtype=np.float32)
    matrix[:, base + 8] = events["event_type"].eq("file_delete").to_numpy(dtype=np.float32)
    matrix[:, base + 9] = np.clip(np.log1p(bytes_count) / np.log1p(1_000_000_000), 0.0, 1.0)
    return matrix


def vae_feature_matrix(events: pd.DataFrame) -> np.ndarray:
    """Encode each event with its preceding events in chronological order.

    Windows are zero padded at case boundaries. A nonempty incident_id keeps
    separate cases from sharing temporal context when a CSV contains several.
    """
    base = event_feature_matrix(events)
    result = np.zeros((len(events), len(VAE_FEATURE_NAMES)), dtype=np.float32)
    if len(events) == 0:
        return result
    case_ids = events["incident_id"].fillna("").astype(str).to_numpy() if "incident_id" in events else np.full(len(events), "", dtype=str)
    width = base.shape[1]
    for case_id in np.unique(case_ids):
        positions = np.flatnonzero(case_ids == case_id)
        case_features = base[positions]
        for lag in range(VAE_WINDOW_SIZE):
            if lag >= len(positions):
                break
            start = (VAE_WINDOW_SIZE - 1 - lag) * width
            result[positions[lag:], start : start + width] = case_features[: len(positions) - lag]
    return result


def rbm_feature_matrix(events: pd.DataFrame) -> np.ndarray:
    """Create Bernoulli co-occurrence features over recent events in each case."""
    continuous = event_feature_matrix(events)
    event_width = len(EVENT_TYPES)
    event_bits = continuous[:, :event_width]
    indicators = continuous[:, event_width + 3 : event_width + 9]
    per_event = np.concatenate((event_bits, indicators), axis=1).astype(np.float32)
    result = np.zeros((len(events), len(RBM_FEATURE_NAMES)), dtype=np.float32)
    if len(events) == 0:
        return result
    case_ids = events["incident_id"].fillna("").astype(str).to_numpy() if "incident_id" in events else np.full(len(events), "", dtype=str)
    for case_id in np.unique(case_ids):
        positions = np.flatnonzero(case_ids == case_id)
        case_features = per_event[positions]
        for lag in range(min(VAE_WINDOW_SIZE, len(positions))):
            result[positions[lag:]] = np.maximum(result[positions[lag:]], case_features[: len(positions) - lag])
    return result


def sequence_feature_vector(events: pd.DataFrame) -> np.ndarray:
    """Aggregate one incident's events into a fixed-length classifier vector."""
    if events.empty:
        return np.zeros(len(SEQUENCE_FEATURE_NAMES), dtype=np.float32)
    counts = events["event_type"].value_counts()
    total = max(len(events), 1)
    vector = [float(counts.get(event_type, 0)) / total for event_type in EVENT_TYPES]
    transferred = pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0).sum()
    timestamps = pd.to_datetime(events["timestamp"], utc=True, errors="coerce").dropna()
    duration = (timestamps.max() - timestamps.min()).total_seconds() / 60.0 if len(timestamps) > 1 else 0.0
    failed_ratio = float(events["event_type"].eq("login_failure").sum()) / total
    ordered_types = events["event_type"].astype(str).tolist()
    positions = {kind: [index for index, value in enumerate(ordered_types) if value == kind]
                 for kind in ("login_failure", "login_success", "privilege_change", "process_start",
                              "file_access", "external_connection", "data_transfer")}
    failure_before_success = bool(positions["login_failure"] and positions["login_success"]
                                  and min(positions["login_failure"]) < max(positions["login_success"]))
    privilege_before_process = bool(positions["privilege_change"] and positions["process_start"]
                                    and min(positions["privilege_change"]) < max(positions["process_start"]))
    transfer_destinations = events.loc[events["event_type"].eq("data_transfer"), "destination"].astype(str)
    external_transfer = transfer_destinations.str.strip().ne("") & ~transfer_destinations.str.match(
        r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")
    file_before_external_transfer = bool(positions["file_access"] and positions["data_transfer"]
                                         and min(positions["file_access"]) < max(positions["data_transfer"])
                                         and external_transfer.any())
    stage_groups = (
        bool(set(ordered_types) & {"login_failure", "login_success"}),
        bool(set(ordered_types) & {"account_change", "privilege_change", "password_reset"}),
        bool(set(ordered_types) & {"process_start", "service_install", "scheduled_task", "malware_detected"}),
        bool(set(ordered_types) & {"external_connection", "dns_query", "lateral_movement"}),
        bool(set(ordered_types) & {"data_transfer", "file_delete"}),
    )
    destinations = events.loc[events["destination"].astype(str).str.strip().ne(""), "destination"].nunique()
    vector.extend((
        float(min(np.log1p(total) / np.log1p(10_000), 1.0)),
        float(np.clip(np.log1p(transferred) / np.log1p(1_000_000_000), 0.0, 1.0)),
        float(min(events["user"].nunique(), 10) / 10.0),
        float(min(events["system"].nunique(), 10) / 10.0),
        float(min(np.log1p(max(duration, 0.0)) / np.log1p(1440), 1.0)),
        failed_ratio,
        float(min(destinations, 10) / 10.0),
        float(pd.to_numeric(events["bytes_transferred"], errors="coerce").fillna(0).ge(10_000_000).any()),
        float(events["event_type"].isin(("privilege_change", "account_change")).any()),
        float(events["event_type"].eq("lateral_movement").any()),
        float(events["event_type"].isin(("malware_detected", "endpoint_alert", "service_install", "scheduled_task")).any()
              or events["process"].astype(str).str.contains(r"powershell|wscript|cscript|rundll32|encoded|mimikatz", case=False, regex=True).any()),
        float(events["event_type"].eq("firewall_block").any()),
        float(failure_before_success),
        float(privilege_before_process),
        float(file_before_external_transfer),
        float(sum(stage_groups) / len(stage_groups)),
    ))
    return np.asarray(vector, dtype=np.float32)


def labeled_incident_groups(events: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return incident sequences where both incident id and supported label exist."""
    if "incident_id" not in events or "incident_type" not in events:
        return []
    valid = events[events["incident_id"].astype(str).str.strip().ne("") & events["incident_type"].notna()]
    groups: list[tuple[str, pd.DataFrame]] = []
    for incident_id, group in valid.groupby("incident_id", sort=False):
        labels = group["incident_type"].dropna().unique()
        if len(labels) == 1:
            groups.append((str(labels[0]), group.reset_index(drop=True)))
    return groups


def feature_names() -> Iterable[str]:
    return FEATURE_NAMES
