"""Deterministic, imbalanced case simulation with overlapping event patterns.

The data resemble structured audit logs but are not captured telemetry.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from trace_core.config import INCIDENT_CLASSES

DEFAULT_CLASS_COUNTS = {
    "Normal Activity": 1500,
    "Account Compromise": 410,
    "Privilege Escalation": 315,
    "Suspicious Process Execution": 260,
    "Data Exfiltration": 215,
    "Denial-of-Service-like Activity": 165,
    "Multi-stage Attack": 120,
}

ATTACK_PATHS = {
    "Account Compromise": (
        ("login_failure", "login_failure", "login_success", "account_change"),
        ("login_failure", "login_success", "password_reset", "external_connection"),
        ("login_success", "account_change", "file_access"),
        ("login_failure", "login_failure", "login_success", "file_access"),
        ("login_success", "password_reset", "external_connection"),
    ),
    "Privilege Escalation": (
        ("login_success", "account_change", "privilege_change", "process_start"),
        ("login_success", "privilege_change", "service_install"),
        ("login_success", "privilege_change", "scheduled_task"),
        ("account_change", "privilege_change", "file_access"),
        ("login_failure", "login_success", "privilege_change", "process_start"),
    ),
    "Suspicious Process Execution": (
        ("file_access", "process_start", "process_start", "endpoint_alert"),
        ("login_success", "process_start", "scheduled_task"),
        ("process_start", "external_connection", "malware_detected"),
        ("file_access", "process_start", "service_install"),
        ("login_success", "process_start", "file_delete", "endpoint_alert"),
    ),
    "Data Exfiltration": (
        ("file_access", "file_access", "external_connection", "data_transfer"),
        ("login_success", "file_access", "data_transfer", "file_delete"),
        ("usb_mount", "file_access", "data_transfer"),
        ("file_access", "dns_query", "external_connection", "data_transfer"),
        ("file_access", "process_start", "data_transfer"),
    ),
    "Denial-of-Service-like Activity": (
        ("external_connection", "external_connection", "firewall_block"),
        ("login_failure", "login_failure", "login_failure", "firewall_block"),
        ("dns_query", "external_connection", "external_connection", "firewall_block"),
        ("external_connection", "firewall_block", "endpoint_alert"),
        ("login_failure", "external_connection", "firewall_block"),
    ),
    "Multi-stage Attack": (
        ("login_failure", "login_success", "privilege_change", "process_start", "external_connection", "data_transfer"),
        ("login_success", "account_change", "lateral_movement", "file_access", "data_transfer"),
        ("file_access", "process_start", "service_install", "lateral_movement", "file_delete"),
        ("login_success", "privilege_change", "scheduled_task", "external_connection", "file_delete"),
        ("login_success", "process_start", "lateral_movement", "data_transfer", "file_delete"),
    ),
}

ROUTINE = ("login_success", "file_access", "process_start", "dns_query", "external_connection",
           "file_access", "logout", "data_transfer", "login_failure", "other")
BENIGN_MIMICS = ("account_change", "privilege_change", "scheduled_task", "firewall_block",
                 "endpoint_alert", "usb_mount", "password_reset", "service_install")
NOISE = ROUTINE + BENIGN_MIMICS + ("process_start", "file_access", "external_connection")
LOG_SOURCES = {
    "login_success": "Windows Security", "login_failure": "Identity provider",
    "logout": "Windows Security", "file_access": "File audit",
    "process_start": "Sysmon", "dns_query": "DNS resolver",
    "external_connection": "Network proxy", "data_transfer": "Network proxy",
    "account_change": "Identity provider", "privilege_change": "Windows Security",
    "password_reset": "Identity provider", "scheduled_task": "Task Scheduler",
    "service_install": "Service Control Manager", "endpoint_alert": "Endpoint protection",
    "malware_detected": "Endpoint protection", "firewall_block": "Edge firewall",
    "lateral_movement": "Windows Security", "file_delete": "File audit",
    "usb_mount": "Device control", "other": "System audit",
}
DESCRIPTIONS = {
    "login_success": ("Account sign-in succeeded", "Interactive sign-in completed", "Service sign-in recorded"),
    "login_failure": ("Account sign-in failed", "Password validation failed", "Remote sign-in was rejected"),
    "logout": ("Account session ended", "User signed out"),
    "file_access": ("File opened on shared storage", "Document read by application", "File access recorded"),
    "process_start": ("Process creation recorded", "Application started", "Command shell launched"),
    "dns_query": ("DNS lookup completed", "Application resolved a service name"),
    "external_connection": ("Network connection observed", "Proxy recorded a connection"),
    "data_transfer": ("Data transfer recorded", "File copy completed", "Network upload observed"),
    "privilege_change": ("Account rights changed", "Administrator role assignment recorded"),
    "account_change": ("Account setting updated", "Group membership changed"),
    "password_reset": ("Account password reset", "Password change completed"),
    "scheduled_task": ("Scheduled task registered", "Maintenance task created"),
    "service_install": ("System service installed", "Background service registered"),
    "endpoint_alert": ("Endpoint alert created", "Endpoint protection raised an alert"),
    "malware_detected": ("Endpoint protection flagged a file",),
    "firewall_block": ("Firewall blocked a connection", "Rate limit applied"),
    "lateral_movement": ("Remote service connection recorded",),
    "file_delete": ("Temporary file deleted",),
    "usb_mount": ("Removable device attached",),
    "other": ("System audit event recorded",),
}


def create_training_cases(counts: dict[str, int] | None = None, seed: int = 41) -> list[tuple[str, pd.DataFrame]]:
    """Create case-level labels with varied lengths, context, and ambiguous cues."""
    counts = DEFAULT_CLASS_COUNTS if counts is None else counts
    if set(counts) != set(INCIDENT_CLASSES) or min(counts.values()) < 20:
        raise ValueError("Give at least 20 cases for each of the seven categories.")
    rng = np.random.default_rng(seed)
    output: list[tuple[str, pd.DataFrame]] = []
    case_number = 0
    for label in INCIDENT_CLASSES:
        for _ in range(counts[label]):
            case_number += 1
            is_attack = label != "Normal Activity"
            if is_attack:
                path = list(ATTACK_PATHS[label][int(rng.integers(len(ATTACK_PATHS[label])))])
                if len(path) > 3 and rng.random() < .32:
                    path.pop(int(rng.integers(1, len(path) - 1)))
                core = [(kind, True) for kind in path]
            else:
                core = [("login_success", False)]
                if rng.random() < .38:
                    core.append((str(rng.choice(BENIGN_MIMICS)), False))
            context_count = (
                int(rng.integers(1, 5)) if is_attack else int(rng.integers(3, 8))
            ) if rng.random() < .30 else int(rng.integers(5, 19))
            context = [(str(rng.choice(NOISE if is_attack else ROUTINE)), False)
                       for _ in range(context_count)]
            if not is_attack and rng.random() < .15:
                context.append((str(rng.choice(BENIGN_MIMICS)), False))
            if is_attack and rng.random() < .32:
                context.append((str(rng.choice(tuple(kind for other, paths in ATTACK_PATHS.items()
                                                    if other != label for path in paths for kind in path))), False))
            insertion = sorted(rng.choice(np.arange(len(context) + 1), size=len(core), replace=True).tolist())
            events = context
            for position, event in reversed(list(zip(insertion, core))):
                events.insert(position, event)
            user = f"user{int(rng.integers(1, 4500)):05d}"
            host = f"{str(rng.choice(('WS', 'SRV', 'LAP', 'VDI')))}-{int(rng.integers(1, 950)):04d}"
            internal = f"10.{int(rng.integers(12, 35))}.{int(rng.integers(1, 250))}.{int(rng.integers(2, 250))}"
            external = f"{str(rng.choice(('192.0.2', '198.51.100', '203.0.113')))}.{int(rng.integers(1, 255))}"
            hour = int(rng.integers(0, 24)) if is_attack or rng.random() < .18 else int(rng.integers(7, 19))
            start = datetime(2026, int(rng.integers(1, 10)), int(rng.integers(1, 28)), hour,
                             int(rng.integers(0, 60)), tzinfo=timezone.utc)
            elapsed = 0
            records = []
            for event_id, (kind, core_event) in enumerate(events, 1):
                elapsed += int(rng.integers(0, 3 if label == "Denial-of-Service-like Activity" else 15))
                if core_event and rng.random() < .45:
                    severity = str(rng.choice(("low", "medium", "high")))
                elif core_event:
                    severity = str(rng.choice(("medium", "high", "critical")))
                else:
                    severity = str(rng.choice(("low", "low", "medium", "high")))
                status = "failure" if kind == "login_failure" else "blocked" if kind == "firewall_block" else "alert" if kind in ("endpoint_alert", "malware_detected") else "success"
                transfer = 0
                if kind == "data_transfer":
                    transfer = int(rng.lognormal(np.log(28_000_000 if core_event else 220_000), 1.4))
                    transfer = min(transfer, 950_000_000)
                process = ""
                if kind in ("process_start", "scheduled_task", "service_install", "malware_detected"):
                    risky = core_event and rng.random() < .67 or not core_event and rng.random() < .12
                    process = str(rng.choice(("powershell.exe", "wscript.exe", "rundll32.exe") if risky
                                             else ("msedge.exe", "excel.exe", "svchost.exe", "python.exe", "cmd.exe")))
                outside = kind in ("external_connection", "dns_query") and (core_event or rng.random() < .48)
                if kind == "data_transfer":
                    outside = core_event and rng.random() < .83 or not core_event and rng.random() < .07
                destination = external if outside else f"10.24.{int(rng.integers(1, 80))}.{int(rng.integers(2, 250))}" if kind in ("external_connection", "data_transfer", "lateral_movement") else ""
                resource = ""
                if kind in ("file_access", "data_transfer", "file_delete"):
                    resource = str(rng.choice(("finance-q3.xlsx", "customer-export.csv", "team-notes.docx",
                                               "service.log", "sales-archive.zip", "policy.pdf")))
                row = {
                    "timestamp": pd.Timestamp(start + timedelta(minutes=elapsed, seconds=int(rng.integers(0, 60)))),
                    "incident_id": f"SIM-{case_number:06d}", "incident_type": label,
                    "user": user, "system": host, "event_type": kind, "source": internal,
                    "destination": destination, "process": process, "resource": resource,
                    "severity": severity, "status": status, "bytes_transferred": transfer,
                    "log_source": LOG_SOURCES[kind], "source_record_id": f"{case_number}-{event_id}",
                    "raw_event": str(rng.choice(DESCRIPTIONS[kind])),
                    "event_id": event_id,
                }
                records.append(row)
            output.append((label, pd.DataFrame.from_records(records)))
    rng.shuffle(output)
    return output
