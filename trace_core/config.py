"""Shared project configuration and event vocabulary."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = Path(os.environ.get("TRACE_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts"))).expanduser()

EVENT_TYPES = (
    "login_success",
    "login_failure",
    "privilege_change",
    "process_start",
    "external_connection",
    "data_transfer",
    "file_delete",
    "file_access",
    "account_change",
    "service_install",
    "dns_query",
    "firewall_block",
    "endpoint_alert",
    "password_reset",
    "scheduled_task",
    "lateral_movement",
    "malware_detected",
    "usb_mount",
    "logout",
    "other",
)

INCIDENT_CLASSES = (
    "Normal Activity",
    "Account Compromise",
    "Privilege Escalation",
    "Suspicious Process Execution",
    "Data Exfiltration",
    "Denial-of-Service-like Activity",
    "Multi-stage Attack",
)

MARKOV_STATES = (
    "normal",
    "initial_access",
    "privilege_change",
    "suspicious_execution",
    "external_communication",
    "data_transfer",
    "evidence_removal",
)

SEQUENCE_FEATURE_NAMES = tuple(EVENT_TYPES) + (
    "log_event_count",
    "log_data_volume",
    "unique_users",
    "unique_systems",
    "duration_minutes",
    "failed_login_ratio",
    "external_destination_count",
    "large_transfer_present",
    "privilege_change_present",
    "lateral_movement_present",
    "suspicious_process_present",
    "firewall_block_present",
    "failure_before_success",
    "privilege_before_process",
    "file_before_external_transfer",
    "attack_stage_diversity",
)

FEATURE_NAMES = tuple(EVENT_TYPES) + (
    "hour_sin",
    "hour_cos",
    "severity_scaled",
    "status_failure",
    "is_external",
    "is_large_transfer",
    "is_privileged",
    "is_suspicious_process",
    "is_evidence_removal",
    "bytes_log_scaled",
)

VAE_WINDOW_SIZE = 4
VAE_FEATURE_NAMES = tuple(
    f"lag_{lag}:{feature}"
    for lag in range(VAE_WINDOW_SIZE - 1, -1, -1)
    for feature in FEATURE_NAMES
)

RBM_FEATURE_NAMES = tuple(f"window:{event_type}" for event_type in EVENT_TYPES) + (
    "window:status_failure",
    "window:is_external",
    "window:is_large_transfer",
    "window:is_privileged",
    "window:is_suspicious_process",
    "window:is_evidence_removal",
)

ARTIFACT_FILES = {
    "vae": "vae_metadata.json",
    "vae_encoder": "vae_encoder.keras",
    "vae_decoder": "vae_decoder.keras",
    "rbm": "rbm.keras",
    "rbm_metadata": "rbm_metadata.json",
    "classifier": "incident_classifier.joblib",
    "markov": "markov.json",
    "summary": "training_summary.json",
}
