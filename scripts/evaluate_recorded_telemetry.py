"""Evaluate saved models on two independent OTRF recorded lab captures.

These are attack emulations recorded on lab hosts, not enterprise incidents.
The source tactic is only a coarse case label. This is a two-case stress check,
not a seven-class accuracy estimate.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_core.config import ARTIFACT_DIR, ARTIFACT_FILES
from trace_core.pipeline import analyze_events, load_models
from trace_core.preprocessing import normalize_events

SOURCES = (
    ("Privilege Escalation", "cmd_service_mod_fax.zip",
     "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/privilege_escalation/host/cmd_service_mod_fax.zip"),
    ("Suspicious Process Execution", "empire_launcher_vbs.zip",
     "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/execution/host/empire_launcher_vbs.zip"),
)

EVENT_MAP = {
    4624: "login_success", 4625: "login_failure", 4634: "logout",
    4672: "privilege_change", 4688: "process_start", 4697: "service_install",
    4663: "file_access", 7045: "service_install",
}
SYSMON_MAP = {1: "process_start", 3: "external_connection", 11: "file_access",
              22: "dns_query", 23: "file_delete", 26: "file_delete"}


def convert(payload: bytes, case_id: str) -> tuple[pd.DataFrame, int]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError("Expected one JSON-lines capture per archive.")
        source_rows = [json.loads(line) for line in archive.read(members[0]).splitlines() if line.strip()]
    records = []
    for index, item in enumerate(source_rows, 1):
        event_id = int(item.get("EventID", -1))
        source_name = str(item.get("SourceName", ""))
        kind = SYSMON_MAP.get(event_id) if "Sysmon" in source_name else EVENT_MAP.get(event_id)
        if kind is None:
            continue
        process = str(item.get("Image") or item.get("NewProcessName") or item.get("ProcessName") or "")
        records.append({
            "timestamp": item.get("@timestamp") or item.get("TimeCreated"),
            "incident_id": case_id,
            "user": item.get("User") or item.get("SubjectUserName") or item.get("TargetUserName") or "unknown-user",
            "system": item.get("Hostname") or item.get("Computer") or "unknown-system",
            "event_type": kind,
            "source": item.get("SourceIp") or item.get("SourceAddress") or "",
            "destination": item.get("DestinationIp") or item.get("DestinationAddress") or "",
            "process": process,
            "resource": item.get("TargetFilename") or item.get("ObjectName") or "",
            "severity": "medium" if kind in ("privilege_change", "service_install") else "low",
            "status": "failure" if kind == "login_failure" else "success",
            "bytes_transferred": 0,
            "raw_event": f"Recorded EventID {event_id}: {source_name or 'Windows event'}",
            "source_record_id": str(index),
        })
    if not records:
        raise ValueError(f"No supported events in {case_id}.")
    return normalize_events(pd.DataFrame.from_records(records)), len(source_rows)


def evaluate() -> dict[str, object]:
    models = load_models()
    if any(status != "Loaded" for status in models.statuses.values()):
        raise RuntimeError(f"Saved artifacts are required: {models.statuses}")
    output = []
    for expected, filename, url in SOURCES:
        request = urllib.request.Request(url, headers={"User-Agent": "TRACE-Research-Evaluation"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        events, source_count = convert(payload, filename.replace(".zip", ""))
        report = analyze_events(events, models)
        output.append({
            "source": url, "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "capture": filename, "raw_events": source_count, "mapped_events": len(events),
            "mapping_coverage": len(events) / source_count,
            "mapped_event_types": events["event_type"].value_counts().to_dict(),
            "source_tactic": expected, "predicted_category": report["incident_type"],
            "incident_detected": bool(report["incident_detected"]),
            "model_score": float(report["selected_hypothesis_probability"]),
        })
    result = {
        "scope": "Two independent OTRF recorded attack-emulation captures; not a seven-class real-world accuracy estimate",
        "source_license": "MIT; https://github.com/OTRF/Security-Datasets/blob/master/LICENSE",
        "classifier_sha256": hashlib.sha256((ARTIFACT_DIR / ARTIFACT_FILES["classifier"]).read_bytes()).hexdigest(),
        "cases": output,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "recorded_telemetry_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
