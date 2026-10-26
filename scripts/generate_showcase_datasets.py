"""Write seven deterministic, realistic synthetic cases for dashboard demos.

These records are fictional and must never be presented as captured incident logs.
Public documentation IP ranges and example domains keep the files safe to share.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "showcase"
FIELDS = (
    "timestamp", "incident_id", "user", "system", "event_type", "source",
    "destination", "process", "resource", "severity", "status",
    "bytes_transferred", "log_source", "source_record_id", "raw_event",
)


def event(offset: int, kind: str, message: str, **details: object) -> dict[str, object]:
    return {"offset": offset, "event_type": kind, "raw_event": message, **details}


CASES: tuple[dict[str, object], ...] = (
    {
        "file": "01_routine_activity.csv", "label": "Normal Activity",
        "case": "SHOWCASE-ROUTINE-001", "start": "2026-09-14T08:42:00Z",
        "user": "maya.chen", "system": "WS-FIN-014", "source": "10.24.5.18",
        "events": (
            event(0, "login_success", "Employee signed in to the finance workstation with a managed device.", log_source="Windows Security", status="success"),
            event(2, "file_access", "Quarterly planning workbook opened from the internal team share.", log_source="File audit", process="excel.exe", resource=r"\\files-01\Finance\Planning\Q3-plan.xlsx"),
            event(4, "process_start", "Approved spreadsheet application started from Program Files.", log_source="Sysmon", process=r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"),
            event(9, "file_access", "Workbook autosave completed on the internal team share.", log_source="File audit", process="excel.exe", resource=r"\\files-01\Finance\Planning\Q3-plan.xlsx"),
            event(12, "process_start", "Managed browser started for the company expense portal.", log_source="Sysmon", process="msedge.exe"),
            event(16, "file_access", "Expense policy page opened from the internal portal.", log_source="Web proxy", process="msedge.exe", resource="intranet.example.net/expenses/policy"),
            event(23, "data_transfer", "Small workbook update saved to the internal file share.", log_source="File audit", process="excel.exe", destination="10.24.2.11", bytes_transferred=86240),
            event(28, "file_access", "Team calendar file read from the internal share.", log_source="File audit", process="msedge.exe", resource=r"\\files-01\Teams\Calendar\week-38.ics"),
            event(35, "process_start", "Approved PDF reader opened a local policy document.", log_source="Sysmon", process="AcroRd32.exe", resource=r"C:\Users\maya.chen\Documents\policy.pdf"),
            event(47, "logout", "Employee signed out of the workstation at the end of the session.", log_source="Windows Security"),
        ),
    },
    {
        "file": "02_account_takeover.csv", "label": "Account Compromise",
        "case": "SHOWCASE-ACCOUNT-002", "start": "2026-09-15T02:13:00Z",
        "user": "alex.patel", "system": "IDP-01", "source": "203.0.113.46",
        "events": (
            event(0, "login_failure", "Remote sign-in failed for alex.patel: incorrect password from an unfamiliar address.", log_source="Identity provider", status="failure", severity="medium"),
            event(1, "login_failure", "Second remote sign-in failed for alex.patel from the same unfamiliar address.", log_source="Identity provider", status="failure", severity="medium"),
            event(2, "login_failure", "Third remote sign-in failed for alex.patel from the same unfamiliar address.", log_source="Identity provider", status="failure", severity="high"),
            event(4, "login_success", "Unusual successful sign-in for alex.patel from a new location at 02:17 UTC.", log_source="Identity provider", severity="critical", status="success"),
            event(5, "account_change", "A new device was registered to the account after the unusual sign-in.", log_source="Identity provider", severity="high", resource="alex.patel/device-registration"),
            event(7, "password_reset", "Account password was changed from the newly registered session.", log_source="Identity provider", severity="high", resource="alex.patel/password"),
            event(9, "external_connection", "Account session connected to an unfamiliar remote address.", log_source="Web proxy", destination="198.51.100.74", process="msedge.exe", severity="high"),
            event(11, "file_access", "Account opened a personal profile page that is rarely accessed at this hour.", log_source="Application audit", resource="profile/alex.patel", severity="medium"),
            event(15, "logout", "Remote account session ended.", log_source="Identity provider", severity="medium"),
        ),
    },
    {
        "file": "03_higher_access.csv", "label": "Privilege Escalation",
        "case": "SHOWCASE-PRIVILEGE-003", "start": "2026-09-16T11:20:00Z",
        "user": "svc-patch", "system": "APP-02", "source": "10.24.8.31",
        "events": (
            event(0, "login_success", "Service account signed in to the application server during a maintenance window.", log_source="Windows Security", severity="medium"),
            event(3, "file_access", "Maintenance package was read from the approved internal share.", log_source="File audit", resource=r"\\files-01\Packages\service-update.msi"),
            event(5, "account_change", "Service account was added to a local administrator group outside the change ticket.", log_source="Windows Security", resource="Local Administrators", severity="high"),
            event(7, "privilege_change", "Administrative access was granted to svc-patch on APP-02.", log_source="Windows Security", resource="Local Administrators", severity="critical"),
            event(9, "process_start", "Elevated PowerShell session started under the service account.", log_source="Sysmon", process="powershell.exe", severity="high"),
            event(11, "service_install", "A new background service was installed with elevated access.", log_source="Service Control Manager", process="update-helper.exe", resource="UpdateHelper", severity="high"),
            event(14, "account_change", "Audit settings for the service account were changed after elevation.", log_source="Windows Security", resource="svc-patch/audit-policy", severity="high"),
            event(17, "process_start", "Privileged command shell read the new service configuration.", log_source="Sysmon", process="powershell.exe", severity="medium"),
            event(23, "logout", "Service account session ended.", log_source="Windows Security", severity="medium"),
        ),
    },
    {
        "file": "04_unusual_program.csv", "label": "Suspicious Process Execution",
        "case": "SHOWCASE-PROCESS-004", "start": "2026-09-17T14:06:00Z",
        "user": "nina.rao", "system": "WS-MKT-022", "source": "10.24.6.42",
        "events": (
            event(0, "login_success", "Employee signed in to a managed workstation.", log_source="Windows Security"),
            event(3, "file_access", "Employee opened an email attachment from a new sender.", log_source="File audit", resource=r"C:\Users\nina.rao\Downloads\invoice-review.docm", severity="medium"),
            event(4, "process_start", "Word opened the downloaded document.", log_source="Sysmon", process="WINWORD.EXE", severity="medium"),
            event(5, "process_start", "Word started an encoded PowerShell command from the downloaded document.", log_source="Sysmon", process="powershell.exe", resource=r"C:\Users\nina.rao\Downloads\invoice-review.docm", severity="critical"),
            event(6, "process_start", "PowerShell launched an unsigned program from the temporary directory.", log_source="Sysmon", process=r"C:\Users\nina.rao\AppData\Local\Temp\update-check.exe", severity="critical"),
            event(8, "scheduled_task", "An unrecognized scheduled task was created to start the temporary program at sign-in.", log_source="Task Scheduler", process="update-check.exe", resource="UpdateCheckUser", severity="high"),
            event(9, "endpoint_alert", "Endpoint protection flagged the temporary program as suspicious.", log_source="EDR", process="update-check.exe", status="alert", severity="critical"),
            event(11, "malware_detected", "Endpoint protection quarantined a suspected malicious file.", log_source="EDR", process="update-check.exe", status="alert", resource=r"C:\Users\nina.rao\AppData\Local\Temp\update-check.exe", severity="critical"),
            event(17, "file_access", "Security analyst opened the quarantine record for review.", log_source="EDR", resource="quarantine/update-check.exe", severity="medium"),
        ),
    },
    {
        "file": "05_possible_data_loss.csv", "label": "Data Exfiltration",
        "case": "SHOWCASE-DATA-005", "start": "2026-09-18T18:42:00Z",
        "user": "samir.khan", "system": "WS-SALES-007", "source": "10.24.7.27",
        "events": (
            event(0, "login_success", "Employee signed in to the sales workstation after office hours.", log_source="Windows Security", severity="medium"),
            event(3, "file_access", "Customer account export was opened from the internal sales share.", log_source="File audit", resource=r"\\files-01\Sales\Exports\customers-sep.csv", severity="high"),
            event(4, "file_access", "Pricing workbook was opened from the same internal share.", log_source="File audit", resource=r"\\files-01\Sales\Pricing\renewal-rates.xlsx", severity="high"),
            event(6, "process_start", "Archive utility started to package the two sales files.", log_source="Sysmon", process="7z.exe", severity="medium"),
            event(8, "file_access", "Temporary archive containing sales exports was created.", log_source="File audit", process="7z.exe", resource=r"C:\Users\samir.khan\AppData\Local\Temp\sales-review.zip", severity="high"),
            event(10, "dns_query", "Workstation looked up files.sync-gateway.example.net.", log_source="DNS resolver", destination="198.51.100.88", severity="medium"),
            event(11, "external_connection", "Archive utility opened an encrypted connection to an unapproved file-sharing address.", log_source="Web proxy", process="7z.exe", destination="198.51.100.88", severity="high"),
            event(13, "data_transfer", "Large upload of the sales archive was recorded to the unapproved address.", log_source="Web proxy", process="7z.exe", resource="sales-review.zip", destination="198.51.100.88", bytes_transferred=68453210, severity="critical"),
            event(15, "file_delete", "Temporary sales archive was deleted from the workstation after upload.", log_source="File audit", process="7z.exe", resource=r"C:\Users\samir.khan\AppData\Local\Temp\sales-review.zip", severity="high"),
            event(19, "logout", "Employee session ended after the transfer.", log_source="Windows Security", severity="medium"),
        ),
    },
    {
        "file": "06_service_disruption.csv", "label": "Denial-of-Service-like Activity",
        "case": "SHOWCASE-SERVICE-006", "start": "2026-09-19T10:02:00Z",
        "user": "web-gateway", "system": "WEB-EDGE-01", "source": "203.0.113.118",
        "events": (
            event(0, "external_connection", "Web gateway recorded a sharp rise in inbound requests from one external address.", log_source="Web server", destination="10.24.1.20", severity="high"),
            event(1, "login_failure", "Automated sign-in attempt failed during the request burst.", log_source="Identity provider", status="failure", severity="medium"),
            event(2, "login_failure", "Another automated sign-in attempt failed from the same source.", log_source="Identity provider", status="failure", severity="medium"),
            event(3, "login_failure", "Third automated sign-in attempt failed from the same source.", log_source="Identity provider", status="failure", severity="high"),
            event(4, "external_connection", "The gateway recorded approximately 4,800 requests per minute from the external source.", log_source="Web server", destination="10.24.1.20", severity="critical"),
            event(5, "firewall_block", "Edge firewall blocked requests that exceeded the rate limit.", log_source="Firewall", destination="10.24.1.20", status="blocked", severity="high"),
            event(6, "external_connection", "New connection attempts continued after the first rate-limit block.", log_source="Web server", destination="10.24.1.20", severity="high"),
            event(7, "firewall_block", "Firewall blocked a second high-rate connection group.", log_source="Firewall", destination="10.24.1.20", status="blocked", severity="high"),
            event(8, "endpoint_alert", "Availability monitor reported elevated response times for the public portal.", log_source="Availability monitor", status="alert", severity="high"),
            event(9, "firewall_block", "Firewall continued blocking bursts while the portal recovered.", log_source="Firewall", destination="10.24.1.20", status="blocked", severity="high"),
            event(12, "external_connection", "Inbound request rate returned toward its normal range.", log_source="Web server", destination="10.24.1.20", severity="medium"),
        ),
    },
    {
        "file": "07_linked_incident.csv", "label": "Multi-stage Attack",
        "case": "SHOWCASE-LINKED-007", "start": "2026-09-20T01:18:00Z",
        "user": "olivia.brown", "system": "WS-HR-009", "source": "203.0.113.61",
        "events": (
            event(0, "login_failure", "Remote sign-in for olivia.brown failed from an unfamiliar address.", log_source="Identity provider", status="failure", severity="medium"),
            event(1, "login_failure", "Second sign-in from the same address failed.", log_source="Identity provider", status="failure", severity="medium"),
            event(3, "login_success", "Unusual sign-in succeeded from the unfamiliar address at 01:21 UTC.", log_source="Identity provider", severity="critical"),
            event(6, "account_change", "A recovery method was changed on the account after sign-in.", log_source="Identity provider", resource="olivia.brown/recovery-method", severity="high"),
            event(8, "privilege_change", "The account gained local administrator access on the workstation.", log_source="Windows Security", resource="Local Administrators", severity="critical"),
            event(10, "process_start", "An encoded PowerShell command started under the elevated account.", log_source="Sysmon", process="powershell.exe", severity="critical"),
            event(12, "lateral_movement", "Workstation opened a remote service connection to the HR file server.", log_source="Windows Security", destination="10.24.2.15", process="powershell.exe", severity="high"),
            event(15, "file_access", "Employee records were opened from the HR file server.", log_source="File audit", destination="10.24.2.15", resource=r"\\files-hr\HR\Exports\employee-roster.csv", severity="high"),
            event(18, "external_connection", "Workstation connected to an unfamiliar outside address.", log_source="Web proxy", destination="198.51.100.115", process="powershell.exe", severity="high"),
            event(21, "data_transfer", "Large upload of employee records was recorded to the outside address.", log_source="Web proxy", destination="198.51.100.115", process="powershell.exe", resource="employee-roster.csv", bytes_transferred=34790122, severity="critical"),
            event(24, "file_delete", "Temporary copy of the employee roster was deleted after the upload.", log_source="File audit", resource=r"C:\Users\olivia.brown\AppData\Local\Temp\employee-roster.csv", severity="high"),
            event(27, "logout", "The unusual account session ended.", log_source="Identity provider", severity="medium"),
        ),
    },
)


def generate() -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in CASES:
        start = datetime.fromisoformat(str(case["start"]).replace("Z", "+00:00"))
        events = case["events"]
        rows = []
        for index, specification in enumerate(events, start=1):
            details = dict(specification)
            offset = int(details.pop("offset"))
            row: dict[str, object] = {
                "timestamp": (start + timedelta(minutes=offset)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "incident_id": case["case"], "user": case["user"], "system": case["system"],
                "event_type": details.pop("event_type"), "source": case["source"],
                "destination": "", "process": "", "resource": "", "severity": "low",
                "status": "success", "bytes_transferred": 0,
                "log_source": details.pop("log_source", "System audit"),
                "source_record_id": f"{str(case['case']).split('-')[-1]}-{index:04d}",
                "raw_event": details.pop("raw_event"),
            }
            row.update(details)
            rows.append(row)
        path = OUTPUT / str(case["file"])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
        print(f"{path.relative_to(ROOT)}: {len(rows)} records - {case['label']}")
    return written


if __name__ == "__main__":
    generate()
