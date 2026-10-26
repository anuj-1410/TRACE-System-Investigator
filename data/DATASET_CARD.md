# Training data card

| Property | Value |
| --- | --- |
| File | `trace_synthetic_cases.csv` |
| Cases | 2,985 |
| Event rows | 34,831 in the seeded generator output |
| Events per case | 4–24 |
| Origin | Programmatically simulated; no captured organization logs |
| Generator | `trace_core/training_data.py`, seed 41 |
| Split | 60% fit, 20% calibration, 20% untouched test, grouped by `incident_id` |
| Purpose | Train and inspect the seven-category TRACE prototype |

| Category | Cases |
| --- | ---: |
| Normal Activity | 1,500 |
| Account Compromise | 410 |
| Privilege Escalation | 315 |
| Suspicious Process Execution | 260 |
| Data Exfiltration | 215 |
| Denial-of-Service-like Activity | 165 |
| Multi-stage Attack | 120 |

Cases have variable length, time, user, host, source, destination, process, resource, status, severity, transfer size, and benign context. Several categories share event types; some routine cases include administrator work, failures, alerts, or transfers. The dataset is intentionally imbalanced and less separable than the earlier four-motif set. Documentation IP ranges identify no live host.

**Scope:** test cases remain from the same generator and do not measure accuracy on an independent organization. The seven showcase CSVs are separately authored fictional examples. `scripts/evaluate_recorded_telemetry.py` checks two recorded OTRF lab attack captures, with limited event mapping; those captures do not constitute a seven-class real-world benchmark.

For genuine field performance, obtain permissioned, analyst-labeled enterprise case histories for all seven labels; preserve raw source fields; split by organization or time as well as case; and evaluate class support, false-alert rate, calibration, and distribution shift. Useful public sources to assess for a future adapter are [LANL host and network events](https://csr.lanl.gov/data/2017/), [OTRF Security-Datasets](https://github.com/OTRF/Security-Datasets), and [CICIDS2017 network flows](https://www.unb.ca/cic/datasets/ids-2017.html). Their original labels and fields are not interchangeable with TRACE's seven incident-level labels.
