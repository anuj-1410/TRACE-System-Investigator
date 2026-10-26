# Seven upload-ready investigation cases

These CSVs are **realistic synthetic examples**, not logs captured from real people, companies, or attacks. They are separate from the generated training dataset. Names and systems are fictional, external addresses use documentation-only IP ranges, and domains use `example.net`. Each file contains one case with ordered events from several plausible log sources. The `raw_event` text gives readable context; the structured columns drive TRACE's analysis.

| Example in the dashboard | CSV | Records | Story shown |
| --- | --- | ---: | --- |
| Routine activity | [01_routine_activity.csv](01_routine_activity.csv) | 10 | Managed workstation sign-in, internal documents, small internal save, sign-out |
| Account takeover | [02_account_takeover.csv](02_account_takeover.csv) | 9 | Repeated failed sign-ins, unusual success, account changes |
| Higher access | [03_higher_access.csv](03_higher_access.csv) | 9 | Service account gains administrator rights and installs a service |
| Unusual program | [04_unusual_program.csv](04_unusual_program.csv) | 9 | Document starts PowerShell, temporary program, endpoint alert |
| Possible data loss | [05_possible_data_loss.csv](05_possible_data_loss.csv) | 10 | Sales files archived and a large outside upload recorded |
| Service disruption | [06_service_disruption.csv](06_service_disruption.csv) | 11 | Web request burst, firewall rate limits, slow portal response |
| Several suspicious steps | [07_linked_incident.csv](07_linked_incident.csv) | 12 | Unusual sign-in, higher access, remote movement, outside transfer |

## Use them

1. Start TRACE with `python -m streamlit run app.py`.
2. Select **Upload a CSV** in the sidebar and upload one file from this folder.
3. Review the four top-line measures, evidence, timeline, people and next steps, then download the report.

The files use the accepted `timestamp`, `incident_id`, `user`, `system`, `event_type`, `source`, `destination`, `process`, `resource`, `severity`, `status`, and `bytes_transferred` fields. `log_source` and `source_record_id` preserve context for a human reader. Every file has exactly one `incident_id`, so the upload opens directly without a case-selection step. No answer label is included in the CSV; TRACE infers the pattern from the records.

With the bundled saved artifacts, all seven files normalized without dropped records and each returned its intended category when checked on 25 October 2026. Results may differ after retraining or changing the analysis code. The model scores are estimates on constructed examples, not measured real-world detection rates.

Regenerate the same files from the project root with `python scripts/generate_showcase_datasets.py`. The script uses only the Python standard library.
