"""Downloadable JSON and standalone HTML investigation reports."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone


def report_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


def report_html(report: dict[str, object]) -> str:
    """Build a self-contained HTML report with escaped log-derived content."""
    category_names = {
        "Normal Activity": "Routine activity",
        "Account Compromise": "Possible account takeover",
        "Privilege Escalation": "Account gained higher access",
        "Suspicious Process Execution": "An unusual program ran",
        "Data Exfiltration": "Possible data sent out",
        "Denial-of-Service-like Activity": "A service may have been overwhelmed",
        "Multi-stage Attack": "Linked suspicious activity",
    }
    stage_names = {
        "Authentication / initial access": "Sign-in activity",
        "Privilege or account change": "Account access changed",
        "Process execution": "A program ran",
        "External communication": "A device contacted another address",
        "Data transfer": "Data was transferred",
        "Possible evidence removal": "A file was deleted",
    }
    raw_category = str(report.get("incident_type", "Incident Investigation"))
    title = html.escape(category_names.get(raw_category, raw_category))
    incident_class = "alert" if report.get("incident_detected") else "normal"
    outcome = "This activity needs a closer look" if report.get("incident_detected") else "No clear warning found"
    summary_text = (
        f"We reviewed {int(report.get('event_count', 0))} records. The activity most closely matches {category_names.get(raw_category, raw_category).lower()}. "
        "This is a starting assessment from the supplied records, not confirmation of a breach."
    )
    impact = report.get("impact", {})
    evidence = report.get("evidence", [])
    timeline = report.get("timeline", [])
    hypotheses = report.get("incident_probabilities", [])
    root_causes = report.get("root_causes", [])
    next_stages = report.get("next_stage_forecast", [])
    recommendations = report.get("recommendations", [])
    provenance = report.get("content_provenance", {})
    counterfactual = report.get("counterfactual", {})
    root_score_text = f"{float(report.get('root_cause_probability', 0)):.1%}" if report.get("incident_detected") else "Not assigned"
    evidence_rows = "".join(
        f"<li><strong>{html.escape(str(item['title']))}</strong> — {html.escape(str(item['detail']))} "
        f"<small>Related records: {html.escape(', '.join('#' + str(value) for value in item.get('event_ids', [])) or 'none')}</small></li>" for item in evidence
    ) or "<li>No positive evidence indicators were found.</li>"
    timeline_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['timestamp']))}</td>"
        f"<td>{html.escape(stage_names.get(str(item['stage_label']), str(item['stage_label'])))}</td>"
        f"<td>{html.escape(str(item['raw_event']))}</td>"
        f"<td>{html.escape(str(item['user']))} / {html.escape(str(item['system']))}</td>"
        f"<td>{float(item['transition_probability']):.1%}</td></tr>" for item in timeline
    )
    hypothesis_rows = "".join(
        f"<tr><td>{html.escape(category_names.get(str(item['incident_type']), str(item['incident_type'])))}</td><td>{float(item['probability']):.1%}</td></tr>" for item in hypotheses
    )
    cause_rows = "".join(
        f"<tr><td>{html.escape(str(item['cause']))}</td><td>{float(item['probability']):.1%}</td></tr>" for item in root_causes
    )
    next_stage_rows = "".join(
        f"<tr><td>{html.escape(stage_names.get(str(item['stage_label']), str(item['stage_label'])))}</td><td>{float(item['probability']):.1%}</td></tr>" for item in next_stages
    )
    action_rows = "".join(
        f"<li><strong>{html.escape(str(item['priority']))}:</strong> {html.escape(str(item['action']))} "
        f"<small>{html.escape(str(item.get('reason', '')))}"
        f"{'; events ' + html.escape(', '.join('#' + str(value) for value in item.get('event_ids', []))) if item.get('event_ids') else ''}</small></li>"
        for item in recommendations
    )
    if counterfactual.get("available"):
        counterfactual_html = (
            f"<p><strong>Intervention:</strong> {html.escape(str(counterfactual.get('action')))} after "
            f"{html.escape(str(counterfactual.get('pivot_description')))}.</p>"
            f"<div class=\"metrics\"><div class=\"metric\">Expected downstream risk<b>{float(counterfactual.get('baseline_expected_risk', 0)):.2f}</b></div>"
            f"<div class=\"metric\">After simulated response<b>{float(counterfactual.get('counterfactual_expected_risk', 0)):.2f}</b></div>"
            f"<div class=\"metric\">Estimated reduction<b>{float(counterfactual.get('estimated_risk_reduction_percent', 0)):.1f}%</b></div></div>"
            f"<p class=\"note\">{html.escape(str(counterfactual.get('limitation', '')))}</p>"
        )
    else:
        counterfactual_html = f"<p>{html.escape(str(counterfactual.get('message', 'No what-if simulation was run.')))}</p>"
    users = ", ".join(html.escape(str(value)) for value in impact.get("affected_users", [])) or "Not identified"
    systems = ", ".join(html.escape(str(value)) for value in impact.get("affected_systems", [])) or "Not identified"
    resources = ", ".join(html.escape(str(value)) for value in impact.get("affected_resources", [])) or "Not identified"
    generated_value = str(report.get("generated_at", datetime.now(timezone.utc).isoformat()))
    try:
        generated = datetime.fromisoformat(generated_value.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
    except ValueError:
        generated = html.escape(generated_value)
    limitations = "".join(f"<li>{html.escape(str(item))}</li>" for item in report.get("limitations", []))
    provenance_rows = "".join(
        f"<tr><td>{html.escape(str(component))}</td><td>{html.escape(str(method))}</td></tr>"
        for component, method in provenance.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRACE Investigation — {title}</title>
<style>
:root{{font-family:Inter,Segoe UI,Arial,sans-serif;color:#f3f8fc;background:#0b1627}}body{{margin:0;line-height:1.55}}main{{max-width:1080px;margin:32px auto;padding:0 20px}}header,.panel{{background:#14263c;border:1px solid #304b62;border-radius:14px;padding:24px;margin-bottom:16px}}h1{{margin:6px 0 10px;font-size:32px;line-height:1.2}}h2{{font-size:21px;margin:0 0 14px}}.eyebrow{{color:#78e1da;font-size:13px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.alert{{border-left:5px solid #ff8897}}.normal{{border-left:5px solid #69ddbc}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}.metric{{background:#1a3049;padding:14px;border:1px solid #304b62;border-radius:9px}}.metric b{{display:block;color:#fff;font-size:21px;margin-top:6px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #304b62;vertical-align:top}}th{{color:#b7cfdd}}li{{margin:9px 0;line-height:1.5}}small,.note{{color:#bcd0de;font-size:13px}}@media print{{:root,body{{color:#172334;background:white}}main{{margin:0 auto}}header,.panel,.metric{{background:white;color:#172334;border-color:#dce4ed;break-inside:avoid}}.metric b{{color:#172334}}th,small,.note{{color:#53687d}}}}
</style></head><body><main>
<header class="{incident_class}"><div class="eyebrow">TRACE · Incident investigation report</div><h1>{title}</h1>
<p><strong>{outcome}</strong> · Generated {generated}</p>
<p>{html.escape(summary_text)}</p>
<div class="metrics"><div class="metric">How urgent<b>{html.escape(str(impact.get('severity', 'Unknown')))}</b></div>
<div class="metric">Potential data exposure<b>{html.escape(str(impact.get('potential_data_exposure', 'Unknown')))}</b></div>
<div class="metric">Records reviewed<b>{int(report.get('event_count', 0))}</b></div></div>
<p class="note">Scores are model estimates, not confirmation of a security incident.</p></header>
<section class="panel"><h2>What raised concern</h2><ul>{evidence_rows}</ul></section>
<section class="panel"><h2>What happened, in order</h2><table><thead><tr><th>Time (UTC)</th><th>Activity</th><th>Original record</th><th>Person / device</th><th>Sequence score</th></tr></thead><tbody>{timeline_rows}</tbody></table></section>
<section class="panel"><h2>Who and what appear in the records</h2><div class="metrics"><div class="metric">People named<b>{len(impact.get('affected_users', []))}</b><small>{users}</small></div>
<div class="metric">Devices named<b>{len(impact.get('affected_systems', []))}</b><small>{systems}</small></div>
<div class="metric">Outside connections<b>{int(impact.get('events_with_external_communication', 0))}</b></div>
<div class="metric">Data moved<b>{int(impact.get('transferred_bytes', 0)):,} bytes</b></div></div>
<p><strong>Files and resources:</strong> {resources}</p></section>
<section class="panel"><h2>What to check next</h2><ul>{action_rows}</ul><p class="note">Review these suggestions before making changes.</p></section>
<section class="panel"><h2>A possible response</h2>{counterfactual_html}</section>
<section class="panel"><h2>Other possible explanations</h2><table><thead><tr><th>Pattern</th><th>Model score</th></tr></thead><tbody>{hypothesis_rows}</tbody></table>
<h2>Possible cause: {html.escape(str(report.get('probable_root_cause', 'Unknown')))}</h2><table><thead><tr><th>Cause</th><th>Model score</th></tr></thead><tbody>{cause_rows}</tbody></table>
<h2>Possible next activity</h2><table><thead><tr><th>Activity</th><th>Model score</th></tr></thead><tbody>{next_stage_rows}</tbody></table>
<p class="note">These scores are estimates from example patterns and assumptions. They are not guarantees or confirmed causes.</p></section>
<section class="panel"><h2>How TRACE made this report</h2><table><thead><tr><th>Part</th><th>Source</th></tr></thead><tbody>{provenance_rows}</tbody></table>
<p class="note">Unusual activity: {float(report.get('anomaly_score', 0)):.1%} model score · Closest pattern: {float(report.get('selected_hypothesis_probability', 0)):.1%} model score · Possible cause: {root_score_text}</p></section>
<section class="panel"><h2>What this report cannot tell you</h2><ul>{limitations}</ul></section>
</main></body></html>"""
