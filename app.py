"""TRACE offline Streamlit investigation dashboard."""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace_core.config import ARTIFACT_DIR
from trace_core.counterfactual import INTERVENTIONS, simulate_intervention
from trace_core.markov import event_stage
from trace_core.pipeline import analyze_events, load_models, read_training_summary
from trace_core.preprocessing import normalize_events
from trace_core.reporting import report_html, report_json
from trace_core.synthetic import scenario_records


st.set_page_config(page_title="TRACE | Incident Investigation", page_icon="🔎", layout="wide", initial_sidebar_state="auto")

st.markdown("<style>" + (PROJECT_ROOT / "assets" / "dashboard.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_models():
    return load_models(ARTIFACT_DIR)


def _display_percent(value: float) -> str:
    return f"{value:.0%}"


CATEGORY_NAMES = {
    "Normal Activity": "Routine activity",
    "Account Compromise": "Possible account takeover",
    "Privilege Escalation": "Account gained higher access",
    "Suspicious Process Execution": "An unusual program ran",
    "Data Exfiltration": "Possible data sent out",
    "Denial-of-Service-like Activity": "A service may have been overwhelmed",
    "Multi-stage Attack": "Linked suspicious activity",
}

CATEGORY_SUMMARIES = {
    "Normal Activity": "routine activity",
    "Account Compromise": "a possible account takeover",
    "Privilege Escalation": "an account gaining higher access",
    "Suspicious Process Execution": "an unusual program running",
    "Data Exfiltration": "data possibly being sent out",
    "Denial-of-Service-like Activity": "a service being overwhelmed",
    "Multi-stage Attack": "several suspicious steps linked together",
}

CAUSE_NAMES = {
    "Compromised account": "Someone may have used an account without permission",
    "Insider activity": "The activity may have come from someone with access",
    "Misconfiguration": "A system setting may have allowed the activity",
    "Other / unknown": "The cause is still unclear",
    "Undetermined from available evidence": "The available records do not show a clear cause",
}

STAGE_NAMES = {
    "Routine activity": "Routine activity",
    "Authentication / initial access": "Sign-in activity",
    "Privilege or account change": "Account access changed",
    "Process execution": "A program ran",
    "External communication": "A device contacted another address",
    "Data transfer": "Data was transferred",
    "Possible evidence removal": "A file was deleted",
}

EVIDENCE_NAMES = {
    "unusual_authentication": ("An unusual sign-in", "The records show a sign-in pattern that deserves a closer look."),
    "privilege_change": ("Account access changed", "The records show a change to an account or its permissions."),
    "suspicious_process": ("An unusual program or alert", "A program, scheduled task, or security alert appeared in the records."),
    "external_communication": ("A network connection", "A device contacted another address or another system."),
    "large_transfer": ("A large data transfer", "The records show a large transfer or one marked as an upload."),
    "evidence_removal": ("A file was deleted", "A deletion appears in the records. Its purpose is not known."),
    "repeated_auth_failures": ("Repeated failed sign-ins", "Several sign-in attempts failed."),
}


def _friendly_category(value: str) -> str:
    return CATEGORY_NAMES.get(value, value)


def _friendly_cause(value: str) -> str:
    return CAUSE_NAMES.get(value, value)


def _friendly_stage(value: str) -> str:
    return STAGE_NAMES.get(value, value)


def _count_label(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _metric_grid(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        '<div class="trace-metric">'
        f'<span class="trace-metric-label">{escape(label)}</span>'
        f'<span class="trace-metric-value">{escape(value)}</span>'
        f'<span class="trace-metric-note">{escape(note)}</span>'
        '</div>'
        for label, value, note in items
    )
    st.markdown(f'<div class="trace-metric-grid">{cards}</div>', unsafe_allow_html=True)


def _section(kicker: str, title: str, description: str = "") -> None:
    detail = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(
        f'<div class="trace-section-header"><span>{escape(kicker)}</span>'
        f'<h2>{escape(title)}</h2>{detail}</div>',
        unsafe_allow_html=True,
    )


def _signal_list(items: list[tuple[str, float]]) -> None:
    rows = "".join(
        '<div class="trace-signal">'
        f'<span class="trace-signal-name">{escape(label)}</span>'
        f'<span class="trace-signal-value">{value:.1%}</span>'
        f'<div class="trace-signal-track"><div class="trace-signal-fill {"normal" if label == "Routine activity" else ""}" '
        f'style="width:{max(0.0, min(value * 100, 100)):.1f}%"></div></div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="trace-signal-list">{rows}</div>', unsafe_allow_html=True)


def _source_list(items: dict[str, str]) -> None:
    rows = "".join(
        f'<div class="trace-source"><b>{escape(component)}</b><span>{escape(source)}</span></div>'
        for component, source in items.items()
    )
    st.markdown(f'<div class="trace-source-list">{rows}</div>', unsafe_allow_html=True)


def _show_actions(items: list[dict[str, object]]) -> None:
    cards = "".join(
        '<div class="trace-action">'
        f'<span class="trace-action-badge {"high" if item["priority"] == "High" else ""}">{"Check soon" if item["priority"] == "High" else "Follow up"}</span>'
        '<div class="trace-action-body">'
        f'<span class="trace-action-text">{escape(str(item["action"]))}</span>'
        f'<small>{escape(str(item["reason"]))}</small>'
        f'{"<small>Related records: " + escape(", ".join(f"#{event_id}" for event_id in item["event_ids"])) + "</small>" if item["event_ids"] else ""}'
        '</div></div>'
        for item in items
    )
    st.markdown(f'<div class="trace-action-list">{cards}</div>', unsafe_allow_html=True)


def _entity_grid(items: list[tuple[str, list[str], str]]) -> None:
    cards = "".join(
        '<div class="trace-entity">'
        f'<strong>{escape(label)}</strong>'
        f'<span>{escape(", ".join(values) if values else empty)}</span>'
        '</div>'
        for label, values, empty in items
    )
    st.markdown(f'<div class="trace-entity-grid">{cards}</div>', unsafe_allow_html=True)


def _evidence_grid(items: list[dict[str, object]]) -> None:
    if not items:
        st.markdown('<div class="trace-summary">No specific warning signs were found in these records. Keep checking if you have other information that suggests a problem.</div>', unsafe_allow_html=True)
        return
    cards = "".join(
        '<div class="trace-entity">'
        f'<strong>{escape(EVIDENCE_NAMES.get(str(item["key"]), (str(item["title"]), ""))[0])}</strong>'
        f'<span>{escape(EVIDENCE_NAMES.get(str(item["key"]), ("", str(item["detail"])))[1])}</span>'
        f'<small>Related records: {escape(", ".join(f"#{event_id}" for event_id in item["event_ids"]) or "see full details")}</small>'
        '</div>'
        for item in items
    )
    st.markdown(f'<div class="trace-entity-grid">{cards}</div>', unsafe_allow_html=True)


def _demo_options() -> dict[str, str]:
    return {
        "Several suspicious steps": "Multi-stage Attack",
        "Possible data sent out": "Data Exfiltration",
        "Possible account takeover": "Account Compromise",
        "Account gained higher access": "Privilege Escalation",
        "An unusual program ran": "Suspicious Process Execution",
        "A service was overwhelmed": "Denial-of-Service-like Activity",
        "Routine activity": "Normal Activity",
    }


def _get_events() -> pd.DataFrame | None:
    st.sidebar.markdown('<div class="trace-sidebar-label">Start here</div>', unsafe_allow_html=True)
    input_mode = st.sidebar.radio("Choose your source", ("Built-in example", "Upload a CSV"), horizontal=False, key="data_source")
    if input_mode == "Built-in example":
        selected = st.sidebar.selectbox("Choose an example", tuple(_demo_options()))
        return scenario_records(_demo_options()[selected], "demo-case-001", seed=19)

    uploaded = st.sidebar.file_uploader("Choose your activity log (CSV)", type=["csv"], help="The file is read in this local app session. TRACE does not send it to an external service.")
    if uploaded is None:
        return None
    try:
        raw = pd.read_csv(uploaded)
        normalized = normalize_events(raw)
        st.sidebar.caption(f"Read {len(raw):,} rows and kept {len(normalized):,} distinct records.")
        case_ids = normalized["incident_id"].astype(str).str.strip()
        if case_ids.nunique() > 1:
            selected_case = st.sidebar.selectbox("Choose one case", tuple(case_ids.unique()), format_func=lambda case: case or "Unassigned records")
            normalized = normalized.loc[case_ids.eq(selected_case)].reset_index(drop=True).copy()
            normalized["event_id"] = range(1, len(normalized) + 1)
        return normalized
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"We could not read this CSV: {exc}")
        return None


def _show_welcome() -> None:
    st.markdown(
        '<div class="trace-welcome">'
        '<span class="trace-welcome-kicker">YOUR ACTIVITY, EXPLAINED</span>'
        '<h1>Find out what happened.</h1>'
        '<p>Upload a record of system activity or explore an example. TRACE will show the events in order, explain why anything looks unusual, and suggest what to check next.</p>'
        '</div>', unsafe_allow_html=True,
    )
    _section("GET STARTED", "Choose how to begin")
    st.markdown(
        '<div class="trace-start-grid">'
        '<div class="trace-start-card"><span>01</span><strong>Use your own records</strong><p>Choose a CSV activity log in the left panel. Your file stays in this local app session.</p></div>'
        '<div class="trace-start-card"><span>02</span><strong>Try an example</strong><p>See a complete investigation immediately, then switch between example situations.</p></div>'
        '<div class="trace-start-card"><span>03</span><strong>Understand the result</strong><p>Start with the plain-language summary. Open the details only when you want them.</p></div>'
        '</div>', unsafe_allow_html=True,
    )
    st.button("Explore an example", type="primary", on_click=_choose_example)
    st.caption("TRACE provides an informed starting point. It does not confirm a breach or make changes to your system.")


def _choose_example() -> None:
    st.session_state["data_source"] = "Built-in example"


def _build_timeline_chart(timeline: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(timeline)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    colors = {
        "normal": "#93a4b9",
        "initial_access": "#d8a24b",
        "privilege_change": "#d77760",
        "suspicious_execution": "#ce5772",
        "external_communication": "#796ed3",
        "data_transfer": "#4b70e6",
        "evidence_removal": "#2d496b",
    }
    figure = go.Figure()
    frame = frame.sort_values("timestamp", kind="stable")
    figure.add_trace(go.Scatter(
        x=frame["timestamp"], y=frame["stage_label"], mode="lines",
        line={"color": "#5f8298", "width": 2.5}, name="Observed order",
        hoverinfo="skip", showlegend=False,
    ))
    for stage in frame["stage"].drop_duplicates():
        subset = frame[frame["stage"].eq(stage)]
        figure.add_trace(go.Scatter(
            x=subset["timestamp"],
            y=subset["stage_label"],
            mode="markers+text",
            text=subset["event_id"].astype(str),
            textposition="top center",
            name=str(subset["stage_label"].iloc[0]),
            marker={"size": 17, "color": colors.get(stage, "#80b4c6"), "line": {"width": 2, "color": "#dcecf5"}},
            customdata=subset[["raw_event", "user", "system", "transition_probability"]].to_numpy(),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S UTC}<br><b>%{customdata[0]}</b><br>User: %{customdata[1]}<br>System: %{customdata[2]}<br>Markov transition: %{customdata[3]:.1%}<extra></extra>",
        ))
    figure.update_layout(
        height=min(540, max(380, 60 * frame["stage_label"].nunique() + 105)),
        margin={"l": 18, "r": 24, "t": 28, "b": 52},
        xaxis_title="Event time (UTC)", yaxis_title="",
        plot_bgcolor="#14263c", paper_bgcolor="#14263c", showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "size": 14, "color": "#dceaf4"},
        xaxis={"gridcolor": "#304b62", "automargin": True, "zeroline": False},
        yaxis={
            "gridcolor": "#304b62", "automargin": True, "categoryorder": "array",
            "categoryarray": ["Routine activity", "Authentication / initial access", "Privilege or account change", "Process execution", "External communication", "Data transfer", "Possible evidence removal"],
            "tickmode": "array",
            "tickvals": ["Routine activity", "Authentication / initial access", "Privilege or account change", "Process execution", "External communication", "Data transfer", "Possible evidence removal"],
            "ticktext": ["Routine", "Initial access", "Privilege change", "Process execution", "External comms", "Data transfer", "Evidence removal"],
        },
    )
    return figure


def _show_case_header(report: dict[str, object], events: pd.DataFrame) -> None:
    case_id = str(events.iloc[0].get("incident_id", "")).strip() or "Unassigned case"
    display_case = case_id[:45] + ("…" if len(case_id) > 45 else "")
    start = pd.Timestamp(report["time_window"]["start"]).strftime("%d %b %Y · %H:%M UTC")
    st.markdown(
        '<div class="trace-hero"><div class="trace-hero-kicker">YOUR INVESTIGATION</div>'
        '<h1>See what happened on your system.</h1>'
        '<p>Review the activity in order, see what raised concern, and find practical steps to check next.</p>'
        '<div class="trace-hero-meta">'
        f'<span>Case {escape(display_case)}</span><span>{int(report["event_count"])} records reviewed</span>'
        f'<span>Started {escape(start)}</span><span>Checked on this device</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _show_decision(report: dict[str, object]) -> None:
    detected = bool(report["incident_detected"])
    decision = "This activity needs a closer look" if detected else "No clear warning found"
    context = (
        "The records show signs that may point to a security problem. Review the evidence before acting."
        if detected else "These records look mostly routine. Check other information if you still have concerns."
    )
    st.markdown(
        f'<div class="trace-decision {"" if detected else "normal"}" role="status">'
        '<div><div class="trace-decision-kicker">Current assessment</div>'
        f'<strong>{escape(decision)}</strong><p>{escape(context)}</p></div>'
        f'<span class="trace-decision-tag">{escape(_friendly_category(str(report["incident_type"])))}</span></div>',
        unsafe_allow_html=True,
    )
    impact = report["impact"]
    _metric_grid([
        ("Pattern match", _display_percent(float(report["selected_hypothesis_probability"])), "Model score for the selected pattern; not certainty"),
        ("Unusual activity", _display_percent(float(report["anomaly_score"])), "Signal strength compared with normal activity"),
        ("Estimated urgency", str(impact["severity"]), f"{impact['severity_score']}/100 from these records"),
        ("People and devices", f"{_count_label(len(impact['affected_users']), 'person', 'people')} · {_count_label(len(impact['affected_systems']), 'device', 'devices')}", "Named in these records"),
    ])


def _show_overview(report: dict[str, object]) -> None:
    _section("01 / THE STORY", "What we found", "This is a starting assessment based only on the activity you provided.")
    pattern = CATEGORY_SUMMARIES.get(str(report["incident_type"]), str(report["incident_type"]).lower())
    if report["incident_detected"]:
        root = _friendly_cause(str(report["probable_root_cause"]))
        summary = f"The {report['event_count']} records show a pattern that looks like {pattern}. One possible explanation is that {root[0].lower() + root[1:]}. The records suggest {str(report['impact']['potential_data_exposure']).lower()} potential data exposure, but they cannot confirm what happened outside this case."
    else:
        summary = f"We reviewed {report['event_count']} records. They most closely resemble routine activity. Nothing here gives a clear sign of an incident, but these records may not show the whole picture."
    st.markdown(f'<div class="trace-summary">{escape(summary)}</div>', unsafe_allow_html=True)

    _section("02 / WHAT STOOD OUT", "Why TRACE noticed this")
    _evidence_grid(report["evidence"][:3])
    if len(report["evidence"]) > 3:
        st.caption(f"See {len(report['evidence']) - 3} more signs in the Activity timeline tab.")

    _section("03 / NEXT STEPS", "What to check next")
    _show_actions(report["recommendations"][:2])
    st.caption("These steps are suggestions linked to the records. A person should review them before making changes.")

    with st.expander("See other possible explanations and model scores"):
        st.caption("Scores show how closely each pattern matched this example. They are not a measure of certainty or a confirmed cause.")
        left, right = st.columns([1.18, 1], gap="large")
        with left:
            _section("PATTERN COMPARISON", "Other possible activity patterns")
            _signal_list([(_friendly_category(str(item["incident_type"])), float(item["probability"])) for item in report["incident_probabilities"]])
        with right:
            _section("POSSIBLE CAUSE", "What might explain it")
            if report["incident_detected"]:
                _signal_list([(_friendly_cause(str(item["cause"])), float(item["probability"])) for item in report["root_causes"]])
            else:
                st.write("No cause was assigned because the records did not show a clear warning.")
        st.caption(f"Unusual-activity signal: {_display_percent(float(report['anomaly_score']))}. This is a model score, not the chance that an incident occurred.")


def _show_timeline(report: dict[str, object], events: pd.DataFrame) -> None:
    _section("ACTIVITY TIMELINE", "What happened, in order", "These are the events recorded in this case. The descriptions below are the original log text.")
    timeline = report["timeline"]
    cards = "".join(
        '<div class="trace-event">'
        f'<div class="trace-event-index">{int(item["event_id"]):02d}</div>'
        '<div class="trace-event-content">'
        f'<span class="trace-event-time">{escape(pd.Timestamp(item["timestamp"]).strftime("%d %b %Y · %H:%M:%S UTC"))}</span>'
        f'<strong>{escape(_friendly_stage(str(item["stage_label"])))}</strong>'
        f'<p>{escape(str(item["raw_event"]))}</p>'
        f'<small>{escape(str(item["user"]))} · {escape(str(item["system"]))}</small>'
        '</div></div>'
        for item in timeline[:25]
    )
    st.markdown(f'<div class="trace-event-list">{cards}</div>', unsafe_allow_html=True)
    if len(timeline) > 25:
        st.caption(f"Showing the first 25 of {len(timeline)} records here. Open the full table below to see all of them.")

    _section("THE SIGNS", "What raised concern")
    _evidence_grid(report["evidence"])

    with st.expander("See the visual timeline and possible next steps"):
        st.plotly_chart(_build_timeline_chart(timeline), width="stretch", key="timeline_detail", config={"displayModeBar": False})
        st.caption("The chart groups events by activity type. The next-step bars describe patterns learned from example cases; they do not predict what will happen.")
        forecast = pd.DataFrame(report["next_stage_forecast"])
        _section("MODEL ESTIMATE", "Possible next activity")
        _signal_list([(_friendly_stage(str(row["stage_label"])), float(row["probability"])) for _, row in forecast.iterrows()])

    with st.expander("See the full records and technical signals"):
        _section("ALL RECORDS", "Full activity table")
        st.caption("This table preserves the recorded text and shows how the sequence model grouped each event.")
        detail = pd.DataFrame(report["timeline"])
        visible = detail[["event_id", "timestamp", "stage_label", "raw_event", "user", "system", "transition_probability"]].rename(columns={"event_id": "#", "timestamp": "Time (UTC)", "stage_label": "Activity type", "raw_event": "Original record", "user": "Person", "system": "Device", "transition_probability": "Sequence score"})
        visible["Time (UTC)"] = pd.to_datetime(visible["Time (UTC)"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
        visible["Activity type"] = visible["Activity type"].map(_friendly_stage)
        visible["Sequence score"] = visible["Sequence score"].map(lambda value: f"{value:.1%}")
        st.dataframe(visible, hide_index=True, width="stretch", height=min(430, 40 * (len(visible) + 1)))
        _section("UNUSUAL ACTIVITY", "Records with the strongest warning signal")
        anomaly = pd.DataFrame(report["anomaly_events"])
        if not anomaly.empty:
            anomaly = anomaly.merge(events[["event_id", "timestamp", "raw_event", "system", "user"]], on="event_id", how="left")
            anomaly["timestamp"] = pd.to_datetime(anomaly["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            columns = [name for name in ["event_id", "timestamp", "raw_event", "system", "user", "anomaly_score", "reconstruction_error", "threshold_ratio", "flagged"] if name in anomaly]
            display = anomaly[columns].copy()
            display["anomaly_score"] = display["anomaly_score"].map(lambda value: f"{value:.1%}")
            if "reconstruction_error" in display:
                display["reconstruction_error"] = display["reconstruction_error"].map(lambda value: "—" if pd.isna(value) else f"{value:.5f}")
            if "threshold_ratio" in display:
                display["threshold_ratio"] = display["threshold_ratio"].map(lambda value: "—" if pd.isna(value) else f"{value:.2f}×")
            st.dataframe(display, hide_index=True, width="stretch", height=min(450, 40 * (len(display) + 1)))
        rbm = report["rbm_patterns"]
        _section("MODEL DETAILS", "Patterns shared across records")
        if rbm.get("available"):
            cols = st.columns(2)
            cols[0].metric("Mean reconstruction error", f"{rbm['mean_reconstruction_error']:.3f}")
            cols[1].metric("Mean hidden activation", f"{rbm['mean_hidden_activation']:.1%}")
            active_patterns = pd.DataFrame(rbm["active_patterns"]).rename(columns={"unit": "Latent unit", "activation": "Mean activation"})
            active_patterns["Mean activation"] = active_patterns["Mean activation"].map(lambda value: f"{value:.1%}")
            st.dataframe(active_patterns, hide_index=True, width="stretch", height=min(270, 40 * (len(active_patterns) + 1)))
            st.caption(rbm["description"])
        else:
            st.info(rbm["message"])


def _show_impact(report: dict[str, object]) -> None:
    impact = report["impact"]
    _section("PEOPLE AND DEVICES", "Who and what appear in the records", "Being named in a log does not mean a person or device was harmed.")
    _metric_grid([
        ("People named", str(len(impact["affected_users"])), "In the records you supplied"),
        ("Devices named", str(len(impact["affected_systems"])), "In the records you supplied"),
        ("Outside connections", str(impact["events_with_external_communication"]), "Connection records"),
        ("Data moved", f"{impact['transferred_bytes']:,} bytes", "Amount recorded; may be incomplete"),
    ])
    _entity_grid([
        ("People", impact["affected_users"], "No person named"),
        ("Devices", impact["affected_systems"], "No device named"),
        ("Files and resources", impact["affected_resources"], "No file or resource named"),
        ("Programs to review", impact["suspicious_processes"], "No unusual program named"),
        ("Outside addresses", impact["external_destinations"], "No outside address named"),
    ])
    _section("PRACTICAL GUIDANCE", "What to check next", "These steps are suggested from the signs in your records. Review them before taking action.")
    _show_actions(report["recommendations"])
    with st.expander("How was the urgency level estimated?"):
        st.progress(min(float(impact["severity_score"]) / 100, 1.0), text=f"Estimated urgency score · {impact['severity_score']}/100")
        st.write("The score combines unusual activity, the closest matching pattern, and the people, devices, and data named in these records. It is not a measure of confirmed damage.")


def _show_what_if(events: pd.DataFrame, models, report: dict[str, object]) -> dict[str, object]:
    _section("TRY A RESPONSE", "What might have changed?", "Choose a possible step and a point in the timeline to explore a rough estimate.")
    st.markdown('<div class="trace-summary">This is only a thought experiment. TRACE will not disable an account, disconnect a device, or block an address.</div>', unsafe_allow_html=True)
    run_simulation = st.checkbox("Compare a possible response and include it in my report", value=False)
    if not run_simulation:
        return {"available": False, "message": "No response comparison was run."}
    available_actions = [name for name, config in INTERVENTIONS.items() if events["event_type"].map(event_stage).isin(config["anchors"]).any()]
    if not available_actions:
        st.info("There is no suitable point in these records for this comparison. Try another case.")
        return {"available": False, "message": "No suitable comparison point was found."}
    action = st.selectbox("Which step would you try?", available_actions)
    allowed_stages = INTERVENTIONS[action]["anchors"]
    eligible = events[events["event_type"].map(event_stage).isin(allowed_stages)]
    eligible_options = {
        int(row["event_id"]): f"#{int(row['event_id'])} · {pd.Timestamp(row['timestamp']).strftime('%H:%M:%S')} UTC · {row['raw_event']}"
        for _, row in eligible.iterrows()
    }
    pivot_event_id = st.selectbox("At what point in the timeline?", tuple(eligible_options), format_func=lambda event_id: eligible_options[event_id])
    result = simulate_intervention(events, models.markov, action, int(pivot_event_id))
    _metric_grid([
        ("Possible risk reduction", f"{result['estimated_risk_reduction_percent']:.0f}%", "Rough model estimate"),
        ("Later records compared", str(result["downstream_event_count"]), "After the point you chose"),
        ("Action taken", "None", "This comparison makes no changes"),
    ])
    st.caption("This number depends on assumptions about how well the response works. It cannot tell us what would really have happened.")
    with st.expander("See the calculation details"):
        st.write(f"Before the selected step: {result['baseline_expected_risk']:.2f} model units. After it: {result['counterfactual_expected_risk']:.2f} model units.")
        st.write(f"Assumed near-term effectiveness: {result['assumed_near_term_control_effectiveness_percent']:.0f}%. Sequence model: {report['markov_method']}.")
        st.write(result["limitation"])
    return {"available": True, **result}


def _show_models(report: dict[str, object]) -> None:
    models = cached_models()
    _section("05 / Provenance", "How this investigation was produced", "Scores, explanations, and response suggestions have different sources.")
    _source_list(report["content_provenance"])
    st.caption("The VAE and RBM analyze event patterns. Report prose and response guidance come from fixed evidence-linked rules, with no external language model.")
    _section("Artifact health", "Loaded model components")
    friendly_names = {"vae": "VAE anomaly detector", "rbm": "RBM hidden patterns", "classifier": "Incident classifier", "markov": "Markov progression"}
    status_table = pd.DataFrame([{"Component": friendly_names[key], "Status": value} for key, value in models.statuses.items()])
    st.dataframe(status_table, hide_index=True, width="stretch", height=178)
    if not all(value == "Loaded" for key, value in models.statuses.items() if key != "markov"):
        st.info("Run the training command in README.md to create model artifacts. The analysis report remains available now, with every fallback labeled.")
    summary = read_training_summary()
    if summary:
        _section("Training record", "Saved evaluation snapshot", "Scores below come from held-out cases generated by the same simulation program.")
        dataset = summary.get("dataset", {})
        trained_at = pd.Timestamp(summary.get("trained_at")).strftime("%d %b %Y · %H:%M UTC") if summary.get("trained_at") else "Unknown time"
        st.caption(f"Trained {trained_at} · {summary.get('source', 'Unknown source')} · {dataset.get('training_sequences', '—')} train / {dataset.get('calibration_sequences', '—')} calibration / {dataset.get('held_out_sequences', '—')} test cases")
        classifier = summary.get("classifier", {})
        vae = summary.get("vae", {}).get("metrics", {})
        markov = summary.get("markov", {}).get("metrics", {})
        audit_path = ARTIFACT_DIR / "evaluation_audit.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            audit = {}
        _metric_grid([
            ("Classifier macro F1", f"{classifier.get('macro_f1', 0):.1%}", "Held-out test cases"),
            ("VAE anomaly F1", f"{vae.get('f1', 0):.1%}", "Held-out incident detection"),
            ("Full pipeline macro F1", f"{audit.get('category_macro_f1', 0):.1%}" if audit else "Not run", "Saved audit result"),
            ("Markov next-stage accuracy", f"{markov.get('next_stage_top1_accuracy', 0):.1%}", "Held-out transitions"),
        ])
        st.warning("Simulation-only results. Cases in each split share a generator and motifs; these values do not measure real-world detection quality.")
        with st.expander("Evaluation details"):
            st.json({"classifier_metrics": {key: classifier.get(key) for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "multiclass_log_loss", "sequence_count", "classes")}, "vae_metrics": vae, "markov_metrics": markov, "end_to_end_audit": {key: audit.get(key) for key in ("case_count", "category_accuracy", "category_macro_f1", "incident_precision", "incident_recall", "incident_f1")}})
            matrix = classifier.get("confusion_matrix")
            classes = classifier.get("classes")
            if matrix and classes:
                short_labels = {"Normal Activity": "Normal", "Account Compromise": "Account", "Privilege Escalation": "Privilege", "Suspicious Process Execution": "Process", "Data Exfiltration": "Exfiltration", "Denial-of-Service-like Activity": "DoS-like", "Multi-stage Attack": "Multi-stage"}
                labels = [short_labels.get(label, label) for label in classes]
                confusion = px.imshow(matrix, x=labels, y=labels, text_auto=True, labels={"x": "Predicted class", "y": "True class", "color": "Cases"}, color_continuous_scale=["#19314a", "#55cfcb"], aspect="auto")
                confusion.update_layout(title="Classifier confusion matrix · held-out simulation", height=440, margin={"l": 16, "r": 20, "t": 52, "b": 22}, paper_bgcolor="#14263c", plot_bgcolor="#14263c", font={"size": 13, "color": "#dceaf4"})
                st.plotly_chart(confusion, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("No training summary found yet. The README includes the short CPU training command and optional curated-data arguments.")
    _section("Model assumptions", "Bayesian root-cause structure")
    st.code("RootCause ──▶ UnusualAuthentication\n          ├─▶ PrivilegeChange\n          ├─▶ SuspiciousProcess\n          ├─▶ ExternalCommunication\n          ├─▶ LargeTransfer\n          ├─▶ EvidenceRemoval\n          └─▶ RepeatedAuthFailures", language="text")
    st.caption("Conditional likelihoods are elicited starting values in this academic prototype. The model updates the root-cause posterior by exact enumeration; validate its CPTs before drawing empirical conclusions.")


st.sidebar.markdown(
    '<div class="trace-brand"><span class="trace-brand-mark">T</span>'
    '<div><strong>TRACE</strong><small>Understand your system activity</small></div></div>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Refresh analysis tools", width="stretch", help="Use this after training new models while the dashboard is open."):
    cached_models.clear()
    st.rerun()

events = _get_events()
if events is None:
    _show_welcome()
    st.stop()

first_open = not st.session_state.get("trace_report_opened", False)
loading_panel = st.empty() if first_open else None
if loading_panel is not None:
    loading_panel.markdown(
        '<div class="trace-loading"><span>OPENING YOUR INVESTIGATION</span>'
        '<strong>Putting the activity in order…</strong>'
        '<p>This can take a few seconds the first time the analysis tools load.</p></div>',
        unsafe_allow_html=True,
    )
with st.spinner("Reading your activity records…"):
    try:
        models = cached_models()
        report = analyze_events(events, models)
    except (ValueError, KeyError, IndexError) as exc:
        if loading_panel is not None:
            loading_panel.empty()
        st.error(f"TRACE could not understand these records: {exc}")
        st.stop()
if loading_panel is not None:
    loading_panel.empty()
    st.session_state["trace_report_opened"] = True

loaded_count = sum(models.statuses.get(name) == "Loaded" for name in ("vae", "rbm", "classifier", "markov"))
st.sidebar.markdown('<div class="trace-sidebar-label">Analysis status</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="trace-sidebar-status"><span class="trace-sidebar-dot {"" if loaded_count == 4 else "partial"}"></span>'
    f'<span>{"Ready to review this case" if loaded_count == 4 else "Basic review is available"}</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("The analysis runs here on your device. Detailed methods are in Report & details.")
_show_case_header(report, events)
_show_decision(report)

overview_tab, timeline_tab, impact_tab, whatif_tab, report_tab = st.tabs(("Overview", "Activity timeline", "People & next steps", "Try a response", "Report & details"))
with overview_tab:
    _show_overview(report)
with timeline_tab:
    _show_timeline(report, events)
with impact_tab:
    _show_impact(report)
with whatif_tab:
    report["counterfactual"] = _show_what_if(events, models, report)
with report_tab:
    _section("KEEP A COPY", "Download your report", "Choose a readable report, the full data, or a cleaned copy of the records you supplied.")
    download_a, download_b, download_c = st.columns(3)
    report_name = f"trace_investigation_{report['incident_type'].lower().replace(' ', '_')}.html"
    download_a.download_button("Readable report", report_html(report), file_name=report_name, mime="text/html", width="stretch")
    download_b.download_button("Full data (JSON)", report_json(report), file_name=report_name.replace(".html", ".json"), mime="application/json", width="stretch")
    download_c.download_button("Activity records (CSV)", events.to_csv(index=False), file_name="trace_normalized_events.csv", mime="text/csv", width="stretch")
    st.markdown('<div class="trace-summary">TRACE combines patterns learned from example cases with clear rules tied to the records. The suggested steps are written from those rules. No chatbot or external service writes this result.</div>', unsafe_allow_html=True)
    with st.expander("See technical methods, model status, and test results"):
        _show_models(report)
        _section("INTERPRETATION", "Important limits")
        st.caption(f"Pattern comparison: {report['classifier_method']}. Unusual activity: {report['anomaly_method']}. Timeline: {report['markov_method']}.")
        for limitation in report["limitations"]:
            st.markdown(f"- {limitation}")
