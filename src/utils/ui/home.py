import altair as alt
import pandas as pd
import streamlit as st
from utils.ui.data import (
    select_gen_id_sidebar,
    load_iterations,
    load_model_collapse,
    load_datalose,
    load_pii,
    load_pii_findings,
    load_toxicity_findings,
    load_distribution_drift_summary,
    kpi_card,
    latest,
)

# Starting point - Title
st.set_page_config(page_title="AI Eval", layout="wide", page_icon="\U0001F9EA")
st.title("AI-EVAL")
st.caption("Executive Overview - high-level summary of AI-EVAL assessment results")

# List the available gen_id for the user to select and visualize
gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

# Load all the data for the selected gen_id from the DB
iterations_df = load_iterations(gen_id)
collapse_df = load_model_collapse(gen_id)
datalose_df = load_datalose(gen_id)
pii_df = load_pii(gen_id)
pii_findings_df = load_pii_findings(gen_id)
toxicity_findings_df = load_toxicity_findings(gen_id)
drift_summary_df = load_distribution_drift_summary(gen_id)

current_iteration = latest(iterations_df, "step_count")

# "Highest risk area" - a simple heuristic (worst normalised latest score), not a
# real threshold-based judgement. "Overall Status" / "First Failure Iteration" stay
# literal "Pending Threshold" placeholders until real pass/warning/fail cutoffs are
# defined - faking them would be worse than leaving them honest.
latest_scores = {}
if collapse_df is not None and not collapse_df.empty:
    latest_scores["Model Collapse"] = 1 - collapse_df.iloc[-1]["step_similarity"]
if datalose_df is not None and not datalose_df.empty:
    latest_scores["Data Loss"] = datalose_df.iloc[-1]["cumulative_loss"] / 100
if pii_df is not None and not pii_df.empty:
    latest_scores["PII Risk"] = pii_df.iloc[-1]["total_score"]
if drift_summary_df is not None and not drift_summary_df.empty:
    latest_scores["Distribution"] = drift_summary_df.iloc[-1]["distribution_drift_pct"] / 100
highest_risk_area = max(latest_scores, key=latest_scores.get) if latest_scores else "-"

critical_findings = (0 if pii_findings_df is None else len(pii_findings_df)) + \
                     (0 if toxicity_findings_df is None else len(toxicity_findings_df)) # Need to Optimise

# Top - KPI

col1, col2, col3, col4, col5 = st.columns(5)
kpi_card("Overall Status", "Pending Threshold", col1)
kpi_card("Current Iteration", current_iteration, col2)
kpi_card("First Failure Iteration", "Pending Threshold", col3)
kpi_card("Highest Risk Area", highest_risk_area, col4)
kpi_card("Critical Findings", critical_findings, col5)

st.divider()

# Chart

st.subheader("Generation Trend")
st.caption(
    "Tracks changes in evaluation metrics across iterations, including step "
    "similarity, baseline drift, data loss and PII risk. Final status "
    "interpretation will depend on agreed evaluation thresholds."
)

trend_frames = []
if collapse_df is not None and not collapse_df.empty:
    trend_frames.append(pd.DataFrame({
        "Iteration": collapse_df["step_count"],
        "Metric": "Step Similarity",
        "Value": collapse_df["step_similarity"],
    }))
    trend_frames.append(pd.DataFrame({
        "Iteration": collapse_df["step_count"],
        "Metric": "Baseline Drift",
        "Value": collapse_df["baseline_drift"] / 100,
    }))
if datalose_df is not None and not datalose_df.empty:
    trend_frames.append(pd.DataFrame({
        "Iteration": datalose_df["step_count"],
        "Metric": "Cumulative Data Loss",
        "Value": datalose_df["cumulative_loss"] / 100,
    }))
if pii_df is not None and not pii_df.empty:
    trend_frames.append(pd.DataFrame({
        "Iteration": pii_df["step_count"],
        "Metric": "PII Risk",
        "Value": pii_df["total_score"],
    }))

if trend_frames:
    trend_df = pd.concat(trend_frames, ignore_index=True)
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Iteration:O", title="Iteration"),
            y=alt.Y("Value:Q", title="Value (0-1 scale)"),
            color=alt.Color("Metric:N"),
            tooltip=["Iteration", "Metric", "Value"],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
else:
    st.info("No trend data available yet for this generation.")

st.divider()

# KPI Cards - Bottom

col1, col2, col3, col4 = st.columns(4)
with col1, st.container(border=True):
    st.caption("Model Collapse")
    st.metric("Step Similarity", latest(collapse_df, "step_similarity", fmt=lambda v: f"{v:.2f}"))
with col2, st.container(border=True):
    st.caption("PII Risk")
    st.metric("Total Score", latest(pii_df, "total_score", fmt=lambda v: f"{v:.2f}"))
with col3, st.container(border=True):
    st.caption("Data Loss")
    st.metric("Cumulative Loss", latest(datalose_df, "cumulative_loss", fmt=lambda v: f"{v:.1f}%"))
with col4, st.container(border=True):
    st.caption("Distribution")
    st.metric("Distribution Drift", latest(drift_summary_df, "distribution_drift_pct", fmt=lambda v: f"{v:.1f}%"))
