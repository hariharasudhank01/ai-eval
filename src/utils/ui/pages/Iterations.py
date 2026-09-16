import altair as alt
import pandas as pd
import streamlit as st
from utils.ui.data import (
    select_gen_id_sidebar,
    load_model_collapse,
    load_datalose,
    load_pii,
    kpi_card,
)

st.set_page_config(page_title="AI Eval - Iterations", layout="wide")
st.title("Iteration Analysis")
st.caption("Compare evaluation results for a selected iteration against its relevant baseline or previous iteration.")

gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

collapse_df = load_model_collapse(gen_id)
datalose_df = load_datalose(gen_id)
pii_df = load_pii(gen_id)

if collapse_df is None or collapse_df.empty:
    st.info("No iteration results yet for this generation (model_collapse/tail_data_lose need at least 2 iterations).")
    st.stop()

available_steps = sorted(collapse_df["step_count"].unique().tolist())
selected_step = st.selectbox("Select iteration", available_steps, index=len(available_steps) - 1)

collapse_row = collapse_df[collapse_df["step_count"] == selected_step].iloc[0]
datalose_row = datalose_df[datalose_df["step_count"] == selected_step].iloc[0] if datalose_df is not None and not datalose_df.empty and selected_step in datalose_df["step_count"].values else None
pii_row = pii_df[pii_df["step_count"] == selected_step].iloc[0] if pii_df is not None and not pii_df.empty and selected_step in pii_df["step_count"].values else None

col1, col2, col3, col4 = st.columns(4)
kpi_card("Status", "Pending Threshold", col1)
kpi_card("Baseline Drift", f"{collapse_row['baseline_drift']:.1f}%", col2)
kpi_card("Step Similarity", f"{collapse_row['step_similarity']:.2f}", col3)
kpi_card("Cumulative Data Loss", f"{datalose_row['cumulative_loss']:.1f}%" if datalose_row is not None else "-", col4)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Iteration Comparison")
    comparison_rows = [
        {"Metric": "Step Similarity", "Comparison Basis": "Previous Iteration", "Value": f"{collapse_row['step_similarity']:.2f}"},
        {"Metric": "Baseline Drift", "Comparison Basis": "Baseline", "Value": f"{collapse_row['baseline_drift']:.1f}%"},
    ]
    if datalose_row is not None:
        comparison_rows.append({"Metric": "Cumulative Data Loss", "Comparison Basis": "Baseline", "Value": f"{datalose_row['cumulative_loss']:.1f}%"})
        comparison_rows.append({"Metric": "Step Data Loss", "Comparison Basis": "Previous Iteration", "Value": f"{datalose_row['diversity_loss_from_previous_iteration']:.1f}%"})
    if pii_row is not None:
        comparison_rows.append({"Metric": "PII Risk Score", "Comparison Basis": "Current Iteration", "Value": f"{pii_row['total_score']:.2f}"})
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")

with right:
    st.subheader("Key Changes")
    st.markdown(f"- Baseline drift reached **{collapse_row['baseline_drift']:.1f}%**")
    st.markdown(f"- Step similarity with the previous iteration is **{collapse_row['step_similarity']:.2f}**")
    if datalose_row is not None:
        st.markdown(f"- Cumulative data loss reached **{datalose_row['cumulative_loss']:.1f}%**")
    if pii_row is not None:
        st.markdown(f"- PII risk score reached **{pii_row['total_score']:.2f}**")

    st.subheader("Evidence Summary")
    for label in ("Model Collapse", "PII", "Data Loss", "Distribution"):
        st.markdown(f"- {label}")
    if st.button("View Detailed Evidence →"):
        st.session_state["evidence_step_filter"] = selected_step
        st.switch_page("pages/5_Evidence.py")

st.divider()
st.subheader("Metric Trend Across Iterations")
st.caption(
    "Step Similarity compares each iteration with the previous iteration. "
    "Baseline Drift compares against the baseline. Drift and loss percentages "
    "are represented as decimal proportions."
)

trend_frames = [
    pd.DataFrame({"Iteration": collapse_df["step_count"], "Metric": "Step Similarity (vs Previous)", "Value": collapse_df["step_similarity"]}),
    pd.DataFrame({"Iteration": collapse_df["step_count"], "Metric": "Baseline Drift (vs Baseline)", "Value": collapse_df["baseline_drift"] / 100}),
]
if datalose_df is not None and not datalose_df.empty:
    trend_frames.append(pd.DataFrame({"Iteration": datalose_df["step_count"], "Metric": "Cumulative Data Loss", "Value": datalose_df["cumulative_loss"] / 100}))
if pii_df is not None and not pii_df.empty:
    trend_frames.append(pd.DataFrame({"Iteration": pii_df["step_count"], "Metric": "PII Risk Score", "Value": pii_df["total_score"]}))

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
