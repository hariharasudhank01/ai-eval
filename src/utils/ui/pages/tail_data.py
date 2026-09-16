import os
import altair as alt
import pandas as pd
import streamlit as st
from utils.ui.theme import inject_theme
from utils.helpers.logger import LOG_DIR
from utils.ui.data import select_gen_id_sidebar, load_datalose, kpi_card, latest

inject_theme()
st.title("Tail Data Analysis")
st.caption(
    "This page isn't in the original design (it was left blank there too) - built "
    "here from what tail_data_lose.py actually produces: HDBSCAN-based outlier "
    "ratio, diversity loss vs. baseline/previous iteration, and the cluster plot."
)

gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

datalose_df = load_datalose(gen_id)

if datalose_df is None or datalose_df.empty:
    st.info("No tail data results yet for this generation (needs at least 2 iterations).")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
kpi_card("Cumulative Data Loss", latest(datalose_df, "cumulative_loss", fmt=lambda v: f"{v:.1f}%"), col1)
kpi_card("Step Data Loss", latest(datalose_df, "diversity_loss_from_previous_iteration", fmt=lambda v: f"{v:.1f}%"), col2)
kpi_card("Outlier Ratio", latest(datalose_df, "outlier_ratio", fmt=lambda v: f"{v:.1f}%"), col3)
kpi_card("Diversity Loss vs Baseline", latest(datalose_df, "diversity_loss_from_baseline", fmt=lambda v: f"{v:.1f}%"), col4)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Data Loss Trend Across Iterations")
    trend_df = pd.concat([
        pd.DataFrame({"Iteration": datalose_df["step_count"], "Metric": "Cumulative Loss", "Value": datalose_df["cumulative_loss"]}),
        pd.DataFrame({"Iteration": datalose_df["step_count"], "Metric": "Step Loss", "Value": datalose_df["diversity_loss_from_previous_iteration"]}),
        pd.DataFrame({"Iteration": datalose_df["step_count"], "Metric": "Outlier Ratio", "Value": datalose_df["outlier_ratio"]}),
    ])
    chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Iteration:O", title="Iteration"),
            y=alt.Y("Value:Q", title="% "),
            color=alt.Color("Metric:N"),
            tooltip=["Iteration", "Metric", "Value"],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

with right:
    st.subheader("Cluster View")
    cluster_path = os.path.join(LOG_DIR, f"gen{gen_id}_clusters.png")
    if os.path.exists(cluster_path):
        st.image(cluster_path, caption=f"HDBSCAN clusters for gen {gen_id} (PCA projection)", width="stretch")
    else:
        st.info("No cluster plot saved for this generation yet.")

st.divider()
st.subheader("Detail")
detail = datalose_df[["step_count", "cumulative_loss", "diversity_loss_from_previous_iteration", "outlier_ratio", "diversity_loss_from_baseline"]].copy()
detail.columns = ["Iteration", "Cumulative Loss (%)", "Step Loss (%)", "Outlier Ratio (%)", "Diversity Loss vs Baseline (%)"]
st.dataframe(detail, hide_index=True, width="stretch")
