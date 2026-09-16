import altair as alt
import pandas as pd
import streamlit as st
from utils.ui.theme import inject_theme
from utils.ui.data import (
    select_gen_id_sidebar,
    load_distribution_drift,
    load_distribution_drift_summary,
    kpi_card,
)

inject_theme()
st.title("Distribution Analysis")
st.caption("Compare data distribution across iterations to identify shifts, bias amplification, or loss of minority representation.")

gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

drift_df = load_distribution_drift(gen_id)
summary_df = load_distribution_drift_summary(gen_id)

if summary_df is None or summary_df.empty:
    st.info("No distribution drift results yet for this generation.")
    st.stop()

available_steps = sorted(summary_df["step_count"].unique().tolist())
selected_step = st.selectbox("Select iteration", available_steps, index=len(available_steps) - 1)

summary_row = summary_df[summary_df["step_count"] == selected_step].iloc[0]
category_rows = drift_df[drift_df["step_count"] == selected_step] if drift_df is not None else pd.DataFrame()

col1, col2, col3, col4, col5 = st.columns(5)
kpi_card("Distribution Drift", f"{summary_row['distribution_drift_pct']:.1f}%", col1)
kpi_card("Current Iteration", f"Iteration {selected_step}", col2)
kpi_card("Largest Shift", summary_row["largest_shift_category"], col3)
kpi_card("Minority Retention", f"{summary_row['minority_retention_pct']:.0f}%", col4)
kpi_card("Status", "Pending Threshold", col5)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader(f"Category Distribution - Baseline vs Iteration {selected_step}")
    if not category_rows.empty:
        long_df = pd.concat([
            pd.DataFrame({"Category": category_rows["category_label"], "Series": "Baseline", "Proportion": category_rows["baseline_proportion_pct"]}),
            pd.DataFrame({"Category": category_rows["category_label"], "Series": f"Iteration {selected_step}", "Proportion": category_rows["current_proportion_pct"]}),
        ])
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                x=alt.X("Category:N"),
                y=alt.Y("Proportion:Q", title="Proportion (%)"),
                color=alt.Color("Series:N"),
                xOffset="Series:N",
                tooltip=["Category", "Series", "Proportion"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No category data for this iteration.")

with right:
    st.subheader("Key Distribution Changes")
    if not category_rows.empty:
        changes = category_rows.copy()
        changes["change_pct"] = changes["current_proportion_pct"] - changes["baseline_proportion_pct"]
        changes = changes.reindex(changes["change_pct"].abs().sort_values(ascending=False).index)
        for _, row in changes.head(4).iterrows():
            direction = "increased" if row["change_pct"] >= 0 else "decreased"
            st.markdown(
                f"- **{row['category_label']}** {direction} from "
                f"{row['baseline_proportion_pct']:.0f}% to {row['current_proportion_pct']:.0f}% "
                f"({row['change_pct']:+.0f} pts)"
            )

st.divider()

st.subheader("Distribution Drift Across Iterations")
if summary_df is not None and not summary_df.empty:
    chart = (
        alt.Chart(summary_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("step_count:O", title="Iteration"),
            y=alt.Y("distribution_drift_pct:Q", title="Distribution Drift (%)"),
            tooltip=["step_count", "distribution_drift_pct"],
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

st.subheader(f"Category Distribution Detail (Baseline vs Iteration {selected_step})")
if not category_rows.empty:
    detail = category_rows[["category_label", "baseline_proportion_pct", "current_proportion_pct"]].copy()
    detail["Change (pts)"] = detail["current_proportion_pct"] - detail["baseline_proportion_pct"]
    detail.columns = ["Category", "Baseline (%)", f"Iteration {selected_step} (%)", "Change (pts)"]
    st.dataframe(detail, hide_index=True, width="stretch")
