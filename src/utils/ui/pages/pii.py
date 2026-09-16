import altair as alt
import pandas as pd
import streamlit as st
from utils.ui.data import (
    select_gen_id_sidebar,
    load_pii,
    load_pii_findings,
    kpi_card,
    latest,
)

st.set_page_config(page_title="AI Eval - PII", layout="wide")
st.title("PII Analysis")
st.caption("Evaluate the presence and handling of personal identifiable information (PII) across iterations.")

gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

pii_df = load_pii(gen_id)
findings_df = load_pii_findings(gen_id)

entity_types = findings_df["entity_type"].nunique() if findings_df is not None and not findings_df.empty else 0
top_entity = (
    findings_df["entity_type"].value_counts().idxmax()
    if findings_df is not None and not findings_df.empty
    else "-"
)
findings_count = 0 if findings_df is None else len(findings_df)

col1, col2, col3, col4 = st.columns(4)
kpi_card("PII Risk Score", latest(pii_df, "total_score", fmt=lambda v: f"{v:.2f}"), col1)
kpi_card("Detected Entity Types", entity_types, col2)
kpi_card("Highest Risk Entity", top_entity, col3)
kpi_card("PII Findings Count", findings_count, col4)

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("PII Risk Score Across Iterations")
    if pii_df is not None and not pii_df.empty:
        chart = (
            alt.Chart(pii_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("step_count:O", title="Iteration"),
                y=alt.Y("total_score:Q", title="PII Risk Score"),
                tooltip=["step_count", "total_score"],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No PII results yet for this generation.")

with right:
    st.subheader("Entity Type Summary")
    if findings_df is not None and not findings_df.empty:
        counts = findings_df["entity_type"].value_counts().reset_index()
        counts.columns = ["Entity Type", "Count"]
        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q"),
                y=alt.Y("Entity Type:N", sort="-x"),
                tooltip=["Entity Type", "Count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No findings yet.")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("PII Findings")
    if findings_df is not None and not findings_df.empty:
        entity_filter = st.multiselect("Entity Type", sorted(findings_df["entity_type"].unique()))
        display_df = findings_df if not entity_filter else findings_df[findings_df["entity_type"].isin(entity_filter)]
        st.dataframe(
            display_df[["step_count", "entity_type", "text", "source", "score"]].rename(columns={
                "step_count": "Iteration", "entity_type": "Entity Type", "text": "Detected Text",
                "source": "Detection Source", "score": "Score",
            }),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No findings to display.")

with right:
    st.subheader("Evidence Detail")
    if findings_df is not None and not findings_df.empty:
        idx = st.number_input("Finding #", min_value=1, max_value=len(findings_df), value=1, step=1) - 1
        row = findings_df.iloc[idx]
        st.caption(f"{idx + 1} / {len(findings_df)}")
        st.markdown(f"**Detected Text**")
        st.code(row["text"])
        st.markdown(f"**Entity Type:** {row['entity_type']}")
        st.markdown(f"**Iteration:** {row['step_count']}")
        st.markdown(f"**Detection Source:** {row['source']}")
        st.markdown(f"**Confidence Score:** {row['score']:.2f}")
    else:
        st.info("Nothing to show.")
