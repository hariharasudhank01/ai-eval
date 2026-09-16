import pandas as pd
import streamlit as st
from utils.ui.theme import inject_theme
from utils.ui.data import select_gen_id_sidebar, load_pii_findings, load_toxicity_findings

inject_theme()
st.title("Evidence")
st.caption("View detailed evidence for flagged findings across all evaluation dimensions.")

gen_id = select_gen_id_sidebar()
if gen_id is None:
    st.stop()

pii_df = load_pii_findings(gen_id)
toxicity_df = load_toxicity_findings(gen_id)

rows = []
if pii_df is not None and not pii_df.empty:
    for i, row in pii_df.iterrows():
        rows.append({
            "ID": f"PII-{row['finding_id']:02d}",
            "Type": "PII",
            "Iteration": row["step_count"],
            "Finding": f"{row['entity_type']} detected",
            "Detected Text": row["text"],
            "Source": row["source"],
            "Score": row["score"],
            "Result": "Flagged",
        })
if toxicity_df is not None and not toxicity_df.empty:
    for i, row in toxicity_df.iterrows():
        rows.append({
            "ID": f"TOX-{row['finding_id']:02d}",
            "Type": "Toxicity",
            "Iteration": row["step_count"],
            "Finding": f"{row['category_name']} detected",
            "Detected Text": row["text"],
            "Source": row["source"],
            "Score": row["score"],
            "Result": "Flagged",
        })

if not rows:
    st.info("No findings recorded for this generation yet.")
    st.stop()

evidence_df = pd.DataFrame(rows)

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    iteration_filter = st.selectbox("Iteration", ["All"] + sorted(evidence_df["Iteration"].unique().tolist()))
with col2:
    type_filter = st.selectbox("Evidence Type", ["All"] + sorted(evidence_df["Type"].unique().tolist()))
with col3:
    result_filter = st.selectbox("Result", ["All"] + sorted(evidence_df["Result"].unique().tolist()))

filtered_df = evidence_df.copy()
if iteration_filter != "All":
    filtered_df = filtered_df[filtered_df["Iteration"] == iteration_filter]
if type_filter != "All":
    filtered_df = filtered_df[filtered_df["Type"] == type_filter]
if result_filter != "All":
    filtered_df = filtered_df[filtered_df["Result"] == result_filter]

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader(f"Evidence List ({len(filtered_df)} findings)")
    st.dataframe(
        filtered_df[["ID", "Type", "Iteration", "Finding", "Result"]],
        hide_index=True,
        width="stretch",
        height=420,
    )

with right:
    st.subheader("Finding Detail")
    if filtered_df.empty:
        st.info("No findings match the current filters.")
    else:
        ids = filtered_df["ID"].tolist()
        default_id = st.session_state.get("evidence_selected_id")
        default_index = ids.index(default_id) if default_id in ids else 0
        selected_id = st.selectbox("Finding ID", ids, index=default_index, key="evidence_selected_id")

        detail_row = filtered_df[filtered_df["ID"] == selected_id].iloc[0]
        st.markdown(f"### {detail_row['ID']}  \n**{detail_row['Result']}**")
        st.caption(f"{detail_row['Type']} | Iteration {detail_row['Iteration']} | {detail_row['Finding']}")

        st.markdown("**Detected Text**")
        st.code(detail_row["Detected Text"])

        st.markdown("**Detection Details**")
        st.markdown(f"- Type: {detail_row['Type']}")
        st.markdown(f"- Detection Source(s): {detail_row['Source']}")
        st.markdown(f"- Confidence Score: {detail_row['Score']:.2f}")

        # Iteration trace: does this same finding text/type show up across other iterations too
        st.markdown("**Present in these iterations**")
        same_finding = evidence_df[
            (evidence_df["Type"] == detail_row["Type"]) & (evidence_df["Finding"] == detail_row["Finding"])
        ]
        trace_iterations = sorted(same_finding["Iteration"].unique().tolist())
        st.write(", ".join(f"I{i}" for i in trace_iterations))
