# WIP
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="AI Eval", layout="wide")
st.title("AI Eval")

def load_data():
    conn = st.connection("postgresql", type="sql")
    df = conn.query("SELECT * from test_results;", ttl="10m")
    pivoted_df = df.pivot(index="iteration_id", columns="metric_name", values="value")
    st.dataframe(pivoted_df)
    return df

def altair_chart(df):
    col1, col2 = st.columns(2)
    with col1:
        gen_id = st.selectbox(
            "Select Gen ID",
            df["gen_id"].unique(),
            index=None,
            placeholder="Select ID",
        )
        st.write("Gen ID:", gen_id)

    filtered_df = df[df["gen_id"]==gen_id]
    chart = (
        alt.Chart(filtered_df)
        .mark_boxplot()
        .encode(x="metric_name", y="value", color="metric_name")
        .properties(width="container")
    ).interactive()
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

def line_charts_by_test_type(df):
    gen_id = st.selectbox(
        "Select Gen ID for trend view",
        df["gen_id"].unique(),
        index=None,
        placeholder="Select ID",
        key="trend_gen_id",
    )
    if gen_id is None:
        return

    filtered_df = df[df["gen_id"] == gen_id]
    test_types = sorted(filtered_df["test_type"].unique())
    tabs = st.tabs(test_types)

    for tab, test_type in zip(tabs, test_types):
        with tab:
            test_type_df = filtered_df[filtered_df["test_type"] == test_type]
            chart = (
                alt.Chart(test_type_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("iteration_id", type="quantitative", title="Iteration"),
                    y=alt.Y("value", type="quantitative"),
                    color=alt.Color("metric_name", type="nominal"),
                    tooltip=["iteration_id", "metric_name", "value"],
                )
                .properties(width="container", height=350, title=f"{test_type} trend by iteration")
                .interactive()
            )
            st.altair_chart(chart, theme="streamlit", use_container_width=True)

def report():
    df = load_data()
    altair_chart(df)
    line_charts_by_test_type(df)

report()

