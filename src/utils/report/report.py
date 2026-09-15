# WIP
import streamlit as st
import pandas as pd
import altair as alt
import sys
from utils.helpers.logger import get_logger, timed

logger = get_logger("report")

st.set_page_config(page_title="AI Eval", layout="wide")
st.title("AI Eval")

def create_connection():
    try:
        return st.connection("postgresql", type="sql")
    except Exception:
        logger.exception("Failed to connect to db")
        sys.exit(0)

def load_iteration_details(conn):
    try:
        with timed(logger, "Load iteration details"):
            return conn.query("SELECT * from iterations;", ttl="10m")
    except Exception:
        logger.exception("Failed to fetch details from iterations table")
        sys.exit(0)

def load_test_results(conn):
    try:
        with timed(logger, "Load test results"):
            return conn.query("SELECT * from test_results;", ttl="10m")
    except Exception:
        logger.exception("Failed to fetch details from results table")
        sys.exit(0)

def load_generation_details(conn):
    try:
        with timed(logger, "Load generation details"):
            return conn.query("SELECT * from generations;", ttl="10m")
    except Exception:
        logger.exception("Failed to fetch details from generations table")
        sys.exit(0)

def display_gen_details(gen_df):
    col1, col2 = st.columns(2)

    with col1:
        gen_id = st.selectbox(
            "Select Gen ID to view",
            gen_df["gen_id"].unique(),
            index=None,
            placeholder="Select ID",
            key="display_gen_id"
        )

    df = gen_df[gen_df["gen_id"] == gen_id]

    with col2:
        if gen_id:
            info_msg = "Gen ID: " + str(gen_id) + "\n Test Start Time: " + str(df["start_time"]) + "\n Test End Time: " + str(df["end_time"])
            st.info(info_msg)

    return gen_id

def display_iteration_ids(gen_id, iteration_df):
    col1, col2 = st.columns(2)

    itera_df = iteration_df[iteration_df["gen_id"] == gen_id]
    with col1:
        iteration_id = st.selectbox(
            "Select Iteration ID to view",
            itera_df["iteration_id"].unique(),
            index=None,
            placeholder="Select ID",
            key="display_iteration_id"
        )

    with col2:
        if iteration_id:
            info_msg = "Number of Iterations: " + str(len(itera_df["step_count"]))
            st.info(info_msg)


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
    #df = load_data()
    #altair_chart(df)
    #line_charts_by_test_type(df)
    conn = create_connection()
    gen_df = load_generation_details(conn)
    gen_id = display_gen_details(gen_df)
    display_iteration_ids(gen_id, iteration_df=load_iteration_details(conn))


report()

