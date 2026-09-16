"""
App entry point / navigation router. Owns the one st.set_page_config() call
for the whole app; each page below only renders its own content (no more
per-page set_page_config, since Streamlit only allows one per run).
"""
import streamlit as st

st.set_page_config(page_title="AI Eval", layout="wide", page_icon="\U0001F9EA")

pages = [
    st.Page("pages/new_run.py", title="Home", default=True),
    st.Page("pages/overview.py", title="Overview"),
    st.Page("pages/distribution.py", title="Distribution"),
    st.Page("pages/pii.py", title="PII"),
    st.Page("pages/tail_data.py", title="Tail Data"),
    st.Page("pages/Iterations.py", title="Iteration"),
    st.Page("pages/evidence.py", title="Evidence"),
]

st.navigation(pages).run()
