# WIP
import streamlit as st
from utils.db import select
import pandas as pd

@st.cache_data(ttl="10m")
def load_data(session_local, gen_id):
    return select.fetch_test_results(
        session_local,
        gen_id
    )

def generate_report(engine, gen_id):
    session_local = select.initiate_sessionlocal(engine)
    results = load_data(session_local, gen_id)