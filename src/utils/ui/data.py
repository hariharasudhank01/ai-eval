"""
Shared DB access + small UI helpers for the AI-EVAL Streamlit app.
Every page imports from here rather than querying directly.
"""
import sys
import streamlit as st
from utils.helpers.logger import get_logger, timed

logger = get_logger("ui.data")

def get_connection():
    try:
        return st.connection("postgresql", type="sql")
    except Exception:
        logger.exception("Failed to connect to db")
        st.error("Could not connect to the database.")
        sys.exit(0)

def _query(sql, params=None, ttl="5m"):
    conn = get_connection()
    try:
        with timed(logger, f"Query: {sql.strip()[:60]}"):
            return conn.query(sql, params=params or {}, ttl=ttl)
    except Exception:
        logger.exception(f"Query failed: {sql}")
        st.error("A database query failed - check server logs.")
        return None

def load_generations():
    return _query("SELECT * FROM generations ORDER BY gen_id DESC;")

def load_iterations(gen_id):
    return _query(
        "SELECT * FROM iterations WHERE gen_id = :gen_id ORDER BY step_count;",
        {"gen_id": gen_id}
    )

def load_model_collapse(gen_id):
    return _query("""
        SELECT mc.*, i.step_count
        FROM model_collapse mc
        JOIN iterations i ON mc.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_datalose(gen_id):
    return _query("""
        SELECT dl.*, i.step_count
        FROM datalose dl
        JOIN iterations i ON dl.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_pii(gen_id):
    return _query("""
        SELECT p.*, i.step_count
        FROM pii p
        JOIN iterations i ON p.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_pii_findings(gen_id):
    return _query("""
        SELECT pef.finding_id, pef.text, pef.score, pef.source,
               pe.entity_type, p.iteration_id, i.step_count
        FROM pii_entity_findings pef
        JOIN pii_entities pe ON pef.entity_id = pe.entity_id
        JOIN pii p ON pef.pii_id = p.pii_id
        JOIN iterations i ON p.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_case_coverage(gen_id):
    return _query("""
        SELECT cc.*, i.step_count
        FROM case_coverage cc
        JOIN iterations i ON cc.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_toxicity(gen_id):
    return _query("""
        SELECT t.*, i.step_count
        FROM toxicity t
        JOIN iterations i ON t.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_toxicity_findings(gen_id):
    return _query("""
        SELECT tf.finding_id, tf.text, tf.score, tf.source,
               tc.category_name, t.iteration_id, i.step_count
        FROM toxicity_findings tf
        JOIN toxicity_categories tc ON tf.category_id = tc.category_id
        JOIN toxicity t ON tf.toxicity_id = t.toxicity_id
        JOIN iterations i ON t.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_distribution_drift(gen_id):
    return _query("""
        SELECT dd.*, i.step_count
        FROM distribution_drift dd
        JOIN iterations i ON dd.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def load_distribution_drift_summary(gen_id):
    return _query("""
        SELECT dds.*, i.step_count
        FROM distribution_drift_summary dds
        JOIN iterations i ON dds.iteration_id = i.iteration_id
        WHERE i.gen_id = :gen_id
        ORDER BY i.step_count;
    """, {"gen_id": gen_id})

def select_gen_id_sidebar():
    """Renders the generation picker in the sidebar (present on every page) and
    keeps the selection in st.session_state so it's shared across pages."""
    generations = load_generations()
    if generations is None or generations.empty:
        st.sidebar.warning("No generations found in the database yet.")
        return None

    options = generations["gen_id"].tolist()
    default_index = 0
    if "gen_id" in st.session_state and st.session_state["gen_id"] in options:
        default_index = options.index(st.session_state["gen_id"])

    gen_id = st.sidebar.selectbox(
        "Generation",
        options,
        index=default_index,
        key="gen_id_selector",
        format_func=lambda g: f"Gen {g}"
    )
    st.session_state["gen_id"] = gen_id
    return gen_id

def kpi_card(label, value, column=None):
    target = column if column is not None else st
    with target.container(border=True):
        st.metric(label, value)

def latest(df, column, default="-", fmt=None):
    """Grabs the most recent row's value for `column` from an iteration-ordered
    dataframe, or `default` if the dataframe is empty/missing."""
    if df is None or df.empty:
        return default
    value = df.iloc[-1][column]
    return fmt(value) if fmt else value
