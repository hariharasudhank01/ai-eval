import os
import streamlit as st
import requests
from utils.helpers.logger import get_logger, timed
from utils.ui.theme import inject_theme, hero

logger = get_logger("ui.new_run")

API_BASE_URL = os.getenv("AI_EVAL_API_URL", "http://localhost:8000")

inject_theme()
hero(
    "Configure a new evaluation run",
    "Point AI-EVAL at a model and a source file, and it will generate synthetic "
    "data across as many iterations as you need - then score it for drift, PII "
    "exposure, toxicity and distribution shift.",
)

PROVIDER_MODEL_HINTS = {
    "ollama": ("llama3.2:3b", "Any model already pulled in your Ollama instance, e.g. llama3.2:3b, qwen3:8b"),
    "openai": ("gpt-4o-mini", "Any OpenAI model id, e.g. gpt-4o-mini, gpt-4o"),
    "anthropic": ("claude-sonnet-5", "Any Anthropic model id, e.g. claude-sonnet-5, claude-haiku-4-5"),
}

with st.container(border=True):
    st.subheader("Model")
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", list(PROVIDER_MODEL_HINTS.keys()))
    with col2:
        default_model, hint = PROVIDER_MODEL_HINTS[provider]
        #model:"llama3.2:3b" hint:"Any model already pulled in your Ollama instance, e.g. llama3.2:3b, qwen3:8b"
        model = st.text_input("Model", value=default_model, help=hint)

st.write("")

with st.container(border=True):
    st.subheader("Source & Prompt")
    source_file = st.file_uploader("Source file", type=["txt", "pdf"])
    prompt = st.text_area(
        "Prompt",
        height=140,
        placeholder="Instruction for how the model should generate synthetic data from the source file...",
    )
    iteration = st.number_input("Iterations", min_value=1, value=3, step=1)

st.write("")

run_clicked = st.button("Run Evaluation", type="primary", disabled=not (source_file and prompt.strip()))

st.divider()

if run_clicked:
    files = {"source_file": (source_file.name, source_file.getvalue())}
    data = {"prompt": prompt, "model": model, "provider": provider, "iteration": int(iteration)}

    with st.spinner("Running evaluation - this can take a while depending on iterations and model speed..."):
        try:
            with timed(logger, "POST /runs"):
                response = requests.post(f"{API_BASE_URL}/runs", data=data, files=files, timeout=3600)
        except requests.RequestException:
            logger.exception("Failed to reach AI Eval API")
            st.error(f"Could not reach the AI Eval API at {API_BASE_URL}. Is it running?")
            response = None

    if response is not None:
        if response.ok:
            gen_id = response.json()["gen_id"]
            logger.info(f"Run succeeded, gen_id={gen_id}")
            st.session_state["gen_id"] = gen_id
            # The generations list on other pages is cached (ttl) - clear it so
            # this new gen_id shows up immediately instead of after the cache expires.
            st.cache_data.clear()
            st.success(f"Run complete - generation {gen_id} created.")
            if st.button(f"View report for Gen {gen_id} →"):
                st.switch_page("pages/overview.py")

            with st.expander("Run logs"):
                try:
                    logs_response = requests.get(f"{API_BASE_URL}/runs/{gen_id}/logs", timeout=30)
                    if logs_response.ok:
                        st.code(logs_response.text)
                    else:
                        st.info("Logs not available yet for this run.")
                except requests.RequestException:
                    logger.exception("Failed to fetch run logs")
                    st.info("Could not fetch run logs.")
        else:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            logger.error(f"Run failed: {response.status_code} {detail}")
            st.error(f"Run failed: {detail}")
