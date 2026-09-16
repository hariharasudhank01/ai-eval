"""
Sidebar chat assistant that explains a run's AI-EVAL results in plain language.

Scoped to the currently selected gen_id, grounded only in the same data the
report pages already show (metrics + findings loaded via utils.ui.data), and
answered by the same fixed judge model used for PII/toxicity evaluation -
never the model under test, and never a network call the user didn't already
trigger by loading the report.
"""
import streamlit as st
from pydantic import BaseModel
from utils.helpers.logger import get_logger, timed
from utils.llm.llm import JUDGE_PROVIDER, JUDGE_MODEL
from utils.llm.providers import get_provider
from utils.ui.data import (
    load_iterations,
    load_model_collapse,
    load_datalose,
    load_pii,
    load_pii_findings,
    load_case_coverage,
    load_toxicity,
    load_toxicity_findings,
    load_distribution_drift,
    load_distribution_drift_summary,
)

logger = get_logger("ui.chatbot")

SYSTEM_PROMPT = """
Role:
- You are the AI-EVAL assistant, explaining synthetic-data evaluation results
  to the user in clear, plain language.
Task:
- Answer the user's question using ONLY the run summary provided below. It
  contains the same metrics and findings already shown on the report pages.
Rules:
- Do not invent numbers, findings, or iterations that are not in the summary.
- If the summary doesn't contain enough information to answer, say so plainly
  instead of guessing.
- Be concise. Prefer a few sentences or a short list over long paragraphs.
"""

class ChatReply(BaseModel):
    answer: str

def _fmt_rows(df, columns, limit=None):
    if df is None or df.empty:
        return "  (none)"
    rows = df[columns] if columns else df
    if limit:
        rows = rows.tail(limit)
    return "\n".join(f"  - {row.to_dict()}" for _, row in rows.iterrows())

def build_context(gen_id):
    """Builds a plain-text summary of gen_id's results from the exact same
    loaders the report pages use - the chat model sees nothing beyond what's
    already on screen."""
    iterations_df = load_iterations(gen_id)
    collapse_df = load_model_collapse(gen_id)
    datalose_df = load_datalose(gen_id)
    pii_df = load_pii(gen_id)
    pii_findings_df = load_pii_findings(gen_id)
    coverage_df = load_case_coverage(gen_id)
    toxicity_df = load_toxicity(gen_id)
    toxicity_findings_df = load_toxicity_findings(gen_id)
    drift_df = load_distribution_drift(gen_id)
    drift_summary_df = load_distribution_drift_summary(gen_id)

    iteration_count = 0 if iterations_df is None else len(iterations_df)

    parts = [f"Generation gen_id={gen_id}, {iteration_count} iteration(s) completed.\n"]

    parts.append("Model Collapse (step_similarity vs previous, baseline_drift % vs baseline):")
    parts.append(_fmt_rows(collapse_df, ["step_count", "step_similarity", "baseline_drift"]))

    parts.append("\nData Loss (% loss / outlier ratio per iteration):")
    parts.append(_fmt_rows(datalose_df, ["step_count", "cumulative_loss", "outlier_ratio", "diversity_loss_from_baseline"]))

    parts.append("\nPII Risk Score per iteration:")
    parts.append(_fmt_rows(pii_df, ["step_count", "total_score"]))
    parts.append("PII findings (entity type, source, score):")
    parts.append(_fmt_rows(pii_findings_df, ["step_count", "entity_type", "source", "score"], limit=20))

    parts.append("\nToxicity Score per iteration:")
    parts.append(_fmt_rows(toxicity_df, ["step_count", "total_score"]))
    parts.append("Toxicity findings (category, source, score):")
    parts.append(_fmt_rows(toxicity_findings_df, ["step_count", "category_name", "source", "score"], limit=20))

    parts.append("\nCase weightage distribution per iteration:")
    parts.append(_fmt_rows(coverage_df, ["step_count", "case_label", "weightage_pct"], limit=30))

    parts.append("\nDistribution drift summary per iteration:")
    parts.append(_fmt_rows(drift_summary_df, [
        "step_count", "distribution_drift_pct", "largest_shift_category",
        "largest_shift_pct", "minority_category", "minority_retention_pct",
    ]))
    parts.append("Category-level distribution drift:")
    parts.append(_fmt_rows(drift_df, ["step_count", "category_label", "baseline_proportion_pct", "current_proportion_pct"], limit=30))

    return "\n".join(parts)

def ask(gen_id, question, history):
    context = build_context(gen_id)
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history[-6:])
    user_prompt = f"Run summary:\n{context}\n\nConversation so far:\n{history_text}\n\nUser question: {question}"

    try:
        with timed(logger, f"Chat answer for gen_id={gen_id}"):
            reply = get_provider(JUDGE_PROVIDER).generate(SYSTEM_PROMPT, user_prompt, JUDGE_MODEL, ChatReply)
        return reply.answer
    except Exception:
        logger.exception(f"Chat answer failed for gen_id={gen_id}")
        return "Sorry, I couldn't generate an answer just now - check the server logs for details."

def render_chat_sidebar():
    """Renders the chat assistant in the sidebar on every page. Only appears
    once a gen_id is selected/known, and resets its history whenever the
    selected gen_id changes so answers never bleed across runs."""
    gen_id = st.session_state.get("gen_id")
    if gen_id is None:
        return

    if st.session_state.get("chat_gen_id") != gen_id:
        st.session_state["chat_gen_id"] = gen_id
        st.session_state["chat_history"] = []

    with st.sidebar:
        with st.expander(f"\U0001F4AC Ask about Gen {gen_id}", expanded=False):
            for turn in st.session_state["chat_history"]:
                with st.chat_message(turn["role"]):
                    st.write(turn["content"])

            question = st.chat_input("Ask about this run's results", key=f"chat_input_{gen_id}")
            if question:
                st.session_state["chat_history"].append({"role": "user", "content": question})
                with st.spinner("Thinking..."):
                    answer = ask(gen_id, question, st.session_state["chat_history"])
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                st.rerun()
