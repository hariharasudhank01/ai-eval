"""
Shared visual theme for the AI-EVAL Streamlit app - inspired by indigitise.com.au's
enterprise-SaaS aesthetic (dark, confident, minimal, generous whitespace, a single
accent color used sparingly). Every page calls inject_theme() once, right after
st.set_page_config(), before rendering anything else.

This only injects CSS (via Streamlit's own data-testid hooks, which are stable
across Streamlit versions) - no functionality, session state, or data flow is
touched here.
"""
import os
import streamlit as st
from utils.ui.chatbot import render_chat_sidebar

# Indigitise's actual brand purple, taken from their logo SVG (Indigitise_Wordmark_RGB_Tag_V2.svg)
ACCENT = "#695aff"
ACCENT_SOFT = "rgba(105, 90, 255, 0.08)"
BORDER = "rgba(15, 23, 42, 0.10)"
MUTED = "#5b6172"

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.svg")

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ---------- Layout / whitespace ---------- */
[data-testid="stAppViewContainer"] > .main .block-container {{
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* ---------- Typography hierarchy ---------- */
h1 {{
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    font-size: 2.4rem !important;
}}
h2 {{
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    font-size: 1.5rem !important;
}}
h3 {{
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}
[data-testid="stCaptionContainer"], .stCaption {{
    color: {MUTED} !important;
    font-size: 0.95rem !important;
}}

/* ---------- Sidebar / nav ---------- */
[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebarNav"] a {{
    border-radius: 8px;
    margin: 1px 0;
    transition: background 0.15s ease;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: {ACCENT_SOFT};
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: {ACCENT_SOFT};
    border-left: 2px solid {ACCENT};
}}

/* ---------- Cards (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
    border: 1px solid {BORDER} !important;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: rgba(105, 90, 255, 0.45) !important;
}}

/* ---------- Metrics / KPI cards ---------- */
[data-testid="stMetric"] {{
    padding: 0.25rem 0.1rem;
}}
[data-testid="stMetricLabel"] {{
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}}

/* ---------- Buttons ---------- */
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    transition: transform 0.1s ease, opacity 0.15s ease;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-1px);
    opacity: 0.92;
}}
[data-testid="stBaseButton-primary"] {{
    box-shadow: 0 4px 14px rgba(105, 90, 255, 0.24);
}}

/* ---------- Inputs / uploader ---------- */
[data-testid="stFileUploaderDropzone"] {{
    border-radius: 12px !important;
    border: 1.5px dashed {BORDER} !important;
    transition: border-color 0.15s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {ACCENT} !important;
}}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input, [data-baseweb="select"] {{
    border-radius: 8px !important;
}}

/* ---------- Dividers ---------- */
hr {{
    border-color: {BORDER} !important;
}}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {BORDER};
}}
</style>
"""

_HERO_TEMPLATE = """
<div style="
    padding: 2.75rem 2.5rem;
    border-radius: 18px;
    border: 1px solid {border};
    background: radial-gradient(circle at 15% 20%, {accent_soft}, transparent 55%);
    margin-bottom: 2rem;
">
    <div style="
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: {accent_soft};
        color: {accent};
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    ">{eyebrow}</div>
    <h1 style="margin: 0 0 0.5rem 0;">{title}</h1>
    <p style="margin: 0; color: {muted}; font-size: 1.05rem; max-width: 640px;">{subtitle}</p>
</div>
"""

def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH, size="large")
    render_chat_sidebar()

def hero(title, subtitle, eyebrow="AI EVAL"):
    """Landing-page-style header block, styled after indigitise.com.au's hero
    sections (eyebrow tag + large heading + muted subhead), built from our own
    content only."""
    st.markdown(
        _HERO_TEMPLATE.format(
            border=BORDER, accent_soft=ACCENT_SOFT, accent=ACCENT, muted=MUTED,
            eyebrow=eyebrow, title=title, subtitle=subtitle,
        ),
        unsafe_allow_html=True,
    )
