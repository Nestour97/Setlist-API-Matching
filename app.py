"""
app.py  —  Setlist API Reconciliation Agent
Warner Chappell Music Intelligence · Task 3

Run: streamlit run app.py
"""

import io
import os
import json
import time

import streamlit as st
import pandas as pd

from pipeline import (
    run_pipeline,
    build_output_csv,
    CONFIDENCE_EXACT, CONFIDENCE_HIGH, CONFIDENCE_REVIEW, CONFIDENCE_NONE,
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC Setlist Reconciliation",
    page_icon="♪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ────────────────────────────────────────────────────────────────────
BLACK    = "#000000"
BLACK2   = "#0d0d0d"
BLACK3   = "#1a1a1a"
GOLD     = "#F5C518"
GOLD2    = "#c9a010"
GOLD_DIM = "rgba(245,197,24,0.12)"
WHITE    = "#FFFFFF"
GREY1    = "#e0e0e0"
GREY2    = "#a0a0a0"
GREY3    = "#555555"
RED      = "#c0392b"
GREEN    = "#27ae60"
ORANGE   = "#e67e22"
BLUE     = "#7fb3d3"

CONF_COLORS = {
    CONFIDENCE_EXACT:  ("#82c596", "rgba(39,174,96,0.12)"),
    CONFIDENCE_HIGH:   (GOLD,      GOLD_DIM),
    CONFIDENCE_REVIEW: ("#f5c28a", "rgba(230,126,34,0.12)"),
    CONFIDENCE_NONE:   ("#e07060", "rgba(192,57,43,0.10)"),
}

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@300;400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: {WHITE};
}}
.stApp {{ background: {BLACK}; color: {WHITE}; }}

section[data-testid="stSidebar"] {{
    background: {BLACK2} !important;
    border-right: 1px solid {BLACK3} !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}

/* Brand */
.wc-brand {{
    background: {BLACK};
    border-bottom: 2px solid {GOLD};
    padding: 20px 20px 16px;
    margin: -1rem -1rem 20px;
    position: relative;
    overflow: hidden;
}}
.wc-brand::before {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {GOLD}, transparent);
}}
.wc-tagline {{
    font-size: 11px;
    font-style: italic;
    color: {GREY2};
    margin-top: 6px;
    font-weight: 300;
    letter-spacing: 0.02em;
}}
.wc-section {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: {GREY3};
    margin: 20px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid {BLACK3};
}}

/* Main header */
.main-header {{
    border-bottom: 2px solid {GOLD};
    padding-bottom: 14px;
    margin-bottom: 32px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}
.main-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    letter-spacing: 0.08em;
    color: {WHITE};
    line-height: 1;
}}
.main-title span {{ color: {GOLD}; }}
.main-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: {GREY3};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.task-badge {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: {BLACK};
    background: {GOLD};
    padding: 3px 10px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: inline-block;
}}

/* Step cards */
.step-card {{
    background: {BLACK2};
    border: 1px solid {BLACK3};
    border-top: 3px solid {GOLD};
    padding: 16px 20px;
    margin-bottom: 4px;
    height: 100%;
}}
.step-num {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: {GOLD};
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.step-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px;
    letter-spacing: 0.06em;
    color: {WHITE};
    margin-bottom: 6px;
}}
.step-desc {{
    font-size: 12px;
    color: {GREY2};
    font-weight: 300;
    line-height: 1.6;
}}

/* Info boxes */
.info-box {{
    background: {BLACK2};
    border-left: 3px solid {GOLD};
    padding: 12px 18px;
    margin: 8px 0;
    font-size: 13px;
    color: {GREY1};
    font-weight: 300;
    line-height: 1.65;
}}
.success-box {{
    background: rgba(39,174,96,0.07);
    border-left: 3px solid {GREEN};
    padding: 12px 18px;
    margin: 8px 0;
    color: #a8e6c1;
    font-size: 13px;
    font-weight: 300;
}}
.warn-box {{
    background: rgba(230,126,34,0.07);
    border-left: 3px solid {ORANGE};
    padding: 12px 18px;
    margin: 8px 0;
    color: #f5c28a;
    font-size: 13px;
    font-weight: 300;
}}
.error-box {{
    background: rgba(192,57,43,0.08);
    border-left: 3px solid {RED};
    padding: 12px 18px;
    margin: 8px 0;
    color: #e07060;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
}}

/* Code block */
.code-block {{
    background: #050505;
    border: 1px solid {BLACK3};
    border-left: 3px solid {GREY3};
    padding: 14px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 300;
    color: {GREY2};
    white-space: pre-wrap;
    line-height: 1.7;
    letter-spacing: 0.02em;
    max-height: 300px;
    overflow-y: auto;
}}

/* Stats strip */
.stats-strip {{
    display: flex;
    gap: 0;
    margin: 18px 0;
    border: 1px solid {BLACK3};
}}
.stat-item {{
    flex: 1;
    padding: 16px 8px;
    text-align: center;
    border-right: 1px solid {BLACK3};
}}
.stat-item:last-child {{ border-right: none; }}
.stat-num {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    line-height: 1;
}}
.stat-label {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {GREY3};
    margin-top: 4px;
}}

/* Confidence badge */
.badge {{
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    padding: 2px 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 400;
}}

/* Setlist row */
.setlist-row {{
    display: grid;
    grid-template-columns: 90px 140px 1fr 180px 80px;
    gap: 0 12px;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid {BLACK3};
    font-size: 12px;
    transition: background 0.1s;
}}
.setlist-row:hover {{ background: {BLACK2}; }}
.setlist-row.header {{
    background: {BLACK2};
    border-bottom: 2px solid {GOLD};
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {GOLD};
    padding: 10px 12px;
}}
.col-date {{ color: {GREY3}; font-family: 'DM Mono', monospace; font-size: 11px; }}
.col-venue {{ color: {GREY2}; }}
.col-track {{ color: {WHITE}; }}
.col-match {{ color: {GREY2}; font-family: 'DM Mono', monospace; font-size: 11px; }}
.col-conf {{ text-align: center; }}

/* Buttons */
.stButton > button {{
    background: transparent !important;
    border: 1px solid {GOLD} !important;
    border-radius: 0 !important;
    color: {GOLD} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    background: {GOLD_DIM} !important;
}}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: {GREY3} !important;
    border-bottom: 2px solid transparent !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom-color: {GOLD} !important;
}}

/* Data table */
[data-testid="stDataFrame"] {{
    border: 1px solid {BLACK3} !important;
    border-top: 2px solid {GOLD} !important;
}}
[data-testid="stDataFrame"] th {{
    background: {BLACK2} !important;
    color: {GOLD} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: {GREY1} !important;
}}

/* Text input / area */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{
    background: {BLACK2} !important;
    border: 1px solid {BLACK3} !important;
    border-bottom: 2px solid {GREY3} !important;
    border-radius: 0 !important;
    color: {GREY1} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-bottom-color: {GOLD} !important;
    box-shadow: none !important;
}}

/* File uploader */
div[data-testid="stFileUploader"] {{
    border: 1px dashed {GREY3} !important;
    border-radius: 0 !important;
    background: {BLACK2} !important;
}}

/* Progress bar */
.stProgress > div > div {{ background-color: {GOLD} !important; }}

/* Select */
[data-testid="stSelectbox"] div {{ color: {GREY1} !important; font-size: 13px !important; }}

/* Expander */
details summary {{
    color: {GREY2} !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.05em !important;
}}
details summary:hover {{ color: {GOLD} !important; }}

/* Spinner */
.stSpinner > div {{ border-top-color: {GOLD} !important; }}

/* Divider */
hr {{ border: none !important; border-top: 1px solid {BLACK3} !important; margin: 12px 0 !important; }}

/* Empty state */
.empty-state {{
    text-align: center;
    padding: 80px 20px;
    color: {GREY3};
}}
.empty-mark {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 80px;
    color: {GOLD};
    line-height: 1;
    margin-bottom: 12px;
    opacity: 0.25;
}}
.empty-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 36px;
    letter-spacing: 0.08em;
    color: {WHITE};
    margin-bottom: 10px;
}}
.empty-sub {{
    font-size: 13px;
    line-height: 1.9;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto;
    color: {GREY2};
}}

/* Sidebar labels */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {{ color: {GREY1} !important; }}
[data-testid="stSidebar"] input {{
    background: {BLACK3} !important;
    border: 1px solid {GREY3} !important;
    border-radius: 0 !important;
    color: {WHITE} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}}

/* Tour meta card */
.tour-meta {{
    background: {BLACK2};
    border: 1px solid {BLACK3};
    border-left: 3px solid {GOLD};
    padding: 14px 20px;
    display: flex;
    gap: 32px;
    margin-bottom: 20px;
    align-items: center;
}}
.tour-meta-item {{ }}
.tour-meta-label {{
    font-family: 'DM Mono', monospace;
    font-size: 8px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: {GREY3};
    margin-bottom: 3px;
}}
.tour-meta-value {{
    font-size: 14px;
    color: {WHITE};
    font-weight: 500;
}}

/* Cost savings indicator */
.savings-bar {{
    background: {BLACK3};
    height: 6px;
    border-radius: 0;
    margin-top: 8px;
    overflow: hidden;
}}
.savings-fill {{
    height: 100%;
    background: {GOLD};
    transition: width 0.6s ease;
}}
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://music-row-website-assets.s3.amazonaws.com/wp-content/uploads/2019/05/10144300/WCM_Lockup_Black_Gold-TEX.png"


# ── Helper: confidence badge HTML ──────────────────────────────────────────────
def conf_badge(confidence: str) -> str:
    color, bg = CONF_COLORS.get(confidence, (GREY2, BLACK3))
    return (f'<span class="badge" style="color:{color};background:{bg}">'
            f'{confidence}</span>')


# ── Helper: LLM client factory ─────────────────────────────────────────────────
def get_client(provider: str, api_key: str, base_url: str = None):
    from openai import OpenAI
    if provider == "Groq":
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    elif provider == "Custom / Other":
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="wc-brand">
      <img src="{LOGO_URL}"
           alt="Warner Chappell Music"
           style="width:100%;max-width:190px;height:auto;margin-bottom:6px;display:block;" />
      <div class="wc-tagline">Setlist Reconciliation Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="wc-section">LLM Configuration</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        ["OpenAI", "Groq", "Custom / Other"],
        index=1,
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
    )
    if provider == "OpenAI":
        model_default = "gpt-4o-mini"
    elif provider == "Groq":
        model_default = "llama-3.3-70b-versatile"
    else:
        model_default = ""

    model = st.text_input("Model", value=model_default)

    if provider == "Custom / Other":
        base_url = st.text_input("Base URL", placeholder="https://api.example.com/v1")
    else:
        base_url = None

    st.markdown('<div class="wc-section">Tour API Source</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px;color:{GREY3};font-weight:300;margin-bottom:8px;line-height:1.6">'
        f'Enter a URL (e.g. GitHub Gist raw URL) or use the bundled local file.</div>',
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Source",
        ["Local file (tour_payload.json)", "URL"],
        index=0,
        label_visibility="collapsed",
    )
    if source_mode == "URL":
        tour_url = st.text_input(
            "Tour API URL",
            placeholder="https://gist.githubusercontent.com/.../tour_payload.json",
            label_visibility="collapsed",
        )
    else:
        tour_url = None

    st.markdown('<div class="wc-section">Internal Catalog</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px;color:{GREY3};font-weight:300;margin-bottom:8px;line-height:1.6">'
        f'Upload your catalog.csv, or leave blank to use the bundled sample catalog.</div>',
        unsafe_allow_html=True,
    )
    uploaded_catalog = st.file_uploader("", type=["csv"], label_visibility="collapsed",
                                        key="catalog_upload")

    st.markdown('<div class="wc-section">Options</div>', unsafe_allow_html=True)
    show_raw_json    = st.checkbox("Show raw tour JSON", value=False)
    show_catalog_tbl = st.checkbox("Show loaded catalog", value=False)
    show_flat_rows   = st.checkbox("Show flattened setlist rows", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset", use_container_width=True):
        if "pipeline_result" in st.session_state:
            del st.session_state["pipeline_result"]
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <div>
    <div class="task-badge">Task 3</div>
    <div class="main-title">Setlist <span>Reconciliation</span></div>
  </div>
  <div class="main-sub">Fetch · Match · Flag · Deliver</div>
</div>
""", unsafe_allow_html=True)

# ── Step overview ──────────────────────────────────────────────────────────────
cols = st.columns(5)
steps = [
    ("01", "API Ingestion",      "Fetch tour JSON from URL or local file"),
    ("02", "Catalog Load",       "Read internal controlled song catalog"),
    ("03", "Flatten Setlist",    "Explode nested JSON into per-track rows"),
    ("04", "Exact Matching",     "Deterministic: exact, normalized, qualifier-stripped, medley"),
    ("05", "AI Fuzzy Match",     "LLM resolves abbreviations, garbled text, covers"),
]
for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">Step {num}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Source info ────────────────────────────────────────────────────────────────
col_src, col_cat = st.columns(2)
with col_src:
    st.markdown(f'<div class="wc-section">Tour Data Source</div>', unsafe_allow_html=True)
    if source_mode == "URL" and tour_url:
        st.markdown(
            f'<div class="info-box">🌐 URL: <code style="color:{GOLD};font-size:11px">{tour_url}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-box">📄 Local file: <code style="color:{GOLD};font-size:11px">tour_payload.json</code> '
            f'(bundled with repo)</div>',
            unsafe_allow_html=True,
        )

with col_cat:
    st.markdown(f'<div class="wc-section">Internal Catalog</div>', unsafe_allow_html=True)
    if uploaded_catalog:
        # Validate uploaded catalog immediately — show preview + warn on schema issues
        try:
            import csv as _csv, io as _io
            uploaded_catalog.seek(0)
            _rows = list(_csv.DictReader(_io.StringIO(
                uploaded_catalog.read().decode("utf-8", errors="replace")
            )))
            uploaded_catalog.seek(0)
            _cols   = list(_rows[0].keys()) if _rows else []
            _has_id = "catalog_id" in _cols
            _title_col = next((c for c in ("title","song_title","name") if c in _cols), None)

            if not _has_id or not _title_col:
                st.markdown(
                    f'<div class="error-box">⚠ Uploaded file is missing required columns.<br>'
                    f'Found: {", ".join(_cols)}<br>'
                    f'Need: <strong>catalog_id</strong> + <strong>title</strong> (or song_title).<br>'
                    f'<strong>Falling back to bundled catalog.csv</strong></div>',
                    unsafe_allow_html=True,
                )
                uploaded_catalog = None   # force fallback to bundled
            else:
                _ids    = [r["catalog_id"]       for r in _rows]
                _titles = [r.get(_title_col, "") for r in _rows]
                # Check if this looks like the old wrong catalog (CAT-002=Tokyo Midnight etc)
                _id_map = {r["catalog_id"]: r.get(_title_col,"") for r in _rows}
                _wrong_hints = [
                    k for k,v in _id_map.items()
                    if (k=="CAT-002" and "Tokyo Midnight" in v)
                    or (k=="CAT-003" and v in ("Desert Rain","Shatter","Shattered"))
                    or (k=="CAT-013" and "Midnight" in v)
                ]
                if _wrong_hints:
                    st.markdown(
                        f'<div class="error-box">⚠ Uploaded catalog appears to have incorrect IDs '
                        f'(e.g. {_id_map.get("CAT-002","?")} as CAT-002, '
                        f'{_id_map.get("CAT-013","?")} as CAT-013).<br>'
                        f'This is an old/wrong version of the catalog.<br>'
                        f'<strong>Using bundled catalog.csv instead.</strong></div>',
                        unsafe_allow_html=True,
                    )
                    uploaded_catalog = None   # force correct bundled catalog
                else:
                    st.markdown(
                        f'<div class="success-box">✓ Uploaded: <strong>{uploaded_catalog.name}</strong> '
                        f'· {len(_rows)} songs · column: <code>{_title_col}</code></div>',
                        unsafe_allow_html=True,
                    )
                    # Show first few rows as preview
                    preview_html = "".join(
                        f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{GREY2};'
                        f'padding:2px 0;border-bottom:1px solid {BLACK3}">'
                        f'<span style="color:{GOLD};margin-right:12px">{r["catalog_id"]}</span>'
                        f'{r.get(_title_col,"")}</div>'
                        for r in _rows[:6]
                    )
                    if len(_rows) > 6:
                        preview_html += f'<div style="font-size:10px;color:{GREY3};margin-top:4px">…and {len(_rows)-6} more</div>'
                    st.markdown(
                        f'<div style="background:{BLACK2};border:1px solid {BLACK3};'
                        f'border-left:3px solid {GOLD};padding:10px 14px;margin-top:6px">'
                        f'{preview_html}</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as _e:
            st.markdown(
                f'<div class="error-box">⚠ Could not read uploaded catalog: {_e}<br>'
                f'Using bundled catalog.csv</div>',
                unsafe_allow_html=True,
            )
            uploaded_catalog = None

    if not uploaded_catalog:
        # Load bundled catalog for preview
        try:
            import csv as _csv2
            with open("catalog.csv") as _f:
                _brows = list(_csv2.DictReader(_f))
            preview_html = "".join(
                f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{GREY2};'
                f'padding:2px 0;border-bottom:1px solid {BLACK3}">'
                f'<span style="color:{GOLD};margin-right:12px">{r["catalog_id"]}</span>'
                f'{r.get("title", r.get("song_title",""))}</div>'
                for r in _brows[:6]
            )
            if len(_brows) > 6:
                preview_html += f'<div style="font-size:10px;color:{GREY3};margin-top:4px">…and {len(_brows)-6} more</div>'
            st.markdown(
                f'<div class="info-box" style="margin-bottom:6px">📄 Using bundled '
                f'<code style="color:{GOLD};font-size:11px">catalog.csv</code> '
                f'· {len(_brows)} controlled songs</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:{BLACK2};border:1px solid {BLACK3};'
                f'border-left:3px solid {GOLD};padding:10px 14px">'
                f'{preview_html}</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                f'<div class="warn-box">catalog.csv not found in working directory</div>',
                unsafe_allow_html=True,
            )

# ── Run button ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_run, col_warn = st.columns([1, 3])
with col_run:
    run_clicked = st.button("▶  Run Reconciliation", use_container_width=True)
with col_warn:
    if not api_key:
        st.markdown(
            f'<div class="warn-box" style="margin:4px 0">Set your API key in the sidebar to run.</div>',
            unsafe_allow_html=True,
        )

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run_clicked:
    if not api_key:
        st.error("Please enter an API key in the sidebar.")
        st.stop()
    if not model:
        st.error("Please enter a model name in the sidebar.")
        st.stop()

    # Determine tour source
    if source_mode == "URL":
        if not tour_url:
            st.error("Please enter a Tour API URL.")
            st.stop()
        tour_source = tour_url
    else:
        tour_source = "tour_payload.json"

    # Determine catalog source (uploaded_catalog may have been set to None above if invalid)
    if uploaded_catalog:
        uploaded_catalog.seek(0)
        catalog_source = uploaded_catalog
    else:
        catalog_source = "catalog.csv"

    try:
        client = get_client(provider, api_key, base_url)
    except Exception as e:
        st.error(f"Failed to initialise LLM client: {e}")
        st.stop()

    progress_bar = st.progress(0)
    status_box = st.empty()

    def on_progress(step, total, message):
        progress_bar.progress(step / total)
        status_box.markdown(
            f'<div class="info-box">⟳ <strong>Step {step}/{total}</strong> — {message}</div>',
            unsafe_allow_html=True,
        )

    with st.spinner(""):
        try:
            result = run_pipeline(
                tour_source=tour_source,
                catalog_file=catalog_source,
                client=client,
                model=model,
                progress_callback=on_progress,
            )
            st.session_state["pipeline_result"] = result
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    progress_bar.progress(1.0)
    status_box.empty()


# ── Results ────────────────────────────────────────────────────────────────────
if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    errors   = result.get("errors", [])
    stats    = result.get("stats", {})
    results  = result.get("results", [])
    catalog  = result.get("catalog", [])
    flat_rows = result.get("flat_rows", [])
    tour_meta = result.get("tour_meta", {})

    # ── Errors ─────────────────────────────────────────────────────────────────
    for err in errors:
        st.markdown(f'<div class="error-box">⚠ {err}</div>', unsafe_allow_html=True)

    # ── Tour meta bar ──────────────────────────────────────────────────────────
    if tour_meta:
        st.markdown(f"""
        <div class="tour-meta">
          <div class="tour-meta-item">
            <div class="tour-meta-label">Artist</div>
            <div class="tour-meta-value">{tour_meta.get('artist','—')}</div>
          </div>
          <div class="tour-meta-item">
            <div class="tour-meta-label">Tour</div>
            <div class="tour-meta-value">{tour_meta.get('tour','—')}</div>
          </div>
          <div class="tour-meta-item">
            <div class="tour-meta-label">Shows</div>
            <div class="tour-meta-value">{tour_meta.get('show_count','—')}</div>
          </div>
          <div class="tour-meta-item">
            <div class="tour-meta-label">Total Tracks</div>
            <div class="tour-meta-value">{stats.get('total_tracks','—')}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats strip ────────────────────────────────────────────────────────────
    ec = CONF_COLORS[CONFIDENCE_EXACT][0]
    hc = CONF_COLORS[CONFIDENCE_HIGH][0]
    rc = CONF_COLORS[CONFIDENCE_REVIEW][0]
    nc = CONF_COLORS[CONFIDENCE_NONE][0]
    savings = stats.get("llm_savings_pct", 0)

    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-item">
        <div class="stat-num" style="color:{ec}">{stats.get('exact_matches',0)}</div>
        <div class="stat-label">Exact Match</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{hc}">{stats.get('high_matches',0)}</div>
        <div class="stat-label">High Confidence</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{rc}">{stats.get('review_matches',0)}</div>
        <div class="stat-label">Needs Review</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{nc}">{stats.get('no_matches',0)}</div>
        <div class="stat-label">Not Controlled</div>
      </div>
      <div class="stat-item">
        <div class="stat-num" style="color:{GOLD}">{savings}%</div>
        <div class="stat-label">LLM Savings</div>
        <div class="savings-bar">
          <div class="savings-fill" style="width:{savings}%"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cost explanation
    n_det = stats.get("deterministic", 0)
    n_llm = stats.get("llm_resolved", 0)
    total = stats.get("total_tracks", 1)
    st.markdown(
        f'<div class="info-box" style="margin-bottom:20px">'
        f'<strong style="color:{GOLD}">Cost Optimisation:</strong> '
        f'{n_det}/{total} tracks resolved deterministically (no LLM cost). '
        f'Only {n_llm} track{"s" if n_llm!=1 else ""} sent to the AI — '
        f'saving approximately {savings}% of potential LLM calls.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Debug toggles ──────────────────────────────────────────────────────────
    if show_raw_json:
        st.markdown(f'<div class="wc-section">Raw Tour JSON</div>', unsafe_allow_html=True)
        with st.expander("View payload", expanded=False):
            try:
                import json
                with open("tour_payload.json") as f:
                    raw_json = json.load(f)
                st.markdown(
                    f'<div class="code-block">{json.dumps(raw_json, indent=2)}</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Load the JSON from sidebar to view it here.")

    if show_catalog_tbl and catalog:
        st.markdown(f'<div class="wc-section">Internal Catalog ({len(catalog)} songs)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(catalog), use_container_width=True, hide_index=True)

    if show_flat_rows and flat_rows:
        st.markdown(f'<div class="wc-section">Flattened Setlist Rows ({len(flat_rows)} tracks)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(flat_rows), use_container_width=True, hide_index=True)

    # ── Main results table ─────────────────────────────────────────────────────
    tabs = st.tabs(["Reconciliation Report", "By Show", "Dataframe View"])

    with tabs[0]:
        st.markdown(f'<div class="wc-section">Matched Setlists — All Shows</div>', unsafe_allow_html=True)

        # Header row
        st.markdown(f"""
        <div class="setlist-row header">
          <span>Date</span>
          <span>Venue</span>
          <span>Setlist Track</span>
          <span>Catalog ID</span>
          <span>Confidence</span>
        </div>
        """, unsafe_allow_html=True)

        current_date = None
        for row in results:
            date = row["show_date"]

            # Date divider
            if date != current_date:
                current_date = date
                venue = row["venue_name"]
                st.markdown(
                    f'<div style="padding:10px 12px 4px;font-family:DM Mono,monospace;'
                    f'font-size:9px;letter-spacing:0.2em;text-transform:uppercase;color:{GOLD}">'
                    f'▸ {date} · {venue}</div>',
                    unsafe_allow_html=True,
                )

            cat_id = row["matched_catalog_id"]
            conf   = row["match_confidence"]
            track  = row["setlist_track_name"]
            notes  = row.get("match_notes", "")
            badge  = conf_badge(conf)

            # Dim uncontrolled rows
            opacity = "0.45" if conf == CONFIDENCE_NONE else "1"

            st.markdown(f"""
            <div class="setlist-row" style="opacity:{opacity}" title="{notes}">
              <span class="col-date">{date}</span>
              <span class="col-venue">{row['venue_name']}</span>
              <span class="col-track">{track}</span>
              <span class="col-match">{cat_id}</span>
              <span class="col-conf">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

        # Match notes legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="wc-section">Match Notes</div>', unsafe_allow_html=True)
        for row in results:
            notes = row.get("match_notes", "")
            if notes:
                track = row["setlist_track_name"]
                conf  = row["match_confidence"]
                color, _ = CONF_COLORS.get(conf, (GREY2, BLACK3))
                st.markdown(
                    f'<div style="display:flex;gap:12px;padding:5px 0;border-bottom:1px solid {BLACK3};'
                    f'font-size:12px;align-items:flex-start">'
                    f'<span style="color:{color};font-family:DM Mono,monospace;font-size:10px;'
                    f'min-width:90px">{conf}</span>'
                    f'<span style="color:{GREY1};min-width:200px">{track}</span>'
                    f'<span style="color:{GREY3};font-family:DM Mono,monospace;font-size:11px">{notes}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tabs[1]:
        st.markdown(f'<div class="wc-section">Results by Show</div>', unsafe_allow_html=True)
        shows = {}
        for row in results:
            key = f"{row['show_date']} · {row['venue_name']}"
            shows.setdefault(key, []).append(row)

        for show_label, show_rows in shows.items():
            matched = sum(1 for r in show_rows if r["matched_catalog_id"] != "None")
            total_s = len(show_rows)
            st.markdown(
                f'<div style="font-family:Bebas Neue,sans-serif;font-size:22px;'
                f'letter-spacing:0.06em;color:{WHITE};margin-top:20px;padding-bottom:6px;'
                f'border-bottom:1px solid {GOLD}">'
                f'{show_label} '
                f'<span style="font-family:DM Mono,monospace;font-size:11px;color:{GREY3}">'
                f'({matched}/{total_s} controlled)</span></div>',
                unsafe_allow_html=True,
            )
            for r in show_rows:
                conf  = r["match_confidence"]
                color, bg = CONF_COLORS.get(conf, (GREY2, BLACK3))
                cat_id = r["matched_catalog_id"]
                track  = r["setlist_track_name"]
                notes  = r.get("match_notes", "")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;padding:7px 0;'
                    f'border-bottom:1px solid {BLACK3}">'
                    f'{conf_badge(conf)}'
                    f'<span style="color:{WHITE};flex:1">{track}</span>'
                    f'<span style="color:{GREY3};font-family:DM Mono,monospace;font-size:11px">{cat_id}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        st.markdown(f'<div class="wc-section">Raw Output DataFrame</div>', unsafe_allow_html=True)
        if results:
            df_out = pd.DataFrame(results)
            st.dataframe(df_out, use_container_width=True, hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = build_output_csv(results)

    col_dl, col_info = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇  Download CSV",
            data=csv_data,
            file_name="matched_setlists.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_info:
        controlled = sum(1 for r in results if r["matched_catalog_id"] != "None")
        st.markdown(
            f'<div class="info-box" style="margin:0;padding:10px 16px">'
            f'Output: <code style="color:{GOLD}">matched_setlists.csv</code> · '
            f'{len(results)} total tracks · {controlled} controlled · '
            f'{len(results)-controlled} uncontrolled/cover</div>',
            unsafe_allow_html=True,
        )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-mark">♫</div>
      <div class="empty-title">Ready to Reconcile</div>
      <div class="empty-sub">
        Configure your LLM in the sidebar,<br>
        then click <strong style="color:{GOLD}">▶ Run Reconciliation</strong>.<br><br>
        The bundled <code style="color:{GOLD}">tour_payload.json</code> and
        <code style="color:{GOLD}">catalog.csv</code> work out of the box.<br><br>
        Hover over any result row to see match reasoning.<br>
        Dimmed rows = uncontrolled / cover songs.
      </div>
    </div>
    """, unsafe_allow_html=True)
