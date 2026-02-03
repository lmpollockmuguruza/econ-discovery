"""
Literature Discovery — Premium Research Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A refined interface for discovering relevant academic papers,
designed with minimalist aesthetics and intelligent personalization.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, List
import time

from api_client import (
    fetch_recent_papers,
    get_economics_journals,
    get_polisci_journals,
    get_all_journals,
    get_journal_options,
    OpenAlexError
)
from processor import (
    process_papers_with_gemini,
    create_user_profile,
    get_profile_options,
    get_sdk_info,
    get_suggested_authors,
    UserProfile
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Literature Discovery",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREMIUM CSS — APPLE-ESQUE DESIGN SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════
       TYPOGRAPHY — SF Pro / System Font Stack
       ═══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --font-serif: 'Source Serif 4', 'New York', Georgia, serif;
        
        /* Colors */
        --bg-primary: #FAFAFA;
        --bg-card: #FFFFFF;
        --bg-elevated: #F5F5F7;
        --text-primary: #1D1D1F;
        --text-secondary: #6E6E73;
        --text-tertiary: #86868B;
        --accent: #0071E3;
        --accent-hover: #0077ED;
        --success: #34C759;
        --warning: #FF9500;
        --error: #FF3B30;
        --border: #D2D2D7;
        --border-light: #E8E8ED;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 12px 40px rgba(0,0,0,0.12);
        
        /* Spacing */
        --space-xs: 4px;
        --space-sm: 8px;
        --space-md: 16px;
        --space-lg: 24px;
        --space-xl: 40px;
        --space-2xl: 64px;
        
        /* Radius */
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
    }
    
    /* ═══════════════════════════════════════════════════════════════
       BASE STYLES
       ═══════════════════════════════════════════════════════════════ */
    .stApp {
        background-color: var(--bg-primary);
    }
    
    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ═══════════════════════════════════════════════════════════════
       WIZARD PROGRESS BAR
       ═══════════════════════════════════════════════════════════════ */
    .wizard-progress {
        display: flex;
        justify-content: center;
        gap: var(--space-lg);
        padding: var(--space-xl) 0;
        margin-bottom: var(--space-xl);
    }
    
    .wizard-step {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        color: var(--text-tertiary);
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .wizard-step.active {
        color: var(--text-primary);
    }
    
    .wizard-step.completed {
        color: var(--success);
    }
    
    .step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 13px;
        background: var(--bg-elevated);
        border: 2px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .wizard-step.active .step-number {
        background: var(--text-primary);
        color: white;
        border-color: var(--text-primary);
    }
    
    .wizard-step.completed .step-number {
        background: var(--success);
        color: white;
        border-color: var(--success);
    }
    
    .step-line {
        width: 60px;
        height: 2px;
        background: var(--border);
        transition: background 0.3s ease;
    }
    
    .step-line.completed {
        background: var(--success);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       HERO SECTION
       ═══════════════════════════════════════════════════════════════ */
    .hero {
        text-align: center;
        padding: var(--space-2xl) var(--space-xl);
        max-width: 700px;
        margin: 0 auto;
    }
    
    .hero h1 {
        font-family: var(--font-serif);
        font-size: 48px;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: var(--space-md);
        color: var(--text-primary);
    }
    
    .hero p {
        font-size: 18px;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ═══════════════════════════════════════════════════════════════
       CARDS
       ═══════════════════════════════════════════════════════════════ */
    .config-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
        margin-bottom: var(--space-lg);
    }
    
    .config-card h3 {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-tertiary);
        margin-bottom: var(--space-md);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       MATCH CARDS — Paper Results
       ═══════════════════════════════════════════════════════════════ */
    .match-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        margin-bottom: var(--space-md);
        border: 1px solid var(--border-light);
        transition: all 0.2s ease;
    }
    
    .match-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--border);
    }
    
    .match-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: var(--space-md);
        margin-bottom: var(--space-md);
    }
    
    .match-card h4 {
        font-family: var(--font-serif);
        font-size: 18px;
        font-weight: 600;
        line-height: 1.4;
        color: var(--text-primary);
        margin: 0;
        flex: 1;
    }
    
    .relevance-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        white-space: nowrap;
    }
    
    .relevance-high {
        background: #E8F5E9;
        color: #2E7D32;
    }
    
    .relevance-medium {
        background: #FFF3E0;
        color: #E65100;
    }
    
    .relevance-low {
        background: var(--bg-elevated);
        color: var(--text-tertiary);
    }
    
    .match-meta {
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: var(--space-md);
    }
    
    .match-summary {
        font-size: 15px;
        line-height: 1.6;
        color: var(--text-primary);
        margin-bottom: var(--space-md);
        padding-left: var(--space-md);
        border-left: 3px solid var(--accent);
    }
    
    .match-relevance {
        background: var(--bg-elevated);
        padding: var(--space-md);
        border-radius: var(--radius-md);
        font-size: 14px;
        color: var(--text-secondary);
        margin-bottom: var(--space-md);
    }
    
    .match-relevance strong {
        color: var(--text-primary);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       PILL TAGS
       ═══════════════════════════════════════════════════════════════ */
    .pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-sm);
        margin-top: var(--space-sm);
    }
    
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .pill-method {
        background: #E3F2FD;
        color: #1565C0;
    }
    
    .pill-topic {
        background: #F3E5F5;
        color: #7B1FA2;
    }
    
    .pill-oa {
        background: #E8F5E9;
        color: #2E7D32;
    }
    
    .pill-journal {
        background: var(--bg-elevated);
        color: var(--text-secondary);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       BUTTONS
       ═══════════════════════════════════════════════════════════════ */
    .stButton > button {
        font-family: var(--font-sans);
        font-weight: 600;
        border-radius: var(--radius-md);
        padding: 12px 24px;
        transition: all 0.2s ease;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--text-primary);
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #333;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-card);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       FORM ELEMENTS
       ═══════════════════════════════════════════════════════════════ */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        border-radius: var(--radius-md);
        border-color: var(--border);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(0,113,227,0.15);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: var(--accent);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       SYNTHESIS BOX
       ═══════════════════════════════════════════════════════════════ */
    .synthesis-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        color: white;
        margin-bottom: var(--space-xl);
    }
    
    .synthesis-box h3 {
        font-family: var(--font-serif);
        font-size: 20px;
        font-weight: 600;
        margin-bottom: var(--space-md);
        opacity: 0.95;
    }
    
    .synthesis-box p {
        font-size: 15px;
        line-height: 1.7;
        opacity: 0.9;
    }
    
    /* ═══════════════════════════════════════════════════════════════
       METRICS
       ═══════════════════════════════════════════════════════════════ */
    .metric-row {
        display: flex;
        gap: var(--space-md);
        margin-bottom: var(--space-xl);
    }
    
    .metric-box {
        flex: 1;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: var(--space-lg);
        text-align: center;
        border: 1px solid var(--border-light);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 13px;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: var(--space-xs);
    }
    
    /* ═══════════════════════════════════════════════════════════════
       STATUS MESSAGES
       ═══════════════════════════════════════════════════════════════ */
    .status-success {
        background: #E8F5E9;
        color: #2E7D32;
        padding: var(--space-md);
        border-radius: var(--radius-md);
        font-size: 14px;
    }
    
    .status-warning {
        background: #FFF3E0;
        color: #E65100;
        padding: var(--space-md);
        border-radius: var(--radius-md);
        font-size: 14px;
    }
    
    .status-error {
        background: #FFEBEE;
        color: #C62828;
        padding: var(--space-md);
        border-radius: var(--radius-md);
        font-size: 14px;
    }
    
    /* ═══════════════════════════════════════════════════════════════
       RESPONSIVE
       ═══════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {
        .hero h1 { font-size: 32px; }
        .wizard-progress { flex-wrap: wrap; }
        .step-line { width: 30px; }
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "wizard_step": 1,
        "processed_papers": [],
        "status_messages": [],
        "synthesis": None,
        "user_profile": None,
        "api_key": "",
        # Wizard data
        "w_academic_level": "PhD Student (ABD)",
        "w_primary_field": "Labor Economics",
        "w_interests": ["Causal Inference"],
        "w_methods": ["Difference-in-Differences"],
        "w_region": "United States",
        "w_method_lean": 0.7,
        "w_seed_authors": [],
        "w_journals": [],
        "w_days_back": 30,
        "w_max_papers": 20,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHED FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_papers(journals_tuple: tuple, days_back: int, max_papers: int):
    """Cached paper fetching to avoid redundant API calls."""
    return fetch_recent_papers(
        days_back=days_back,
        selected_journals=list(journals_tuple),
        max_results=max_papers
    )


@st.cache_resource
def get_cached_options():
    """Cache profile options."""
    return get_profile_options()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_date(date_str: str) -> str:
    """Format date string for display."""
    if not date_str:
        return "Unknown"
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d, %Y")
    except:
        return date_str


def render_wizard_progress(current_step: int):
    """Render the step progress indicator."""
    steps = ["Profile", "Interests", "Sources", "Discover"]
    
    html = '<div class="wizard-progress">'
    for i, step in enumerate(steps, 1):
        status = "completed" if i < current_step else "active" if i == current_step else ""
        check = "✓" if i < current_step else str(i)
        
        html += f'''
        <div class="wizard-step {status}">
            <div class="step-number">{check}</div>
            <span>{step}</span>
        </div>
        '''
        if i < len(steps):
            line_status = "completed" if i < current_step else ""
            html += f'<div class="step-line {line_status}"></div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_match_card(paper: dict, index: int):
    """Render a paper result as a match card."""
    score = paper.get("relevance_score", 5)
    has_ai = paper.get("has_ai_analysis", False)
    
    # Determine relevance class
    if not has_ai:
        rel_class = "relevance-low"
        rel_icon = "○"
    elif score >= 8:
        rel_class = "relevance-high"
        rel_icon = "●"
    elif score >= 5:
        rel_class = "relevance-medium"
        rel_icon = "◐"
    else:
        rel_class = "relevance-low"
        rel_icon = "○"
    
    # Authors
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" et al."
    
    # Build pills HTML
    pills_html = '<div class="pill-container">'
    
    # Method matches
    for method in paper.get("method_matches", [])[:2]:
        pills_html += f'<span class="pill pill-method">⚙ {method}</span>'
    
    # Topic matches
    for topic in paper.get("topic_matches", [])[:2]:
        pills_html += f'<span class="pill pill-topic">◆ {topic}</span>'
    
    # Open access
    if paper.get("is_open_access"):
        pills_html += '<span class="pill pill-oa">🔓 Open Access</span>'
    
    # Journal tier
    tier = paper.get("journal_tier", 4)
    if tier == 1:
        pills_html += '<span class="pill pill-journal">★ Top Journal</span>'
    
    pills_html += '</div>'
    
    # Render with Streamlit
    with st.container():
        st.markdown(f"""
        <div class="match-card">
            <div class="match-card-header">
                <h4>{paper.get('title', 'Untitled')}</h4>
                <div class="relevance-badge {rel_class}">
                    {rel_icon} {score}/10
                </div>
            </div>
            <div class="match-meta">
                {paper.get('journal', 'Unknown')} · {author_str} · {format_date(paper.get('publication_date'))}
            </div>
        """, unsafe_allow_html=True)
        
        if has_ai and paper.get("ai_contribution"):
            st.markdown(f"""
            <div class="match-summary">
                {paper.get('ai_contribution', '')}
            </div>
            """, unsafe_allow_html=True)
        
        if has_ai and paper.get("ai_relevance"):
            st.markdown(f"""
            <div class="match-relevance">
                <strong>Why it matches:</strong> {paper.get('ai_relevance', '')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(pills_html, unsafe_allow_html=True)
        
        # Link button
        link = paper.get("doi_url") or paper.get("oa_url")
        if link:
            st.markdown(f"""
                <div style="margin-top: 12px;">
                    <a href="{link}" target="_blank" style="
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        color: #0071E3;
                        font-size: 14px;
                        font-weight: 500;
                        text-decoration: none;
                    ">
                        Read Paper →
                    </a>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WIZARD STEPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_step_1_profile():
    """Step 1: Basic profile setup."""
    options = get_cached_options()
    
    st.markdown("""
    <div class="hero">
        <h1>Tell us about yourself</h1>
        <p>We'll personalize your paper recommendations based on your research focus and career stage.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        
        st.markdown("##### 🔑 API Key")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Paste your Gemini API key",
            help="[Get a free key →](https://aistudio.google.com/app/apikey)"
        )
        st.session_state.api_key = api_key
        
        if api_key:
            if len(api_key) >= 30:
                st.success("✓ Key looks valid", icon="✓")
            else:
                st.warning("Key seems short")
        
        st.divider()
        
        st.markdown("##### 👤 Career Stage")
        academic_level = st.selectbox(
            "Academic Level",
            options=options["academic_levels"],
            index=options["academic_levels"].index(st.session_state.w_academic_level) 
                  if st.session_state.w_academic_level in options["academic_levels"] else 3,
            label_visibility="collapsed"
        )
        st.session_state.w_academic_level = academic_level
        
        st.markdown("##### 📚 Primary Field")
        primary_field = st.selectbox(
            "Primary Field",
            options=options["primary_fields"],
            index=options["primary_fields"].index(st.session_state.w_primary_field)
                  if st.session_state.w_primary_field in options["primary_fields"] else 3,
            label_visibility="collapsed"
        )
        st.session_state.w_primary_field = primary_field
        
        st.markdown("##### 🌍 Regional Focus")
        region = st.selectbox(
            "Region",
            options=options["regional_focus"],
            index=options["regional_focus"].index(st.session_state.w_region)
                  if st.session_state.w_region in options["regional_focus"] else 0,
            label_visibility="collapsed"
        )
        st.session_state.w_region = region
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_b:
            if st.button("Continue →", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please enter your Gemini API key")
                else:
                    st.session_state.wizard_step = 2
                    st.rerun()


def render_step_2_interests():
    """Step 2: Research interests and methodology."""
    options = get_cached_options()
    
    st.markdown("""
    <div class="hero">
        <h1>Your research interests</h1>
        <p>Help us understand what topics and methods matter most to you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        
        st.markdown("##### 🎯 Research Interests")
        st.caption("Select up to 5 topics you care about")
        interests = st.multiselect(
            "Interests",
            options=options["secondary_interests"],
            default=st.session_state.w_interests,
            max_selections=5,
            label_visibility="collapsed"
        )
        st.session_state.w_interests = interests
        
        st.divider()
        
        st.markdown("##### ⚙️ Preferred Methods")
        st.caption("Select up to 4 methodologies")
        methods = st.multiselect(
            "Methods",
            options=options["methodologies"],
            default=st.session_state.w_methods,
            max_selections=4,
            label_visibility="collapsed"
        )
        st.session_state.w_methods = methods
        
        st.divider()
        
        st.markdown("##### 📊 Methodological Leaning")
        st.caption("Where do you fall on the methods spectrum?")
        method_lean = st.slider(
            "Method Lean",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.w_method_lean,
            format="",
            label_visibility="collapsed"
        )
        
        lean_labels = ["Qualitative / Theory", "Mixed Methods", "Quantitative / Causal"]
        if method_lean < 0.35:
            st.caption(f"← **{lean_labels[0]}**")
        elif method_lean > 0.65:
            st.caption(f"**{lean_labels[2]}** →")
        else:
            st.caption(f"**{lean_labels[1]}**")
        
        st.session_state.w_method_lean = method_lean
        
        st.divider()
        
        st.markdown("##### 👥 Seed Authors (Optional)")
        st.caption("Add researchers whose work you follow")
        
        # Show suggestions based on field
        suggestions = get_suggested_authors(st.session_state.w_primary_field)
        if suggestions:
            st.caption(f"Suggestions for {st.session_state.w_primary_field}: {', '.join(suggestions[:3])}")
        
        seed_authors_input = st.text_input(
            "Seed Authors",
            value=", ".join(st.session_state.w_seed_authors),
            placeholder="e.g., Raj Chetty, Esther Duflo",
            label_visibility="collapsed"
        )
        st.session_state.w_seed_authors = [a.strip() for a in seed_authors_input.split(",") if a.strip()]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1
                st.rerun()
        with col_b:
            if st.button("Continue →", type="primary", use_container_width=True):
                if not interests:
                    st.error("Please select at least one interest")
                elif not methods:
                    st.error("Please select at least one method")
                else:
                    st.session_state.wizard_step = 3
                    st.rerun()


def render_step_3_sources():
    """Step 3: Journal and source selection."""
    journal_opts = get_journal_options()
    
    st.markdown("""
    <div class="hero">
        <h1>Choose your sources</h1>
        <p>Select which journals to search and how far back to look.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        
        st.markdown("##### 📖 Field")
        field_choice = st.radio(
            "Field",
            ["Economics", "Political Science", "Both"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("##### 📚 Journals")
        
        # Get available journals based on field choice
        if field_choice == "Economics":
            available = get_economics_journals()
            top_journals = journal_opts["economics"]["top5"]
        elif field_choice == "Political Science":
            available = get_polisci_journals()
            top_journals = journal_opts["polisci"]["top3"]
        else:
            available = get_all_journals()
            top_journals = journal_opts["economics"]["top5"][:3] + journal_opts["polisci"]["top3"][:2]
        
        # Quick select buttons
        qcol1, qcol2 = st.columns(2)
        with qcol1:
            if st.button("Select Top Journals", use_container_width=True):
                st.session_state.w_journals = top_journals
                st.rerun()
        with qcol2:
            if st.button("Select All", use_container_width=True):
                st.session_state.w_journals = available
                st.rerun()
        
        # Filter current selection to available journals
        current_selection = [j for j in st.session_state.w_journals if j in available]
        if not current_selection:
            current_selection = top_journals[:3]
        
        journals = st.multiselect(
            "Journals",
            options=available,
            default=current_selection,
            label_visibility="collapsed"
        )
        st.session_state.w_journals = journals
        
        st.divider()
        
        st.markdown("##### ⏱️ Time Range")
        days_back = st.slider(
            "Days Back",
            min_value=7,
            max_value=90,
            value=st.session_state.w_days_back,
            step=7,
            format="%d days",
            label_visibility="collapsed"
        )
        st.session_state.w_days_back = days_back
        
        st.markdown("##### 📊 Maximum Papers")
        max_papers = st.slider(
            "Max Papers",
            min_value=10,
            max_value=50,
            value=st.session_state.w_max_papers,
            step=5,
            label_visibility="collapsed"
        )
        st.session_state.w_max_papers = max_papers
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col_b:
            if st.button("🔍 Discover Papers", type="primary", use_container_width=True):
                if not journals:
                    st.error("Please select at least one journal")
                else:
                    st.session_state.wizard_step = 4
                    st.rerun()


def render_step_4_results():
    """Step 4: Discovery and results."""
    
    # Check if we need to run discovery
    if not st.session_state.processed_papers:
        run_discovery()
    
    # Show results
    render_results()


def run_discovery():
    """Execute the paper discovery pipeline."""
    
    # Build profile
    profile = create_user_profile(
        academic_level=st.session_state.w_academic_level,
        primary_field=st.session_state.w_primary_field,
        secondary_interests=st.session_state.w_interests,
        preferred_methodology=st.session_state.w_methods,
        regional_focus=st.session_state.w_region,
        seed_authors=st.session_state.w_seed_authors,
        methodological_lean=st.session_state.w_method_lean
    )
    st.session_state.user_profile = profile
    
    # Progress container
    progress_container = st.empty()
    status_container = st.empty()
    
    with progress_container.container():
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 48px; margin-bottom: 20px;">🔬</div>
            <h2 style="font-family: var(--font-serif); margin-bottom: 10px;">Discovering papers...</h2>
            <p style="color: #6E6E73;" id="status-text">Initializing</p>
        </div>
        """, unsafe_allow_html=True)
    
    try:
        # Step 1: Fetch papers
        with status_container:
            st.info("📚 Fetching papers from OpenAlex...")
        
        papers = cached_fetch_papers(
            tuple(st.session_state.w_journals),
            st.session_state.w_days_back,
            st.session_state.w_max_papers
        )
        
        if not papers:
            progress_container.empty()
            status_container.error("No papers found. Try expanding your date range or journal selection.")
            
            if st.button("← Adjust Settings"):
                st.session_state.wizard_step = 3
                st.rerun()
            return
        
        with status_container:
            st.success(f"✓ Found {len(papers)} papers")
        
        time.sleep(0.5)
        
        # Step 2: AI Analysis
        with status_container:
            st.info("🤖 AI analyzing relevance...")
        
        def progress_cb(msg):
            with status_container:
                st.info(msg)
        
        processed, messages, synthesis = process_papers_with_gemini(
            api_key=st.session_state.api_key,
            user_profile=profile,
            papers=papers,
            use_semantic_prerank=True,
            progress_callback=progress_cb
        )
        
        st.session_state.processed_papers = processed
        st.session_state.status_messages = messages
        st.session_state.synthesis = synthesis
        
        # Clear progress UI
        progress_container.empty()
        status_container.empty()
        
        st.rerun()
        
    except OpenAlexError as e:
        progress_container.empty()
        status_container.error(f"API Error: {str(e)}")
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"Error: {str(e)}")


def render_results():
    """Render the discovery results."""
    papers = st.session_state.processed_papers
    profile = st.session_state.user_profile
    synthesis = st.session_state.synthesis
    messages = st.session_state.status_messages
    
    if not papers:
        st.warning("No papers to display. Please run discovery again.")
        if st.button("← Back to Settings"):
            st.session_state.wizard_step = 3
            st.session_state.processed_papers = []
            st.rerun()
        return
    
    # Header
    st.markdown("""
    <div class="hero" style="padding-bottom: 20px;">
        <h1>Your personalized reading list</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    analyzed = sum(1 for p in papers if p.get("has_ai_analysis"))
    high_rel = sum(1 for p in papers if p.get("relevance_score", 0) >= 8 and p.get("has_ai_analysis"))
    
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-value">{len(papers)}</div>
            <div class="metric-label">Papers Found</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{analyzed}</div>
            <div class="metric-label">AI Analyzed</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{high_rel}</div>
            <div class="metric-label">High Relevance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Synthesis box
    if synthesis:
        st.markdown(f"""
        <div class="synthesis-box">
            <h3>📝 Editor's Synthesis</h3>
            <p>{synthesis}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Status messages
    if messages:
        first_msg = messages[0]
        if "✓" in first_msg:
            st.markdown(f'<div class="status-success">{first_msg}</div>', unsafe_allow_html=True)
        elif "⚠️" in first_msg:
            st.markdown(f'<div class="status-warning">{first_msg}</div>', unsafe_allow_html=True)
        else:
            st.info(first_msg)
        
        if len(messages) > 1:
            with st.expander("View all status messages"):
                for msg in messages[1:]:
                    st.caption(msg)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        min_score = st.slider("Min relevance", 1, 10, 1, key="filter_score")
    with col2:
        sort_by = st.selectbox("Sort by", ["Relevance", "Date", "Citations"], key="filter_sort")
    with col3:
        oa_only = st.checkbox("Open Access only", key="filter_oa")
    with col4:
        if st.button("🔄 New Search"):
            st.session_state.processed_papers = []
            st.session_state.wizard_step = 1
            st.rerun()
    
    # Apply filters
    filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
    if oa_only:
        filtered = [p for p in filtered if p.get("is_open_access")]
    
    # Sort
    if sort_by == "Date":
        filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
    elif sort_by == "Citations":
        filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    else:  # Relevance
        filtered.sort(
            key=lambda x: (x.get("has_ai_analysis", False), x.get("relevance_score", 0)),
            reverse=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results
    if not filtered:
        st.info("No papers match your filters. Try adjusting the minimum relevance score.")
    else:
        for idx, paper in enumerate(filtered):
            render_match_card(paper, idx)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #86868B; font-size: 13px; padding: 20px;">
        Data from <a href="https://openalex.org" target="_blank" style="color: #0071E3;">OpenAlex</a> · 
        AI by <a href="https://ai.google.dev" target="_blank" style="color: #0071E3;">Gemini</a>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Main application entry point."""
    
    # Render wizard progress
    render_wizard_progress(st.session_state.wizard_step)
    
    # Render current step
    if st.session_state.wizard_step == 1:
        render_step_1_profile()
    elif st.session_state.wizard_step == 2:
        render_step_2_interests()
    elif st.session_state.wizard_step == 3:
        render_step_3_sources()
    elif st.session_state.wizard_step == 4:
        render_step_4_results()


if __name__ == "__main__":
    main()
