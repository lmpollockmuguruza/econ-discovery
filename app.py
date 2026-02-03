"""
Literature Discovery - Streamlit Application
A minimalist interface for discovering relevant academic papers.
"""

import streamlit as st
from datetime import datetime

from api_client import (
    fetch_recent_papers, 
    get_economics_journals, 
    get_polisci_journals,
    get_all_journals
)
from processor import (
    process_papers_with_gemini,
    create_user_profile,
    get_profile_options
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Literature Discovery",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - INSPIRED BY PREMIER LEAGUE QUIZ AESTHETIC
# ============================================================================

st.markdown("""
<style>
    /* Import Instrument Sans */
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
    
    /* Global Reset */
    *, *::before, *::after {
        box-sizing: border-box;
    }
    
    /* Root variables */
    :root {
        --bg: #FAFAF8;
        --text: #0a0a0a;
        --text-secondary: #666666;
        --text-muted: #999999;
        --border: #e5e5e5;
        --card-bg: #ffffff;
    }
    
    /* Main app styling */
    .stApp {
        background-color: var(--bg) !important;
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    }
    
    /* Hide default Streamlit elements - but keep sidebar toggle visible */
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Ensure sidebar toggle is always visible */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        color: #0a0a0a !important;
        background: #ffffff !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        margin: 1rem !important;
    }
    
    /* All text should use our font */
    html, body, [class*="css"], .stMarkdown, p, span, div, h1, h2, h3, h4, label {
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 900px !important;
    }
    
    /* Headers */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
        color: var(--text) !important;
        margin-bottom: 0.25rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
    }
    
    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: var(--text) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid var(--border) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: var(--text-muted) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Form inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    
    .stTextInput > div > div > input {
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.9rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--text) !important;
        box-shadow: none !important;
    }
    
    /* Primary button */
    .stButton > button {
        background-color: var(--text) !important;
        color: var(--bg) !important;
        border: none !important;
        border-radius: 100px !important;
        padding: 0.75rem 1.75rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: var(--border) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--text) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    /* Checkbox */
    .stCheckbox label span {
        font-size: 0.9rem !important;
        color: var(--text) !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Success/Error/Warning boxes */
    .stAlert {
        border-radius: 8px !important;
        font-size: 0.9rem !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--text) transparent transparent transparent !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        padding: 0.75rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--text) !important;
        border-bottom-color: var(--text) !important;
        background-color: transparent !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #d0d0d0;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #b0b0b0;
    }
    
    /* Selection */
    ::selection {
        background: var(--text);
        color: var(--bg);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if "papers" not in st.session_state:
    st.session_state.papers = []
if "processed_papers" not in st.session_state:
    st.session_state.processed_papers = []
if "profile" not in st.session_state:
    st.session_state.profile = None
if "fetch_error" not in st.session_state:
    st.session_state.fetch_error = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_score_color(score: int) -> str:
    """Return color based on relevance score."""
    if score >= 8:
        return "#2E7D32"  # green
    elif score >= 5:
        return "#F57C00"  # orange
    return "#999999"  # gray


def format_date(date_str: str) -> str:
    """Format publication date nicely."""
    if not date_str:
        return ""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d, %Y")
    except:
        return date_str


def render_paper_card(paper: dict, index: int):
    """Render a single paper using native Streamlit components."""
    score = paper.get("relevance_score", 5)
    score_color = get_score_color(score)
    
    # Authors
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" +{len(authors) - 3}"
    
    # Create card container
    with st.container():
        # Add custom styling for this card
        st.markdown(f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                <h3 style="
                    font-size: 1.05rem;
                    font-weight: 600;
                    color: #0a0a0a;
                    margin: 0;
                    line-height: 1.4;
                    flex: 1;
                    padding-right: 1rem;
                ">{paper.get('title', 'Untitled')}</h3>
                <span style="
                    background: {score_color}15;
                    color: {score_color};
                    padding: 0.25rem 0.75rem;
                    border-radius: 100px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    white-space: nowrap;
                ">{score}/10</span>
            </div>
            
            <div style="
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 1rem;
            ">
                <span>📖 {paper.get('journal', 'Unknown')}</span>
                <span>👤 {author_str or 'Unknown'}</span>
                <span>📅 {format_date(paper.get('publication_date', ''))}</span>
                {'<span style="background: #FFF3E0; color: #E65100; padding: 0.125rem 0.5rem; border-radius: 100px; font-size: 0.7rem; font-weight: 500;">OPEN ACCESS</span>' if paper.get('is_open_access') else ''}
            </div>
        """, unsafe_allow_html=True)
        
        # AI Summary section
        contribution = paper.get('ai_contribution', '')
        relevance = paper.get('ai_relevance', '')
        methodology = paper.get('ai_methodology', '')
        
        if contribution or relevance:
            st.markdown(f"""
            <div style="
                background: #FAFAF8;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <div style="
                    font-size: 0.65rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #999;
                    margin-bottom: 0.5rem;
                ">why you should read this</div>
                <p style="
                    font-size: 0.9rem;
                    line-height: 1.6;
                    color: #0a0a0a;
                    margin: 0 0 0.5rem 0;
                ">{contribution}</p>
                <p style="
                    font-size: 0.85rem;
                    color: #666;
                    font-style: italic;
                    margin: 0;
                ">{relevance}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer with methodology and link
        doi = paper.get("doi", "")
        oa_url = paper.get("oa_url", "")
        link = doi or oa_url
        
        st.markdown(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 0.75rem;
                border-top: 1px solid #e5e5e5;
            ">
                <span style="
                    background: #f0f0f0;
                    color: #666;
                    padding: 0.25rem 0.75rem;
                    border-radius: 100px;
                    font-size: 0.75rem;
                    font-weight: 500;
                ">{methodology or 'Unknown method'}</span>
                {'<a href="' + link + '" target="_blank" style="color: #0a0a0a; text-decoration: none; font-size: 0.85rem; font-weight: 500;">Read paper →</a>' if link else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Key section with instructions
    st.markdown("### api key")
    
    with st.expander("ℹ️ How to get a free API key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click **"Create API key"**
        4. Copy and paste it below
        
        *It's free and takes 30 seconds.*
        """)
    
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your API key here",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Research Profile
    st.markdown("### your research profile")
    
    options = get_profile_options()
    
    academic_level = st.selectbox(
        "Academic Level",
        options=options["academic_levels"],
        index=2
    )
    
    primary_field = st.selectbox(
        "Primary Field",
        options=options["primary_fields"],
        index=0
    )
    
    secondary_interests = st.multiselect(
        "Secondary Interests (up to 5)",
        options=options["secondary_interests"],
        default=["Causal Inference"],
        max_selections=5
    )
    
    preferred_methods = st.multiselect(
        "Preferred Methods (up to 4)",
        options=options["methodologies"],
        default=["Difference-in-Differences"],
        max_selections=4
    )
    
    st.markdown("---")
    
    # Journal Selection with tabs
    st.markdown("### source journals")
    
    journal_type = st.radio(
        "Field",
        options=["Economics", "Political Science", "Both"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if journal_type == "Economics":
        available_journals = get_economics_journals()
    elif journal_type == "Political Science":
        available_journals = get_polisci_journals()
    else:
        available_journals = get_all_journals()
    
    selected_journals = st.multiselect(
        "Select Journals",
        options=available_journals,
        default=available_journals[:5],
        label_visibility="collapsed"
    )
    
    # Time range
    days_back = st.slider(
        "Days to look back",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    st.markdown("---")
    
    # Action button
    fetch_clicked = st.button("🔍 Discover Papers", use_container_width=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero section when no papers
if not st.session_state.processed_papers:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 1rem;">
        <h1 style="
            font-size: clamp(2rem, 6vw, 3rem);
            font-weight: 600;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
        ">Literature Discovery</h1>
        <p style="
            font-size: 1.1rem;
            color: #666;
            max-width: 400px;
            margin: 0 auto 2rem;
            line-height: 1.6;
        ">Find the papers that matter to your research, ranked by AI based on your interests.</p>
        <div style="
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
        ">
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 600; color: #0a0a0a;">20</div>
                <div style="font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em;">journals</div>
            </div>
            <div style="width: 1px; background: #e5e5e5;"></div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 600; color: #0a0a0a;">AI</div>
                <div style="font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em;">powered</div>
            </div>
        </div>
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: #0a0a0a;
            color: #fff;
            padding: 0.75rem 1.5rem;
            border-radius: 100px;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 1.2rem;">←</span>
            <span>Click the arrow to open settings</span>
        </div>
        <p style="font-size: 0.85rem; color: #999;">
            Configure your profile, then click <strong>Discover Papers</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Process fetch request
if fetch_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key. Click the info box in the sidebar for instructions.")
    elif not selected_journals:
        st.error("Please select at least one journal.")
    elif not secondary_interests:
        st.error("Please select at least one secondary interest.")
    else:
        # Create user profile
        profile = create_user_profile(
            academic_level=academic_level,
            primary_field=primary_field,
            secondary_interests=secondary_interests,
            preferred_methodology=preferred_methods
        )
        st.session_state.profile = profile
        
        # Fetch papers
        with st.spinner("Fetching recent papers from OpenAlex..."):
            papers = fetch_recent_papers(
                days_back=days_back,
                selected_journals=selected_journals,
                max_results=50
            )
            st.session_state.papers = papers
        
        if papers:
            st.success(f"Found {len(papers)} papers. Now analyzing with AI...")
            
            # Process with Gemini
            with st.spinner("AI is analyzing papers based on your interests..."):
                try:
                    processed = process_papers_with_gemini(
                        api_key=api_key,
                        user_profile=profile,
                        papers=papers,
                        batch_size=8
                    )
                    st.session_state.processed_papers = processed
                    st.session_state.fetch_error = None
                    st.rerun()
                except Exception as e:
                    st.session_state.fetch_error = str(e)
                    st.error(f"Error processing papers: {str(e)}")
        else:
            st.warning("No papers found. Try expanding your date range or selecting different journals.")


# Display processed papers
if st.session_state.processed_papers:
    papers = st.session_state.processed_papers
    
    # Stats bar
    high_relevance = len([p for p in papers if p.get("relevance_score", 0) >= 8])
    avg_score = sum(p.get("relevance_score", 0) for p in papers) / len(papers)
    
    st.markdown(f"""
    <div style="
        display: flex;
        gap: 2rem;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e5e5e5;
    ">
        <div>
            <span style="font-size: 1.5rem; font-weight: 600; color: #0a0a0a;">{len(papers)}</span>
            <span style="font-size: 0.75rem; color: #999; margin-left: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">papers found</span>
        </div>
        <div>
            <span style="font-size: 1.5rem; font-weight: 600; color: #2E7D32;">{high_relevance}</span>
            <span style="font-size: 0.75rem; color: #999; margin-left: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">high relevance</span>
        </div>
        <div>
            <span style="font-size: 1.5rem; font-weight: 600; color: #0a0a0a;">{avg_score:.1f}</span>
            <span style="font-size: 0.75rem; color: #999; margin-left: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">avg score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        min_score = st.select_slider(
            "Minimum relevance score",
            options=list(range(1, 11)),
            value=1
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Relevance Score", "Publication Date", "Citations"],
            index=0
        )
    
    with col3:
        show_oa_only = st.checkbox("Open Access only")
    
    st.markdown("---")
    
    # Filter and sort
    filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
    if show_oa_only:
        filtered = [p for p in filtered if p.get("is_open_access")]
    
    if sort_by == "Publication Date":
        filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
    elif sort_by == "Citations":
        filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Render papers
    if filtered:
        for i, paper in enumerate(filtered):
            render_paper_card(paper, i)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #666;">
            <h3 style="color: #0a0a0a; margin-bottom: 0.5rem;">No papers match your filters</h3>
            <p>Try lowering the minimum relevance score or adjusting other filters.</p>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 2rem;
    font-size: 0.75rem;
    color: #999;
">
    Data from <a href="https://openalex.org" target="_blank" style="color: #666;">OpenAlex</a> · 
    AI by <a href="https://ai.google.dev" target="_blank" style="color: #666;">Google Gemini</a>
</div>
""", unsafe_allow_html=True)
