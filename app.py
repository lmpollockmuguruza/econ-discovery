"""
Economics Literature Discovery - Streamlit Application
A minimalist, Jony Ive-inspired interface for discovering relevant economics papers.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path

from api_client import fetch_recent_papers, get_journal_names
from processor import (
    process_papers_with_gemini,
    create_user_profile,
    get_profile_options
)

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Econ Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MINIMALIST CSS - JONY IVE INSPIRED DESIGN
# ============================================================================

CUSTOM_CSS = """
<style>
    /* ===== GLOBAL RESET & BASE ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --bg-primary: #FFFFFF;
        --bg-secondary: #F5F5F7;
        --bg-card: #FFFFFF;
        --text-primary: #1D1D1F;
        --text-secondary: #6E6E73;
        --text-tertiary: #86868B;
        --border-light: #E8E8ED;
        --accent-blue: #0071E3;
        --accent-green: #34C759;
        --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.04);
        --shadow-card: 0 4px 12px rgba(0, 0, 0, 0.08);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    
    /* Main app background */
    .stApp {
        background-color: var(--bg-secondary) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ===== TYPOGRAPHY ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
        color: var(--text-primary);
    }
    
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background-color: var(--bg-primary) !important;
        border-right: 1px solid var(--border-light) !important;
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1.5rem !important;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-tertiary) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== FORM ELEMENTS ===== */
    /* Select boxes and multiselect */
    [data-testid="stSelectbox"], 
    [data-testid="stMultiSelect"] {
        background-color: var(--bg-secondary) !important;
        border-radius: var(--radius-sm) !important;
    }
    
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.875rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--text-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-blue) !important;
        transform: translateY(-1px);
        box-shadow: var(--shadow-card);
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.875rem !important;
        padding: 0.75rem !important;
    }
    
    /* ===== MAIN CONTENT AREA ===== */
    .main .block-container {
        padding: 3rem 4rem !important;
        max-width: 1200px !important;
    }
    
    /* ===== PAPER CARDS ===== */
    .paper-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.75rem 2rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-subtle);
        border: 1px solid var(--border-light);
        transition: all 0.2s ease;
    }
    
    .paper-card:hover {
        box-shadow: var(--shadow-card);
        transform: translateY(-2px);
    }
    
    .paper-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    
    .paper-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.4;
        margin: 0;
        flex: 1;
        padding-right: 1rem;
    }
    
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 3rem;
        height: 2rem;
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .score-high {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    
    .score-medium {
        background-color: #FFF8E1;
        color: #F57C00;
    }
    
    .score-low {
        background-color: #FAFAFA;
        color: var(--text-tertiary);
    }
    
    .paper-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
        font-size: 0.8125rem;
        color: var(--text-tertiary);
    }
    
    .paper-meta span {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .paper-summary {
        background-color: var(--bg-secondary);
        border-radius: var(--radius-sm);
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .summary-label {
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-tertiary);
        margin-bottom: 0.5rem;
    }
    
    .summary-text {
        font-size: 0.9375rem;
        line-height: 1.6;
        color: var(--text-primary);
        margin: 0;
    }
    
    .why-relevant {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    .paper-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 1rem;
        border-top: 1px solid var(--border-light);
    }
    
    .methodology-tag {
        display: inline-block;
        background-color: var(--bg-secondary);
        color: var(--text-secondary);
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .paper-link {
        color: var(--accent-blue);
        text-decoration: none;
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .paper-link:hover {
        text-decoration: underline;
    }
    
    /* Open Access badge */
    .oa-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.125rem 0.5rem;
        border-radius: 100px;
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    /* ===== STATS BAR ===== */
    .stats-bar {
        display: flex;
        gap: 2rem;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-light);
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== LOADING STATE ===== */
    .loading-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .loading-text {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 1rem;
    }
    
    /* ===== EMPTY STATE ===== */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: var(--text-secondary);
    }
    
    .empty-state h3 {
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem;
    }
    
    /* ===== INTRO HERO ===== */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: var(--text-secondary);
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* ===== FILTER PILLS ===== */
    .filter-pills {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    .filter-pill {
        background-color: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 100px;
        padding: 0.375rem 0.875rem;
        font-size: 0.8125rem;
        color: var(--text-secondary);
    }
    
    .filter-pill.active {
        background-color: var(--text-primary);
        color: white;
        border-color: var(--text-primary);
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        border-top: 1px solid var(--border-light);
        margin: 1.5rem 0;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div {
        background-color: var(--border-light) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--text-primary) !important;
    }
</style>
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_score_class(score: int) -> str:
    """Return CSS class based on relevance score."""
    if score >= 8:
        return "score-high"
    elif score >= 5:
        return "score-medium"
    return "score-low"


def render_paper_card(paper: dict):
    """Render a single paper card with minimalist styling."""
    score = paper.get("relevance_score", 5)
    score_class = get_score_class(score)
    
    # Format authors
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f" +{len(authors) - 3}"
    
    # Format date
    pub_date = paper.get("publication_date", "")
    if pub_date:
        try:
            date_obj = datetime.strptime(pub_date, "%Y-%m-%d")
            pub_date = date_obj.strftime("%b %d, %Y")
        except:
            pass
    
    # Open access indicator
    oa_badge = ""
    if paper.get("is_open_access"):
        oa_badge = '<span class="oa-badge">🔓 Open Access</span>'
    
    # DOI link
    doi = paper.get("doi", "")
    link_html = ""
    if doi:
        link_html = f'<a href="{doi}" target="_blank" class="paper-link">Read Paper →</a>'
    elif paper.get("oa_url"):
        link_html = f'<a href="{paper["oa_url"]}" target="_blank" class="paper-link">Read Paper →</a>'
    
    card_html = f"""
    <div class="paper-card">
        <div class="paper-header">
            <h3 class="paper-title">{paper.get('title', 'Untitled')}</h3>
            <span class="score-badge {score_class}">{score}/10</span>
        </div>
        
        <div class="paper-meta">
            <span>📖 {paper.get('journal', 'Unknown Journal')}</span>
            <span>👤 {author_str}</span>
            <span>📅 {pub_date}</span>
            {oa_badge}
        </div>
        
        <div class="paper-summary">
            <div class="summary-label">Why You Should Read This</div>
            <p class="summary-text">{paper.get('ai_contribution', 'Summary not available.')}</p>
            <p class="why-relevant">{paper.get('ai_relevance', '')}</p>
        </div>
        
        <div class="paper-footer">
            <span class="methodology-tag">{paper.get('ai_methodology', 'Unknown Method')}</span>
            {link_html}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def load_cached_data() -> dict:
    """Load cached papers and profile from JSON file."""
    cache_path = Path("cache.json")
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {"papers": [], "profile": None, "last_fetch": None}


def save_cached_data(data: dict):
    """Save papers and profile to JSON file."""
    cache_path = Path("cache.json")
    with open(cache_path, "w") as f:
        json.dump(data, f)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Load options
    options = get_profile_options()
    journals = get_journal_names()
    
    # Initialize session state
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "processed_papers" not in st.session_state:
        st.session_state.processed_papers = []
    if "profile" not in st.session_state:
        st.session_state.profile = None
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("# ⚙️ Settings")
        
        # API Key
        st.markdown("### API Key")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key at makersuite.google.com",
            label_visibility="collapsed",
            placeholder="Enter your Gemini API key"
        )
        
        st.markdown("---")
        
        # Interest Matrix
        st.markdown("### Your Research Profile")
        
        academic_level = st.selectbox(
            "Academic Level",
            options=options["academic_levels"],
            index=2  # Default to PhD Student
        )
        
        primary_field = st.selectbox(
            "Primary Field",
            options=options["primary_fields"],
            index=0
        )
        
        secondary_interests = st.multiselect(
            "Secondary Interests",
            options=options["secondary_interests"],
            default=["Causal Inference"],
            max_selections=5
        )
        
        preferred_methods = st.multiselect(
            "Preferred Methodologies",
            options=options["methodologies"],
            default=["Difference-in-Differences"],
            max_selections=4
        )
        
        st.markdown("---")
        
        # Journal Selection
        st.markdown("### Source Journals")
        
        selected_journals = st.multiselect(
            "Select Journals",
            options=journals,
            default=journals[:5],  # Top 5 by default
            label_visibility="collapsed"
        )
        
        # Time Range
        days_back = st.slider(
            "Days to Look Back",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
        
        st.markdown("---")
        
        # Fetch Button
        fetch_clicked = st.button(
            "🔍 Discover Papers",
            use_container_width=True
        )
    
    # ========== MAIN CONTENT ==========
    
    # Hero section when no papers loaded
    if not st.session_state.processed_papers:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Econ Discovery</h1>
            <p class="hero-subtitle">
                Intelligent literature discovery for economists. 
                Find the papers that matter to your research.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="empty-state">
            <h3>Get Started</h3>
            <p>Configure your research profile in the sidebar, add your Gemini API key, 
            and click "Discover Papers" to find relevant new publications.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Process fetch request
    if fetch_clicked:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
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
                st.success(f"Found {len(papers)} papers. Analyzing relevance...")
                
                # Process with Gemini
                with st.spinner("AI is analyzing papers for your interests..."):
                    try:
                        processed = process_papers_with_gemini(
                            api_key=api_key,
                            user_profile=profile,
                            papers=papers,
                            batch_size=10
                        )
                        st.session_state.processed_papers = processed
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing papers: {str(e)}")
            else:
                st.warning("No papers found for the selected criteria. Try expanding your date range or journal selection.")
    
    # Display processed papers
    if st.session_state.processed_papers:
        papers = st.session_state.processed_papers
        
        # Stats bar
        high_relevance = len([p for p in papers if p.get("relevance_score", 0) >= 8])
        avg_score = sum(p.get("relevance_score", 0) for p in papers) / len(papers)
        
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-number">{len(papers)}</div>
                <div class="stat-label">Papers Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{high_relevance}</div>
                <div class="stat-label">High Relevance</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{avg_score:.1f}</div>
                <div class="stat-label">Avg Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            min_score = st.select_slider(
                "Minimum Relevance",
                options=list(range(1, 11)),
                value=1
            )
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                options=["Relevance Score", "Publication Date", "Citations"],
                index=0
            )
        with col3:
            show_oa_only = st.checkbox("Open Access Only")
        
        st.markdown("---")
        
        # Filter and sort papers
        filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
        if show_oa_only:
            filtered = [p for p in filtered if p.get("is_open_access")]
        
        if sort_by == "Publication Date":
            filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
        elif sort_by == "Citations":
            filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
        else:
            filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Render paper cards
        if filtered:
            for paper in filtered:
                render_paper_card(paper)
        else:
            st.markdown("""
            <div class="empty-state">
                <h3>No Papers Match Your Filters</h3>
                <p>Try lowering the minimum relevance score or adjusting other filters.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
