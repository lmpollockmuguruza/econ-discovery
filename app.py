"""
Literature Discovery - Streamlit Application
A clean interface for discovering relevant academic papers.
Built with native Streamlit components for reliability.
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
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MINIMAL CSS - ONLY FOR COLORS/FONTS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    footer {visibility: hidden;}
    
    .stButton > button[kind="primary"] {
        background-color: #0a0a0a;
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #333;
        border: none;
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_date(date_str: str) -> str:
    if not date_str:
        return "Unknown date"
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%b %d, %Y")
    except:
        return date_str


def get_score_emoji(score: int) -> str:
    if score >= 8:
        return "🟢"
    elif score >= 5:
        return "🟡"
    return "⚪"


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    # API KEY
    st.subheader("🔑 API Key")
    st.markdown("Get a free key at [Google AI Studio](https://makersuite.google.com/app/apikey)")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # RESEARCH PROFILE
    st.subheader("👤 Your Profile")
    
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
        "Interests (up to 5)",
        options=options["secondary_interests"],
        default=["Causal Inference"],
        max_selections=5
    )
    
    preferred_methods = st.multiselect(
        "Methods (up to 4)",
        options=options["methodologies"],
        default=["Difference-in-Differences"],
        max_selections=4
    )
    
    st.divider()
    
    # JOURNALS
    st.subheader("📚 Journals")
    
    field_choice = st.radio(
        "Field",
        ["Economics", "Political Science", "Both"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if field_choice == "Economics":
        available_journals = get_economics_journals()
    elif field_choice == "Political Science":
        available_journals = get_polisci_journals()
    else:
        available_journals = get_all_journals()
    
    selected_journals = st.multiselect(
        "Journals",
        options=available_journals,
        default=available_journals[:5],
        label_visibility="collapsed"
    )
    
    days_back = st.slider("Days back", 7, 90, 30, step=7)
    
    st.divider()
    
    # ACTION
    fetch_clicked = st.button(
        "🔍 Discover Papers", 
        use_container_width=True, 
        type="primary"
    )


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📚 Literature Discovery")
st.caption("Find papers that matter to your research, ranked by AI.")

st.divider()

# Handle fetch
if fetch_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not selected_journals:
        st.error("Please select at least one journal.")
    elif not secondary_interests:
        st.error("Please select at least one interest.")
    else:
        profile = create_user_profile(
            academic_level=academic_level,
            primary_field=primary_field,
            secondary_interests=secondary_interests,
            preferred_methodology=preferred_methods
        )
        
        with st.spinner("Fetching papers from OpenAlex..."):
            papers = fetch_recent_papers(
                days_back=days_back,
                selected_journals=selected_journals,
                max_results=50
            )
            st.session_state.papers = papers
        
        if papers:
            st.success(f"Found {len(papers)} papers!")
            
            with st.spinner("AI is analyzing relevance..."):
                try:
                    processed = process_papers_with_gemini(
                        api_key=api_key,
                        user_profile=profile,
                        papers=papers,
                        batch_size=8
                    )
                    st.session_state.processed_papers = processed
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("No papers found. Try a longer time range.")


# Display results
if st.session_state.processed_papers:
    papers = st.session_state.processed_papers
    
    # Stats
    col1, col2, col3 = st.columns(3)
    high_rel = len([p for p in papers if p.get("relevance_score", 0) >= 8])
    avg = sum(p.get("relevance_score", 0) for p in papers) / len(papers) if papers else 0
    
    col1.metric("Papers", len(papers))
    col2.metric("High Relevance", high_rel)
    col3.metric("Avg Score", f"{avg:.1f}")
    
    st.divider()
    
    # Filters
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        min_score = st.slider("Min score", 1, 10, 1)
    with f2:
        sort_by = st.selectbox("Sort", ["Relevance", "Date", "Citations"])
    with f3:
        oa_only = st.checkbox("Open Access")
    
    # Apply filters
    filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
    if oa_only:
        filtered = [p for p in filtered if p.get("is_open_access")]
    
    # Sort
    if sort_by == "Date":
        filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
    elif sort_by == "Citations":
        filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    st.divider()
    
    # Papers
    if not filtered:
        st.info("No papers match your filters.")
    else:
        for paper in filtered:
            score = paper.get("relevance_score", 5)
            emoji = get_score_emoji(score)
            
            with st.container(border=True):
                # Title and score
                tcol, scol = st.columns([5, 1])
                with tcol:
                    st.markdown(f"**{paper.get('title', 'Untitled')}**")
                with scol:
                    st.markdown(f"**{emoji} {score}/10**")
                
                # Meta
                authors = paper.get("authors", [])
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += f" +{len(authors) - 3}"
                
                meta = f"📖 {paper.get('journal', 'Unknown')}"
                if author_str:
                    meta += f" · 👤 {author_str}"
                meta += f" · 📅 {format_date(paper.get('publication_date'))}"
                if paper.get("is_open_access"):
                    meta += " · 🔓 Open Access"
                
                st.caption(meta)
                
                # AI Analysis
                contribution = paper.get("ai_contribution", "")
                relevance = paper.get("ai_relevance", "")
                
                if contribution or relevance:
                    st.markdown("**Why read this:**")
                    if contribution:
                        st.write(contribution)
                    if relevance:
                        st.caption(f"_{relevance}_")
                
                # Footer
                method = paper.get("ai_methodology", "")
                link = paper.get("doi") or paper.get("oa_url")
                
                fcol1, fcol2 = st.columns([3, 1])
                with fcol1:
                    if method:
                        st.caption(f"📊 {method}")
                with fcol2:
                    if link:
                        st.link_button("Read →", link)

else:
    # Empty state
    st.info("👈 Set up your profile in the sidebar and click **Discover Papers**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### How it works")
        st.markdown("""
        1. Enter your Gemini API key
        2. Set your research interests  
        3. Choose journals to search
        4. Click Discover Papers
        5. AI ranks by relevance to you
        """)
    
    with c2:
        st.markdown("### Available sources")
        st.markdown(f"**{len(get_economics_journals())}** Economics journals")
        st.markdown(f"**{len(get_polisci_journals())}** Political Science journals")

st.divider()
st.caption("Data: [OpenAlex](https://openalex.org) · AI: [Gemini](https://ai.google.dev)")
