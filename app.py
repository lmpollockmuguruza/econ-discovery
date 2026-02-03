"""
Literature Discovery - Streamlit Application
A clean interface for discovering relevant academic papers.
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
# MINIMAL CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp { background-color: #FAFAFA; }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    footer { visibility: hidden; }
    
    .stButton > button[kind="primary"] {
        background-color: #0a0a0a;
        color: white;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if "processed_papers" not in st.session_state:
    st.session_state.processed_papers = []
if "ai_errors" not in st.session_state:
    st.session_state.ai_errors = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None


# ============================================================================
# HELPERS
# ============================================================================

def format_date(date_str: str) -> str:
    if not date_str:
        return "Unknown"
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d, %Y")
    except:
        return date_str


def get_score_display(score: int, has_ai: bool) -> str:
    """Return score with color indicator."""
    if not has_ai:
        return "⚪ ?/10"  # No AI analysis
    if score >= 8:
        return f"🟢 {score}/10"
    elif score >= 5:
        return f"🟡 {score}/10"
    return f"🔴 {score}/10"


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    # API KEY
    st.subheader("🔑 API Key")
    st.markdown("[Get free key →](https://makersuite.google.com/app/apikey)")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste Gemini API key",
        label_visibility="collapsed"
    )
    
    if api_key:
        st.success("Key entered", icon="✅")
    
    st.divider()
    
    # PROFILE
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
        "Interests",
        options=options["secondary_interests"],
        default=["Causal Inference"],
        max_selections=5,
        help="Up to 5"
    )
    
    preferred_methods = st.multiselect(
        "Methods",
        options=options["methodologies"],
        default=["Difference-in-Differences"],
        max_selections=4,
        help="Up to 4"
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
        available = get_economics_journals()
    elif field_choice == "Political Science":
        available = get_polisci_journals()
    else:
        available = get_all_journals()
    
    selected_journals = st.multiselect(
        "Journals",
        options=available,
        default=available[:5],
        label_visibility="collapsed"
    )
    
    days_back = st.slider("Days back", 7, 90, 30, step=7)
    
    st.divider()
    
    fetch_clicked = st.button(
        "🔍 Discover Papers", 
        use_container_width=True, 
        type="primary"
    )


# ============================================================================
# MAIN
# ============================================================================

st.title("📚 Literature Discovery")
st.caption("Papers ranked by AI based on YOUR research interests")

st.divider()

# Process request
if fetch_clicked:
    # Validation
    errors = []
    if not api_key:
        errors.append("Enter your Gemini API key")
    if not selected_journals:
        errors.append("Select at least one journal")
    if not secondary_interests:
        errors.append("Select at least one interest")
    
    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
    else:
        # Create profile
        profile = create_user_profile(
            academic_level=academic_level,
            primary_field=primary_field,
            secondary_interests=secondary_interests,
            preferred_methodology=preferred_methods
        )
        st.session_state.user_profile = profile
        
        # Show what we're searching for
        with st.status("Searching...", expanded=True) as status:
            st.write(f"**Your profile:** {primary_field} · {academic_level}")
            st.write(f"**Interests:** {', '.join(secondary_interests)}")
            st.write(f"**Methods:** {', '.join(preferred_methods)}")
            
            st.write("---")
            st.write("🔍 Fetching papers from OpenAlex...")
            
            papers = fetch_recent_papers(
                days_back=days_back,
                selected_journals=selected_journals,
                max_results=30  # Reduced for faster processing
            )
            
            if not papers:
                status.update(label="No papers found", state="error")
                st.error("No papers found. Try a longer time range or different journals.")
            else:
                st.write(f"✓ Found {len(papers)} papers")
                st.write("---")
                st.write("🤖 AI is analyzing relevance to your interests...")
                
                # Process with Gemini
                try:
                    processed, ai_errors = process_papers_with_gemini(
                        api_key=api_key,
                        user_profile=profile,
                        papers=papers,
                        batch_size=5
                    )
                    
                    st.session_state.processed_papers = processed
                    st.session_state.ai_errors = ai_errors
                    
                    # Count successful analyses
                    analyzed = sum(1 for p in processed if p.get("has_ai_analysis"))
                    
                    if analyzed > 0:
                        status.update(label=f"✓ Analyzed {analyzed} papers!", state="complete")
                    else:
                        status.update(label="⚠️ AI analysis failed", state="error")
                    
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Error: {str(e)}")


# Show errors if any
if st.session_state.ai_errors:
    with st.expander("⚠️ Some issues occurred", expanded=False):
        for err in st.session_state.ai_errors:
            st.warning(err)


# Display results
if st.session_state.processed_papers:
    papers = st.session_state.processed_papers
    profile = st.session_state.user_profile
    
    # Stats
    analyzed = sum(1 for p in papers if p.get("has_ai_analysis"))
    high_rel = sum(1 for p in papers if p.get("relevance_score", 0) >= 8 and p.get("has_ai_analysis"))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", len(papers))
    c2.metric("AI Analyzed", f"{analyzed}/{len(papers)}")
    c3.metric("High Relevance", high_rel)
    
    # Show current profile
    if profile:
        st.caption(f"📊 Ranked for: **{profile['primary_field']}** researcher interested in **{', '.join(profile['secondary_interests'][:3])}**")
    
    st.divider()
    
    # Filters
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        min_score = st.slider("Min score", 1, 10, 1)
    with f2:
        sort_by = st.selectbox("Sort", ["Relevance", "Date", "Citations"])
    with f3:
        oa_only = st.checkbox("Open Access")
    
    # Filter
    filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
    if oa_only:
        filtered = [p for p in filtered if p.get("is_open_access")]
    
    # Sort
    if sort_by == "Date":
        filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
    elif sort_by == "Citations":
        filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    else:
        filtered.sort(key=lambda x: (x.get("has_ai_analysis", False), x.get("relevance_score", 0)), reverse=True)
    
    st.divider()
    
    # Papers
    if not filtered:
        st.info("No papers match your filters.")
    else:
        for paper in filtered:
            score = paper.get("relevance_score", 5)
            has_ai = paper.get("has_ai_analysis", False)
            
            with st.container(border=True):
                # Header
                tcol, scol = st.columns([5, 1])
                with tcol:
                    st.markdown(f"**{paper.get('title', 'Untitled')}**")
                with scol:
                    st.markdown(f"**{get_score_display(score, has_ai)}**")
                
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
                    meta += " · 🔓 Open"
                st.caption(meta)
                
                # AI Analysis
                if has_ai:
                    contribution = paper.get("ai_contribution", "")
                    relevance = paper.get("ai_relevance", "")
                    
                    if contribution:
                        st.markdown("**📝 Summary:**")
                        st.write(contribution)
                    
                    if relevance:
                        st.markdown("**🎯 Why relevant to you:**")
                        st.info(relevance)
                else:
                    st.caption("_AI analysis not available for this paper_")
                
                # Footer
                method = paper.get("ai_methodology", "")
                link = paper.get("doi") or paper.get("oa_url")
                
                fcol1, fcol2 = st.columns([3, 1])
                with fcol1:
                    if method:
                        st.caption(f"📊 Method: {method}")
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
        1. Get a free [Gemini API key](https://makersuite.google.com/app/apikey)
        2. Set your research field & interests
        3. Select journals to search
        4. Click **Discover Papers**
        5. AI scores each paper's relevance to YOU
        """)
    
    with c2:
        st.markdown("### What makes it personal")
        st.markdown("""
        The AI reads each paper's abstract and evaluates:
        - Does it match your **primary field**?
        - Does it cover your **interests**?
        - Does it use your **preferred methods**?
        
        Papers are scored 1-10 based on YOUR profile.
        """)


st.divider()
st.caption("Data: [OpenAlex](https://openalex.org) · AI: [Gemini](https://ai.google.dev)")
