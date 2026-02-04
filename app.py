"""
Econvery — Discover research that matters to you.
"""

import streamlit as st
from datetime import datetime
import time

from api_client import (
    fetch_recent_papers, get_economics_journals, get_polisci_journals,
    get_all_journals, get_journal_options, OpenAlexError
)
from processor import (
    process_papers, create_profile, get_profile_options, UserProfile
)

st.set_page_config(
    page_title="Econvery",
    page_icon="E",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS - STYLING (theme colors handled by config.toml)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
    }
    
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    .block-container {
        max-width: 600px;
        padding: 3rem 1.5rem;
    }
    
    /* ===== CUSTOM TEXT CLASSES ===== */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 300 !important;
        text-align: center;
        margin-bottom: 0.25rem;
        color: #1a1a1a !important;
        letter-spacing: -0.03em;
    }
    
    .main-sub {
        font-size: 1rem !important;
        text-align: center;
        color: #666666 !important;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .step-label {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888888 !important;
        margin-bottom: 0.5rem;
    }
    
    .step-title {
        font-size: 1.75rem !important;
        font-weight: 400 !important;
        color: #1a1a1a !important;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .step-hint {
        font-size: 0.9rem !important;
        color: #666666 !important;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    .greeting {
        font-size: 1.1rem !important;
        color: #1a1a1a !important;
        margin-bottom: 2rem;
    }
    
    /* ===== BUTTONS ===== */
    .stButton button {
        width: 100%;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* ===== FORM ELEMENTS ===== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 8px !important;
    }
    
    .stTextInput input {
        border-radius: 8px !important;
    }
    
    /* ===== PROGRESS DOTS ===== */
    .progress-dots {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-bottom: 3rem;
    }
    
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #dddddd;
    }
    
    .dot.active { background-color: #1a1a1a; }
    .dot.done { background-color: #1a1a1a; }
    
    /* ===== PAPER CARDS ===== */
    .paper-card {
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
    }
    
    .paper-card:hover {
        border-color: #cccccc;
    }
    
    .paper-title {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #1a1a1a !important;
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }
    
    .paper-meta {
        font-size: 0.8rem !important;
        color: #666666 !important;
        margin-bottom: 0.75rem;
    }
    
    .paper-match {
        font-size: 0.85rem !important;
        color: #444444 !important;
        background-color: #f5f5f5;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    
    .paper-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    
    .tag {
        font-size: 0.7rem !important;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background-color: #f0f0f0;
        color: #555555 !important;
    }
    
    .tag-method { background-color: #e8f4fd; color: #1a5f7a !important; }
    .tag-interest { background-color: #f3e8fd; color: #5a1a7a !important; }
    
    .score {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
    }
    
    .score-high { color: #1a7a3e !important; }
    .score-med { color: #7a5a1a !important; }
    
    .summary-box {
        background-color: #f0f0f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .summary-num {
        font-size: 2.5rem !important;
        font-weight: 300 !important;
        color: #1a1a1a !important;
    }
    
    .summary-label {
        font-size: 0.8rem !important;
        color: #666666 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOTAL_STEPS = 7

def init():
    defaults = {
        "step": 1,
        "name": "",
        "level": "PhD Student",
        "field": "Labor Economics",
        "interests": [],
        "methods": [],
        "region": "United States",
        "journals": [],
        "days": 30,
        "papers": [],
        "summary": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


def dots(current):
    html = '<div class="progress-dots">'
    for i in range(1, TOTAL_STEPS + 1):
        if i < current:
            html += '<div class="dot done"></div>'
        elif i == current:
            html += '<div class="dot active"></div>'
        else:
            html += '<div class="dot"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def fmt_date(d):
    if not d:
        return ""
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y")
    except:
        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_welcome():
    st.markdown('<div class="main-title">Econvery</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Discover research that matters to you.</div>', unsafe_allow_html=True)
    
    dots(1)
    
    st.markdown('<div class="step-label">Let\'s start</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">What\'s your name?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">We\'ll personalize your experience.</div>', unsafe_allow_html=True)
    
    name = st.text_input("Name", value=st.session_state.name, placeholder="First name", label_visibility="collapsed")
    st.session_state.name = name
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Continue", type="primary", disabled=not name.strip()):
        st.session_state.step = 2
        st.rerun()


def step_level():
    dots(2)
    
    st.markdown(f'<div class="greeting">Nice to meet you, {st.session_state.name}.</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">What\'s your career stage?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">This helps us calibrate recommendations.</div>', unsafe_allow_html=True)
    
    opts = get_profile_options()
    idx = opts["academic_levels"].index(st.session_state.level) if st.session_state.level in opts["academic_levels"] else 2
    level = st.selectbox("Level", opts["academic_levels"], index=idx, label_visibility="collapsed")
    st.session_state.level = level
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", type="secondary"):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("Continue", type="primary"):
            st.session_state.step = 3
            st.rerun()


def step_field():
    dots(3)
    
    st.markdown('<div class="step-title">What\'s your primary field?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">Choose the area closest to your research.</div>', unsafe_allow_html=True)
    
    opts = get_profile_options()
    idx = opts["primary_fields"].index(st.session_state.field) if st.session_state.field in opts["primary_fields"] else 3
    field = st.selectbox("Field", opts["primary_fields"], index=idx, label_visibility="collapsed")
    st.session_state.field = field
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", type="secondary"):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("Continue", type="primary"):
            st.session_state.step = 4
            st.rerun()


def step_interests():
    dots(4)
    
    st.markdown('<div class="step-title">What topics interest you?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">Select up to 5, in order of priority. First picks matter more.</div>', unsafe_allow_html=True)
    
    opts = get_profile_options()
    interests = st.multiselect("Interests", opts["interests"], default=st.session_state.interests, max_selections=5, label_visibility="collapsed")
    st.session_state.interests = interests
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", type="secondary"):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("Continue", type="primary", disabled=len(interests) == 0):
            st.session_state.step = 5
            st.rerun()


def step_methods():
    dots(5)
    
    st.markdown('<div class="step-title">Preferred methodologies?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">Select up to 4 methods you care about most.</div>', unsafe_allow_html=True)
    
    opts = get_profile_options()
    methods = st.multiselect("Methods", opts["methods"], default=st.session_state.methods, max_selections=4, label_visibility="collapsed")
    st.session_state.methods = methods
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", type="secondary"):
            st.session_state.step = 4
            st.rerun()
    with c2:
        if st.button("Continue", type="primary", disabled=len(methods) == 0):
            st.session_state.step = 6
            st.rerun()


def step_sources():
    dots(6)
    
    st.markdown('<div class="step-title">Which journals?</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-hint">Choose the journals and time range to search.</div>', unsafe_allow_html=True)
    
    field_type = st.radio("Field", ["Economics", "Political Science", "Both"], horizontal=True, label_visibility="collapsed")
    
    if field_type == "Economics":
        avail = get_economics_journals()
    elif field_type == "Political Science":
        avail = get_polisci_journals()
    else:
        avail = get_all_journals()
    
    jopts = get_journal_options()
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Top journals", use_container_width=True):
            if field_type == "Economics":
                st.session_state.journals = jopts["economics"]["tier1"]
            elif field_type == "Political Science":
                st.session_state.journals = jopts["polisci"]["tier1"]
            else:
                st.session_state.journals = jopts["economics"]["tier1"][:3] + jopts["polisci"]["tier1"][:2]
            st.rerun()
    with c2:
        if st.button("All journals", use_container_width=True):
            st.session_state.journals = avail
            st.rerun()
    
    curr = [j for j in st.session_state.journals if j in avail]
    journals = st.multiselect("Journals", avail, default=curr, label_visibility="collapsed")
    st.session_state.journals = journals
    
    st.markdown("##### Time range")
    days = st.slider("Days", 7, 90, st.session_state.days, 7, format="%d days", label_visibility="collapsed")
    st.session_state.days = days
    
    opts = get_profile_options()
    st.markdown("##### Region focus")
    idx = opts["regions"].index(st.session_state.region) if st.session_state.region in opts["regions"] else 0
    region = st.selectbox("Region", opts["regions"], index=idx, label_visibility="collapsed")
    st.session_state.region = region
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", type="secondary"):
            st.session_state.step = 5
            st.rerun()
    with c2:
        if st.button("Find papers", type="primary", disabled=len(journals) == 0):
            st.session_state.step = 7
            st.rerun()


def step_results():
    if not st.session_state.papers:
        discover()
        return
    
    papers = st.session_state.papers
    
    st.markdown(f'<div class="main-title">For you, {st.session_state.name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-sub">{st.session_state.summary}</div>', unsafe_allow_html=True)
    
    high = sum(1 for p in papers if p["relevance_score"] >= 7.0)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="summary-box"><div class="summary-num">{len(papers)}</div><div class="summary-label">Papers</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="summary-box"><div class="summary-num">{high}</div><div class="summary-label">Highly Relevant</div></div>', unsafe_allow_html=True)
    
    min_score = st.slider("Minimum relevance", 1.0, 10.0, 1.0, 0.5)
    
    filtered = [p for p in papers if p["relevance_score"] >= min_score]
    
    if st.button("Start over", type="secondary"):
        st.session_state.papers = []
        st.session_state.step = 1
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    for i, p in enumerate(filtered):
        score = p["relevance_score"]
        sc_class = "score-high" if score >= 7 else "score-med" if score >= 4 else ""
        
        authors = p.get("authors", [])
        auth = ", ".join(authors[:2])
        if len(authors) > 2:
            auth += f" +{len(authors)-2}"
        
        st.markdown(f'''
        <div class="paper-card">
            <div style="display:flex;justify-content:space-between;align-items:start;">
                <div class="paper-title">{p.get("title", "Untitled")}</div>
                <div class="score {sc_class}">{score}</div>
            </div>
            <div class="paper-meta">{p.get("journal", "")} · {auth} · {fmt_date(p.get("publication_date"))}</div>
            <div class="paper-match">{p.get("match_explanation", "")}</div>
            <div class="paper-tags">
        ''', unsafe_allow_html=True)
        
        tags = []
        for m in p.get("matched_methods", [])[:2]:
            tags.append(f'<span class="tag tag-method">{m}</span>')
        for t in p.get("matched_interests", [])[:2]:
            tags.append(f'<span class="tag tag-interest">{t}</span>')
        if p.get("is_open_access"):
            tags.append('<span class="tag">Open Access</span>')
        
        st.markdown(" ".join(tags) + '</div></div>', unsafe_allow_html=True)
        
        with st.expander("Read abstract"):
            st.write(p.get("abstract", "No abstract available."))
        
        link = p.get("doi_url") or p.get("oa_url")
        if link:
            st.markdown(f"[Open paper]({link})")
        
        st.markdown("---")


def discover():
    profile = create_profile(
        name=st.session_state.name,
        academic_level=st.session_state.level,
        primary_field=st.session_state.field,
        interests=st.session_state.interests,
        methods=st.session_state.methods,
        region=st.session_state.region
    )
    
    with st.spinner(f"Finding papers for you, {st.session_state.name}..."):
        try:
            papers = fetch_recent_papers(
                days_back=st.session_state.days,
                selected_journals=st.session_state.journals,
                max_results=30
            )
            
            if not papers:
                st.error("No papers found. Try expanding your date range or journals.")
                if st.button("Go back"):
                    st.session_state.step = 6
                    st.rerun()
                return
            
            ranked, summary = process_papers(profile, papers)
            
            st.session_state.papers = ranked
            st.session_state.summary = summary
            
            st.rerun()
            
        except OpenAlexError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    step = st.session_state.step
    
    if step == 1:
        step_welcome()
    elif step == 2:
        step_level()
    elif step == 3:
        step_field()
    elif step == 4:
        step_interests()
    elif step == 5:
        step_methods()
    elif step == 6:
        step_sources()
    elif step == 7:
        step_results()


if __name__ == "__main__":
    main()
