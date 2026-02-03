"""
Literature Discovery — Research Paper Matching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A refined interface for discovering relevant academic papers
using semantic keyword matching.
"""

import streamlit as st
from datetime import datetime
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
    process_papers,
    create_user_profile,
    get_profile_options,
    get_suggested_authors
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Literature Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .hero-box {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    
    .hero-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .hero-sub {
        font-size: 1.05rem;
        color: #666;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .step-bar {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    .step-pill {
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .step-done { background: #198754; color: white; }
    .step-now { background: #1a1a2e; color: white; }
    .step-wait { background: #e9ecef; color: #6c757d; }
    
    .score-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .score-high { background: #d1e7dd; color: #0f5132; }
    .score-med { background: #fff3cd; color: #664d03; }
    .score-low { background: #e9ecef; color: #6c757d; }
    
    .tag-box { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.5rem; }
    
    .tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    .tag-method { background: #cfe2ff; color: #084298; }
    .tag-topic { background: #e2d9f3; color: #432874; }
    .tag-oa { background: #d1e7dd; color: #0f5132; }
    .tag-tier { background: #fff3cd; color: #664d03; }
    
    .match-hint {
        background: #f8f9fa;
        border-left: 3px solid #0d6efd;
        padding: 0.6rem 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #495057;
    }
    
    .metric-grid {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-num {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .metric-txt {
        font-size: 0.7rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .success-msg {
        background: #d1e7dd;
        color: #0f5132;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_state():
    defaults = {
        "step": 1,
        "papers": [],
        "messages": [],
        "profile": None,
        "academic_level": "PhD Student (ABD)",
        "primary_field": "Labor Economics",
        "interests": ["Causal Inference"],
        "methods": ["Difference-in-Differences"],
        "region": "United States",
        "method_lean": 0.7,
        "seed_authors": [],
        "journals": [],
        "days_back": 30,
        "max_papers": 25,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CACHE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_papers_cached(journals_tuple, days, max_p):
    return fetch_recent_papers(days_back=days, selected_journals=list(journals_tuple), max_results=max_p)

@st.cache_resource
def get_options():
    return get_profile_options()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt_date(d):
    if not d:
        return "Unknown"
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y")
    except:
        return d


def show_steps(current):
    steps = ["Profile", "Interests", "Sources", "Results"]
    html = '<div class="step-bar">'
    for i, s in enumerate(steps, 1):
        if i < current:
            cls = "step-done"
            txt = f"✓ {s}"
        elif i == current:
            cls = "step-now"
            txt = f"{i}. {s}"
        else:
            cls = "step-wait"
            txt = f"{i}. {s}"
        html += f'<span class="step-pill {cls}">{txt}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def show_paper(p, rank):
    score = p.get("relevance_score", 5)
    if score >= 7:
        sc, si = "score-high", "●"
    elif score >= 4:
        sc, si = "score-med", "◐"
    else:
        sc, si = "score-low", "○"
    
    authors = p.get("authors", [])
    auth_str = ", ".join(authors[:3])
    if len(authors) > 3:
        auth_str += f" +{len(authors)-3}"
    
    tags = '<div class="tag-box">'
    for m in p.get("method_matches", [])[:2]:
        tags += f'<span class="tag tag-method">⚙ {m}</span>'
    for t in p.get("topic_matches", [])[:2]:
        tags += f'<span class="tag tag-topic">◆ {t}</span>'
    if p.get("is_open_access"):
        tags += '<span class="tag tag-oa">🔓 Open</span>'
    tier = p.get("journal_tier", 4)
    if tier == 1:
        tags += '<span class="tag tag-tier">★ Top 5</span>'
    elif tier == 2:
        tags += '<span class="tag tag-tier">★ Field</span>'
    if p.get("author_match"):
        tags += '<span class="tag tag-topic">👤 Author</span>'
    tags += '</div>'
    
    hints = []
    if p.get("field_score", 0) > 0.3:
        hints.append("Field match")
    if p.get("topic_matches"):
        hints.append(f"Topics: {', '.join(p['topic_matches'][:2])}")
    if p.get("method_matches"):
        hints.append(f"Methods: {', '.join(p['method_matches'][:2])}")
    hint_txt = " • ".join(hints) if hints else "General relevance"
    
    with st.container():
        c1, c2 = st.columns([6, 1])
        with c1:
            st.markdown(f"**{rank}. {p.get('title', 'Untitled')}**")
        with c2:
            st.markdown(f'<span class="score-pill {sc}">{si} {score:.1f}</span>', unsafe_allow_html=True)
        
        st.caption(f"📖 {p.get('journal', 'Unknown')} · 👤 {auth_str} · 📅 {fmt_date(p.get('publication_date'))} · 📊 {p.get('cited_by_count', 0)} cites")
        
        abstract = p.get("abstract", "")
        if abstract:
            with st.expander("Abstract"):
                st.write(abstract)
        
        st.markdown(f'<div class="match-hint">💡 {hint_txt}</div>', unsafe_allow_html=True)
        st.markdown(tags, unsafe_allow_html=True)
        
        link = p.get("doi_url") or p.get("oa_url")
        if link:
            st.markdown(f"[Read paper →]({link})")
        
        st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step1():
    opts = get_options()
    
    st.markdown('<div class="hero-box"><div class="hero-title">Tell us about yourself</div><div class="hero-sub">We\'ll match papers to your research profile</div></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("#### 👤 Career Stage")
        st.session_state.academic_level = st.selectbox("Level", opts["academic_levels"], index=opts["academic_levels"].index(st.session_state.academic_level) if st.session_state.academic_level in opts["academic_levels"] else 3, label_visibility="collapsed")
        
        st.markdown("#### 📚 Primary Field")
        st.session_state.primary_field = st.selectbox("Field", opts["primary_fields"], index=opts["primary_fields"].index(st.session_state.primary_field) if st.session_state.primary_field in opts["primary_fields"] else 3, label_visibility="collapsed")
        
        st.markdown("#### 🌍 Regional Focus")
        st.session_state.region = st.selectbox("Region", opts["regional_focus"], index=opts["regional_focus"].index(st.session_state.region) if st.session_state.region in opts["regional_focus"] else 0, label_visibility="collapsed")
        
        st.markdown("")
        _, bc = st.columns([1, 1])
        with bc:
            if st.button("Continue →", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.rerun()


def step2():
    opts = get_options()
    
    st.markdown('<div class="hero-box"><div class="hero-title">Research Interests</div><div class="hero-sub">Select topics and methods that matter to you</div></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("#### 🎯 Topics (up to 5)")
        st.session_state.interests = st.multiselect("Topics", opts["secondary_interests"], default=st.session_state.interests, max_selections=5, label_visibility="collapsed")
        
        st.markdown("#### ⚙️ Methods (up to 4)")
        st.session_state.methods = st.multiselect("Methods", opts["methodologies"], default=st.session_state.methods, max_selections=4, label_visibility="collapsed")
        
        st.markdown("#### 📊 Orientation")
        st.session_state.method_lean = st.slider("Lean", 0.0, 1.0, st.session_state.method_lean, format="", label_visibility="collapsed")
        lc, mc, rc = st.columns(3)
        with lc:
            st.caption("← Qualitative")
        with mc:
            st.caption("Mixed")
        with rc:
            st.caption("Quantitative →")
        
        st.markdown("#### 👥 Seed Authors (optional)")
        sugg = get_suggested_authors(st.session_state.primary_field)
        if sugg:
            st.caption(f"Try: {', '.join(sugg[:3])}")
        inp = st.text_input("Authors", ", ".join(st.session_state.seed_authors), placeholder="Raj Chetty, Esther Duflo", label_visibility="collapsed")
        st.session_state.seed_authors = [a.strip() for a in inp.split(",") if a.strip()]
        
        st.markdown("")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with bc2:
            if st.button("Continue →", type="primary", use_container_width=True):
                if not st.session_state.interests:
                    st.error("Select at least one topic")
                elif not st.session_state.methods:
                    st.error("Select at least one method")
                else:
                    st.session_state.step = 3
                    st.rerun()


def step3():
    jopts = get_journal_options()
    
    st.markdown('<div class="hero-box"><div class="hero-title">Select Sources</div><div class="hero-sub">Choose journals and time range</div></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("#### 📖 Field")
        field = st.radio("F", ["Economics", "Political Science", "Both"], horizontal=True, label_visibility="collapsed")
        
        if field == "Economics":
            avail = get_economics_journals()
            top = jopts["economics"]["top5"]
        elif field == "Political Science":
            avail = get_polisci_journals()
            top = jopts["polisci"]["top3"]
        else:
            avail = get_all_journals()
            top = jopts["economics"]["top5"][:3] + jopts["polisci"]["top3"][:2]
        
        st.markdown("#### 📚 Journals")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Top Journals", use_container_width=True):
                st.session_state.journals = top
                st.rerun()
        with b2:
            if st.button("All Journals", use_container_width=True):
                st.session_state.journals = avail
                st.rerun()
        
        curr = [j for j in st.session_state.journals if j in avail] or top[:3]
        st.session_state.journals = st.multiselect("J", avail, default=curr, label_visibility="collapsed")
        
        st.markdown("#### ⏱️ Days Back")
        st.session_state.days_back = st.slider("D", 7, 90, st.session_state.days_back, 7, format="%d days")
        
        st.markdown("#### 📊 Max Papers")
        st.session_state.max_papers = st.slider("M", 10, 50, st.session_state.max_papers, 5)
        
        st.markdown("")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("← Back", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
        with bc2:
            if st.button("🔍 Find Papers", type="primary", use_container_width=True):
                if not st.session_state.journals:
                    st.error("Select at least one journal")
                else:
                    st.session_state.step = 4
                    st.rerun()


def step4():
    if not st.session_state.papers:
        discover()
    results()


def discover():
    profile = create_user_profile(
        academic_level=st.session_state.academic_level,
        primary_field=st.session_state.primary_field,
        secondary_interests=st.session_state.interests,
        preferred_methodology=st.session_state.methods,
        regional_focus=st.session_state.region,
        seed_authors=st.session_state.seed_authors,
        methodological_lean=st.session_state.method_lean
    )
    st.session_state.profile = profile
    
    with st.status("🔬 Finding papers...", expanded=True) as status:
        st.write("📡 Fetching from OpenAlex...")
        
        try:
            papers = fetch_papers_cached(tuple(st.session_state.journals), st.session_state.days_back, st.session_state.max_papers)
            
            if not papers:
                status.update(label="No papers found", state="error")
                st.error("No papers found. Try expanding date range or journals.")
                if st.button("← Adjust"):
                    st.session_state.step = 3
                    st.rerun()
                return
            
            st.write(f"✓ Found {len(papers)} papers")
            st.write("🔬 Computing matches...")
            
            processed, msgs = process_papers(profile, papers, lambda m: st.write(m))
            
            st.session_state.papers = processed
            st.session_state.messages = msgs
            
            status.update(label="✓ Done!", state="complete")
            time.sleep(0.3)
            st.rerun()
            
        except OpenAlexError as e:
            status.update(label="Error", state="error")
            st.error(f"API Error: {e}")
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Error: {e}")


def results():
    papers = st.session_state.papers
    msgs = st.session_state.messages
    
    if not papers:
        st.warning("No papers.")
        if st.button("← Back"):
            st.session_state.papers = []
            st.session_state.step = 3
            st.rerun()
        return
    
    st.markdown('<div class="hero-box"><div class="hero-title">Your Reading List</div><div class="hero-sub">Papers ranked by relevance to your profile</div></div>', unsafe_allow_html=True)
    
    high = sum(1 for p in papers if p.get("relevance_score", 0) >= 7)
    med = sum(1 for p in papers if 4 <= p.get("relevance_score", 0) < 7)
    
    st.markdown(f'''
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-num">{len(papers)}</div><div class="metric-txt">Papers</div></div>
        <div class="metric-card"><div class="metric-num">{high}</div><div class="metric-txt">High Match</div></div>
        <div class="metric-card"><div class="metric-num">{med}</div><div class="metric-txt">Medium</div></div>
    </div>
    ''', unsafe_allow_html=True)
    
    for m in msgs:
        st.markdown(f'<div class="success-msg">{m}</div>', unsafe_allow_html=True)
    
    # Filters
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    with f1:
        min_score = st.slider("Min score", 1.0, 10.0, 1.0, 0.5)
    with f2:
        sort_by = st.selectbox("Sort", ["Relevance", "Date", "Citations"])
    with f3:
        oa_only = st.checkbox("Open Access only")
    with f4:
        if st.button("🔄 New"):
            st.session_state.papers = []
            st.session_state.step = 1
            st.rerun()
    
    filtered = [p for p in papers if p.get("relevance_score", 0) >= min_score]
    if oa_only:
        filtered = [p for p in filtered if p.get("is_open_access")]
    
    if sort_by == "Date":
        filtered.sort(key=lambda x: x.get("publication_date", ""), reverse=True)
    elif sort_by == "Citations":
        filtered.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    else:
        filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    st.divider()
    
    if not filtered:
        st.info("No papers match filters.")
    else:
        for i, p in enumerate(filtered, 1):
            show_paper(p, i)
    
    st.caption("Data from [OpenAlex](https://openalex.org)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    show_steps(st.session_state.step)
    
    if st.session_state.step == 1:
        step1()
    elif st.session_state.step == 2:
        step2()
    elif st.session_state.step == 3:
        step3()
    elif st.session_state.step == 4:
        step4()


if __name__ == "__main__":
    main()
