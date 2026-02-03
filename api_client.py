"""
OpenAlex API Client for Economics & Political Science Literature Discovery
Handles fetching papers and reconstructing abstracts from inverted index format.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional
import time


# Top Economics Journals with their ISSNs
ECONOMICS_JOURNALS = {
    "American Economic Review": "0002-8282",
    "Quarterly Journal of Economics": "0033-5533",
    "Journal of Political Economy": "0022-3808",
    "Econometrica": "0012-9682",
    "Review of Economic Studies": "0034-6527",
    "Journal of Finance": "0022-1082",
    "Review of Financial Studies": "0893-9454",
    "Journal of Monetary Economics": "0304-3932",
    "Journal of Economic Theory": "0022-0531",
    "AEJ: Applied Economics": "1945-7782",
}

# Top Political Science Journals with their ISSNs
POLISCI_JOURNALS = {
    "American Political Science Review": "0003-0554",
    "American Journal of Political Science": "0092-5853",
    "Journal of Politics": "0022-3816",
    "British Journal of Political Science": "0007-1234",
    "World Politics": "0043-8871",
    "Comparative Political Studies": "0010-4140",
    "International Organization": "0020-8183",
    "Political Analysis": "1047-1987",
    "Annual Review of Political Science": "1094-2939",
    "Political Science Research and Methods": "2049-8470",
}

# Combined for easy access
ALL_JOURNALS = {**ECONOMICS_JOURNALS, **POLISCI_JOURNALS}


def reconstruct_abstract(abstract_inverted_index: Optional[dict]) -> str:
    """
    Reconstruct plaintext abstract from OpenAlex's inverted index format.
    """
    if not abstract_inverted_index:
        return ""
    
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def fetch_recent_papers(
    days_back: int = 30,
    selected_journals: Optional[list] = None,
    per_page: int = 50,
    max_results: int = 200
) -> list:
    """Fetch recent papers from OpenAlex API for specified journals."""
    base_url = "https://api.openalex.org/works"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    if selected_journals:
        issns = [ALL_JOURNALS[j] for j in selected_journals if j in ALL_JOURNALS]
    else:
        issns = list(ALL_JOURNALS.values())
    
    if not issns:
        return []
    
    issn_filter = "|".join(issns)
    papers = []
    cursor = "*"
    
    while len(papers) < max_results:
        params = {
            "filter": f"primary_location.source.issn:{issn_filter},from_publication_date:{from_date},to_publication_date:{to_date}",
            "per_page": min(per_page, max_results - len(papers)),
            "cursor": cursor,
            "select": "id,doi,title,authorships,publication_date,primary_location,abstract_inverted_index,concepts,cited_by_count,open_access",
            "mailto": "econ-discovery@example.com"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                break
                
            for paper in results:
                processed = process_paper(paper)
                if processed:
                    papers.append(processed)
            
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
                
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            break
    
    return papers


def process_paper(paper: dict) -> Optional[dict]:
    """Process a single paper from OpenAlex response into clean format."""
    title = paper.get("title")
    if not title:
        return None
    
    abstract_index = paper.get("abstract_inverted_index")
    abstract = reconstruct_abstract(abstract_index)
    
    if not abstract or len(abstract) < 50:
        return None
    
    authors = []
    for authorship in paper.get("authorships", [])[:5]:
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)
    
    primary_location = paper.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    journal_name = source.get("display_name", "Unknown Journal")
    
    concepts = []
    for concept in paper.get("concepts", [])[:5]:
        if concept.get("score", 0) > 0.3:
            concepts.append({
                "name": concept.get("display_name"),
                "score": concept.get("score")
            })
    
    oa = paper.get("open_access", {}) or {}
    is_open_access = oa.get("is_oa", False)
    oa_url = oa.get("oa_url")
    
    return {
        "id": paper.get("id", "").replace("https://openalex.org/", ""),
        "doi": paper.get("doi"),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "journal": journal_name,
        "publication_date": paper.get("publication_date"),
        "concepts": concepts,
        "cited_by_count": paper.get("cited_by_count", 0),
        "is_open_access": is_open_access,
        "oa_url": oa_url
    }


def get_economics_journals() -> list:
    return list(ECONOMICS_JOURNALS.keys())


def get_polisci_journals() -> list:
    return list(POLISCI_JOURNALS.keys())


def get_all_journals() -> list:
    return list(ALL_JOURNALS.keys())
