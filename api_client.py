"""
OpenAlex API Client for Economics Literature Discovery
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
    "American Economic Journal: Applied Economics": "1945-7782",
}


def reconstruct_abstract(abstract_inverted_index: Optional[dict]) -> str:
    """
    Reconstruct plaintext abstract from OpenAlex's inverted index format.
    
    OpenAlex stores abstracts as {word: [position1, position2, ...]} dictionaries.
    This function reconstructs the original text by placing words at their positions.
    
    Args:
        abstract_inverted_index: Dictionary mapping words to their positions
        
    Returns:
        Reconstructed plaintext abstract
    """
    if not abstract_inverted_index:
        return ""
    
    # Create a list to hold words at their positions
    word_positions = []
    
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    # Sort by position and join words
    word_positions.sort(key=lambda x: x[0])
    abstract = " ".join(word for _, word in word_positions)
    
    return abstract


def fetch_recent_papers(
    days_back: int = 30,
    selected_journals: Optional[list] = None,
    per_page: int = 50,
    max_results: int = 200
) -> list:
    """
    Fetch recent papers from OpenAlex API for specified economics journals.
    
    Args:
        days_back: Number of days to look back for publications
        selected_journals: List of journal names to filter (uses all if None)
        per_page: Number of results per API call
        max_results: Maximum total results to fetch
        
    Returns:
        List of paper dictionaries with reconstructed abstracts
    """
    base_url = "https://api.openalex.org/works"
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    # Build ISSN filter
    if selected_journals:
        issns = [ECONOMICS_JOURNALS[j] for j in selected_journals if j in ECONOMICS_JOURNALS]
    else:
        issns = list(ECONOMICS_JOURNALS.values())
    
    issn_filter = "|".join(issns)
    
    papers = []
    cursor = "*"
    
    while len(papers) < max_results:
        params = {
            "filter": f"primary_location.source.issn:{issn_filter},from_publication_date:{from_date},to_publication_date:{to_date}",
            "per_page": min(per_page, max_results - len(papers)),
            "cursor": cursor,
            "select": "id,doi,title,authorships,publication_date,primary_location,abstract_inverted_index,concepts,cited_by_count,open_access",
            "mailto": "econ-discovery@example.com"  # Polite pool access
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
            
            # Get next cursor for pagination
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
                
            # Rate limiting - be polite to the API
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            break
    
    return papers


def process_paper(paper: dict) -> Optional[dict]:
    """
    Process a single paper from OpenAlex response into clean format.
    
    Args:
        paper: Raw paper dictionary from OpenAlex
        
    Returns:
        Processed paper dictionary or None if missing critical data
    """
    title = paper.get("title")
    if not title:
        return None
    
    # Reconstruct abstract
    abstract_index = paper.get("abstract_inverted_index")
    abstract = reconstruct_abstract(abstract_index)
    
    # Skip papers without abstracts (can't rank them meaningfully)
    if not abstract or len(abstract) < 50:
        return None
    
    # Extract authors
    authors = []
    for authorship in paper.get("authorships", [])[:5]:  # Limit to first 5 authors
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)
    
    # Extract journal information
    primary_location = paper.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    journal_name = source.get("display_name", "Unknown Journal")
    
    # Extract concepts/topics
    concepts = []
    for concept in paper.get("concepts", [])[:5]:  # Top 5 concepts
        if concept.get("score", 0) > 0.3:  # Only high-confidence concepts
            concepts.append({
                "name": concept.get("display_name"),
                "score": concept.get("score")
            })
    
    # Open access status
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


def get_journal_names() -> list:
    """Return list of available journal names."""
    return list(ECONOMICS_JOURNALS.keys())


# Quick test
if __name__ == "__main__":
    print("Testing OpenAlex API Client...")
    print(f"Available journals: {get_journal_names()}")
    
    # Test abstract reconstruction
    test_index = {
        "This": [0],
        "is": [1],
        "a": [2],
        "test": [3],
        "abstract": [4],
        ".": [5]
    }
    reconstructed = reconstruct_abstract(test_index)
    print(f"Reconstructed abstract: {reconstructed}")
    
    # Test API fetch (small sample)
    print("\nFetching recent papers (last 7 days)...")
    papers = fetch_recent_papers(days_back=7, max_results=5)
    print(f"Found {len(papers)} papers")
    
    for paper in papers[:2]:
        print(f"\n--- {paper['title'][:60]}...")
        print(f"    Journal: {paper['journal']}")
        print(f"    Authors: {', '.join(paper['authors'][:3])}")
        print(f"    Abstract: {paper['abstract'][:150]}...")
