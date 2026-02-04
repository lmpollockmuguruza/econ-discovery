"""
Econvery — OpenAlex API Client
Fetches academic papers from top economics and political science journals.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from functools import lru_cache
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Journal:
    name: str
    issn: str
    field: str
    tier: int  # 1=Top, 2=Excellent, 3=Very Good


# Economics Journals
ECONOMICS_JOURNALS: Dict[str, Journal] = {
    # Tier 1 - Top 5
    "American Economic Review": Journal("American Economic Review", "0002-8282", "economics", 1),
    "Quarterly Journal of Economics": Journal("Quarterly Journal of Economics", "0033-5533", "economics", 1),
    "Journal of Political Economy": Journal("Journal of Political Economy", "0022-3808", "economics", 1),
    "Econometrica": Journal("Econometrica", "0012-9682", "economics", 1),
    "Review of Economic Studies": Journal("Review of Economic Studies", "0034-6527", "economics", 1),
    # Tier 2 - Top Field
    "Journal of Finance": Journal("Journal of Finance", "0022-1082", "economics", 2),
    "Review of Financial Studies": Journal("Review of Financial Studies", "0893-9454", "economics", 2),
    "Journal of Financial Economics": Journal("Journal of Financial Economics", "0304-405X", "economics", 2),
    "Journal of Monetary Economics": Journal("Journal of Monetary Economics", "0304-3932", "economics", 2),
    "Journal of Economic Theory": Journal("Journal of Economic Theory", "0022-0531", "economics", 2),
    "AEJ: Applied Economics": Journal("AEJ: Applied Economics", "1945-7782", "economics", 2),
    "AEJ: Economic Policy": Journal("AEJ: Economic Policy", "1945-7731", "economics", 2),
    "AEJ: Macroeconomics": Journal("AEJ: Macroeconomics", "1945-7707", "economics", 2),
    "AEJ: Microeconomics": Journal("AEJ: Microeconomics", "1945-7669", "economics", 2),
    "Journal of Labor Economics": Journal("Journal of Labor Economics", "0734-306X", "economics", 2),
    "Journal of Public Economics": Journal("Journal of Public Economics", "0047-2727", "economics", 2),
    "Journal of Human Resources": Journal("Journal of Human Resources", "0022-166X", "economics", 2),
    "Journal of Economic Perspectives": Journal("Journal of Economic Perspectives", "0895-3309", "economics", 2),
    # Tier 3 - Excellent
    "Review of Economics and Statistics": Journal("Review of Economics and Statistics", "0034-6535", "economics", 3),
    "Journal of the European Economic Association": Journal("Journal of the European Economic Association", "1542-4766", "economics", 3),
    "Economic Journal": Journal("Economic Journal", "0013-0133", "economics", 3),
    "Journal of Development Economics": Journal("Journal of Development Economics", "0304-3878", "economics", 3),
    "Journal of International Economics": Journal("Journal of International Economics", "0022-1996", "economics", 3),
    "Journal of Economic Growth": Journal("Journal of Economic Growth", "1381-4338", "economics", 3),
    "American Economic Journal: Applied Economics": Journal("American Economic Journal: Applied Economics", "1945-7782", "economics", 3),
    "Journal of Applied Econometrics": Journal("Journal of Applied Econometrics", "0883-7252", "economics", 3),
    "Journal of Business & Economic Statistics": Journal("Journal of Business & Economic Statistics", "0735-0015", "economics", 3),
    "Economic Policy": Journal("Economic Policy", "0266-4658", "economics", 3),
    "Journal of Economic Literature": Journal("Journal of Economic Literature", "0022-0515", "economics", 3),
}

# Political Science Journals
POLISCI_JOURNALS: Dict[str, Journal] = {
    # Tier 1 - Top 3
    "American Political Science Review": Journal("American Political Science Review", "0003-0554", "polisci", 1),
    "American Journal of Political Science": Journal("American Journal of Political Science", "0092-5853", "polisci", 1),
    "Journal of Politics": Journal("Journal of Politics", "0022-3816", "polisci", 1),
    # Tier 2 - Top Field
    "Quarterly Journal of Political Science": Journal("Quarterly Journal of Political Science", "1554-0626", "polisci", 2),
    "British Journal of Political Science": Journal("British Journal of Political Science", "0007-1234", "polisci", 2),
    "World Politics": Journal("World Politics", "0043-8871", "polisci", 2),
    "Comparative Political Studies": Journal("Comparative Political Studies", "0010-4140", "polisci", 2),
    "International Organization": Journal("International Organization", "0020-8183", "polisci", 2),
    "Political Analysis": Journal("Political Analysis", "1047-1987", "polisci", 2),
    "Annual Review of Political Science": Journal("Annual Review of Political Science", "1094-2939", "polisci", 2),
    "Political Science Research and Methods": Journal("Political Science Research and Methods", "2049-8470", "polisci", 2),
    "Journal of Conflict Resolution": Journal("Journal of Conflict Resolution", "0022-0027", "polisci", 2),
    "International Security": Journal("International Security", "0162-2889", "polisci", 2),
    # Tier 3 - Excellent
    "International Studies Quarterly": Journal("International Studies Quarterly", "0020-8833", "polisci", 3),
    "Comparative Politics": Journal("Comparative Politics", "0010-4159", "polisci", 3),
    "Political Behavior": Journal("Political Behavior", "0190-9320", "polisci", 3),
    "Public Opinion Quarterly": Journal("Public Opinion Quarterly", "0033-362X", "polisci", 3),
    "Legislative Studies Quarterly": Journal("Legislative Studies Quarterly", "0362-9805", "polisci", 3),
    "European Journal of Political Research": Journal("European Journal of Political Research", "0304-4130", "polisci", 3),
    "Journal of Peace Research": Journal("Journal of Peace Research", "0022-3433", "polisci", 3),
    "Political Science Quarterly": Journal("Political Science Quarterly", "0032-3195", "polisci", 3),
    "Perspectives on Politics": Journal("Perspectives on Politics", "1537-5927", "polisci", 3),
}

ALL_JOURNALS: Dict[str, Journal] = {**ECONOMICS_JOURNALS, **POLISCI_JOURNALS}


class OpenAlexConfig:
    BASE_URL = "https://api.openalex.org/works"
    TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    RATE_LIMIT_DELAY = 0.1
    POLITE_EMAIL = "econvery@research.app"
    SELECT_FIELDS = ",".join([
        "id", "doi", "title", "authorships", "publication_date",
        "primary_location", "abstract_inverted_index", "concepts",
        "cited_by_count", "open_access", "type"
    ])


class OpenAlexError(Exception):
    pass


def reconstruct_abstract(abstract_inverted_index: Optional[dict]) -> str:
    if not abstract_inverted_index:
        return ""
    try:
        word_positions: List[Tuple[int, str]] = []
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except Exception as e:
        logger.warning(f"Failed to reconstruct abstract: {e}")
        return ""


def _make_request_with_retry(url: str, params: dict, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=OpenAlexConfig.TIMEOUT)
            if response.status_code == 429:
                time.sleep(OpenAlexConfig.RETRY_DELAY * (2 ** attempt))
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP {e.response.status_code}"
            if e.response.status_code >= 500:
                time.sleep(OpenAlexConfig.RETRY_DELAY * (2 ** attempt))
                continue
            raise OpenAlexError(last_error)
        except requests.exceptions.RequestException as e:
            last_error = str(e)
        if attempt < max_retries - 1:
            time.sleep(OpenAlexConfig.RETRY_DELAY * (2 ** attempt))
    raise OpenAlexError(f"Failed after {max_retries} attempts: {last_error}")


def process_paper(paper: dict) -> Optional[dict]:
    title = paper.get("title")
    if not title:
        return None
    
    abstract_index = paper.get("abstract_inverted_index")
    abstract = reconstruct_abstract(abstract_index)
    
    if not abstract or len(abstract) < 80:
        return None
    
    authors = []
    for authorship in paper.get("authorships", [])[:10]:
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)
    
    institutions = []
    for authorship in paper.get("authorships", [])[:5]:
        for inst in authorship.get("institutions", [])[:1]:
            inst_name = inst.get("display_name")
            if inst_name and inst_name not in institutions:
                institutions.append(inst_name)
    
    primary_location = paper.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    journal_name = source.get("display_name", "Unknown Journal")
    
    concepts = []
    for concept in paper.get("concepts", [])[:8]:
        score = concept.get("score", 0)
        if score > 0.2:
            concepts.append({
                "name": concept.get("display_name"),
                "score": round(score, 2)
            })
    
    oa = paper.get("open_access", {}) or {}
    is_open_access = oa.get("is_oa", False)
    oa_url = oa.get("oa_url")
    
    doi = paper.get("doi")
    doi_url = f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else None
    
    journal = ALL_JOURNALS.get(journal_name)
    journal_tier = journal.tier if journal else 4
    
    return {
        "id": paper.get("id", "").replace("https://openalex.org/", ""),
        "doi": doi,
        "doi_url": doi_url,
        "title": title,
        "authors": authors,
        "institutions": institutions[:3],
        "abstract": abstract,
        "journal": journal_name,
        "journal_tier": journal_tier,
        "publication_date": paper.get("publication_date"),
        "concepts": concepts,
        "cited_by_count": paper.get("cited_by_count", 0),
        "is_open_access": is_open_access,
        "oa_url": oa_url,
    }


def fetch_recent_papers(
    days_back: int = 30,
    selected_journals: Optional[List[str]] = None,
    per_page: int = 50,
    max_results: int = 100,
) -> List[dict]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    if selected_journals:
        issns = [ALL_JOURNALS[j].issn for j in selected_journals if j in ALL_JOURNALS]
    else:
        issns = [j.issn for j in ALL_JOURNALS.values()]
    
    if not issns:
        return []
    
    issn_filter = "|".join(issns)
    papers: List[dict] = []
    cursor = "*"
    
    while len(papers) < max_results:
        params = {
            "filter": f"primary_location.source.issn:{issn_filter},from_publication_date:{from_date},to_publication_date:{to_date},type:article",
            "per_page": min(per_page, max_results - len(papers)),
            "cursor": cursor,
            "select": OpenAlexConfig.SELECT_FIELDS,
            "mailto": OpenAlexConfig.POLITE_EMAIL
        }
        
        try:
            data = _make_request_with_retry(OpenAlexConfig.BASE_URL, params)
            results = data.get("results", [])
            if not results:
                break
            
            for paper in results:
                processed = process_paper(paper)
                if processed:
                    papers.append(processed)
                    if len(papers) >= max_results:
                        break
            
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            
            time.sleep(OpenAlexConfig.RATE_LIMIT_DELAY)
        except OpenAlexError:
            raise
        except Exception as e:
            raise OpenAlexError(f"Failed to fetch papers: {str(e)}")
    
    return papers


def get_economics_journals() -> List[str]:
    return list(ECONOMICS_JOURNALS.keys())


def get_polisci_journals() -> List[str]:
    return list(POLISCI_JOURNALS.keys())


def get_all_journals() -> List[str]:
    return list(ALL_JOURNALS.keys())


@lru_cache(maxsize=1)
def get_journal_options() -> dict:
    return {
        "economics": {
            "tier1": [n for n, j in ECONOMICS_JOURNALS.items() if j.tier == 1],
            "tier2": [n for n, j in ECONOMICS_JOURNALS.items() if j.tier == 2],
            "tier3": [n for n, j in ECONOMICS_JOURNALS.items() if j.tier == 3],
        },
        "polisci": {
            "tier1": [n for n, j in POLISCI_JOURNALS.items() if j.tier == 1],
            "tier2": [n for n, j in POLISCI_JOURNALS.items() if j.tier == 2],
            "tier3": [n for n, j in POLISCI_JOURNALS.items() if j.tier == 3],
        }
    }
