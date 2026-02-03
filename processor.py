"""
Gemini AI Processor for Literature Discovery
Handles batch summarization, relevance scoring, and interest matching.
"""

import json
import re
from typing import Optional

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# User Interest Matrix Options
ACADEMIC_LEVELS = [
    "Undergraduate",
    "Graduate Student",
    "PhD Student",
    "Postdoc",
    "Assistant Professor",
    "Associate/Full Professor",
    "Industry Researcher",
    "Policy Researcher"
]

PRIMARY_FIELDS = [
    "Microeconomics",
    "Macroeconomics",
    "Econometrics",
    "Labor Economics",
    "Public Economics",
    "International Economics",
    "Development Economics",
    "Financial Economics",
    "Industrial Organization",
    "Behavioral Economics",
    "Health Economics",
    "Environmental Economics",
    "Political Economy",
    "Comparative Politics",
    "International Relations",
    "American Politics",
    "Political Theory",
    "Public Policy"
]

SECONDARY_INTERESTS = [
    "Causal Inference",
    "Machine Learning/AI",
    "Field Experiments (RCTs)",
    "Natural Experiments",
    "Structural Estimation",
    "Theory/Mechanism Design",
    "Policy Evaluation",
    "Inequality",
    "Climate & Energy",
    "Education",
    "Housing",
    "Trade & Globalization",
    "Monetary Policy",
    "Fiscal Policy",
    "Innovation & Technology",
    "Gender & Discrimination",
    "Crime & Law",
    "Health & Healthcare",
    "Immigration",
    "Democratic Institutions",
    "Voting Behavior",
    "Conflict & Security"
]

METHODOLOGIES = [
    "Difference-in-Differences",
    "Regression Discontinuity",
    "Instrumental Variables",
    "Randomized Controlled Trials",
    "Structural Models",
    "Theoretical Models",
    "Machine Learning Methods",
    "Survey/Experimental Data",
    "Administrative Data",
    "Time Series Analysis",
    "Panel Data Methods",
    "Text Analysis/NLP",
    "Network Analysis"
]


def create_user_profile(
    academic_level: str,
    primary_field: str,
    secondary_interests: list,
    preferred_methodology: list
) -> dict:
    """Create a structured user interest profile."""
    return {
        "academic_level": academic_level,
        "primary_field": primary_field,
        "secondary_interests": secondary_interests,
        "preferred_methodology": preferred_methodology
    }


def build_ranking_prompt(user_profile: dict, papers: list) -> str:
    """Build the prompt for Gemini to rank and summarize papers."""
    profile_text = f"""USER PROFILE:
- Academic Level: {user_profile['academic_level']}
- Primary Field: {user_profile['primary_field']}
- Secondary Interests: {', '.join(user_profile['secondary_interests'])}
- Preferred Methodologies: {', '.join(user_profile['preferred_methodology'])}"""
    
    papers_text = "\n\n".join([
        f"PAPER {i+1} (ID: {p['id']}):\nTitle: {p['title']}\nJournal: {p['journal']}\nAbstract: {p['abstract'][:1200]}"
        for i, p in enumerate(papers)
    ])
    
    prompt = f"""You are an AI research assistant. Analyze these academic papers based on the user's research profile.

{profile_text}

PAPERS TO ANALYZE:
{papers_text}

For EACH paper, provide:
1. A relevance score from 1-10 (10 = perfect match to user interests)
2. A 2-sentence summary where:
   - Sentence 1: The core finding or contribution (what they discovered/argued)
   - Sentence 2: Why this is relevant to THIS user's specific interests

Return ONLY a valid JSON array. Each object must have exactly these keys:
- "paper_id": the paper ID provided (e.g., "W1234567")
- "score": integer 1-10
- "contribution": first sentence
- "relevance_reason": second sentence  
- "key_methodology": main method used (2-4 words)

Example format:
[{{"paper_id": "W123", "score": 8, "contribution": "This paper finds that X causes Y using data from Z.", "relevance_reason": "Directly relevant to your interest in causal inference and labor economics.", "key_methodology": "Diff-in-Diff"}}]

Return ONLY the JSON array, no markdown, no explanation."""

    return prompt


def process_papers_with_gemini(
    api_key: str,
    user_profile: dict,
    papers: list,
    batch_size: int = 8
) -> list:
    """Process papers through Gemini for ranking and summarization."""
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    all_rankings = {}
    errors = []
    
    # Process in batches
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        prompt = build_ranking_prompt(user_profile, batch)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                )
            )
            
            rankings = parse_gemini_response(response.text)
            
            for ranking in rankings:
                paper_id = ranking.get("paper_id")
                if paper_id:
                    all_rankings[paper_id] = ranking
                    
        except Exception as e:
            errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
            continue
    
    # Merge rankings back into papers
    enriched_papers = []
    for paper in papers:
        ranking = all_rankings.get(paper["id"], {})
        enriched = {
            **paper,
            "relevance_score": ranking.get("score", 5),
            "ai_contribution": ranking.get("contribution", ""),
            "ai_relevance": ranking.get("relevance_reason", ""),
            "ai_methodology": ranking.get("key_methodology", "")
        }
        enriched_papers.append(enriched)
    
    # Sort by relevance score
    enriched_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return enriched_papers


def parse_gemini_response(response_text: str) -> list:
    """Parse Gemini's JSON response, handling potential formatting issues."""
    text = response_text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    
    # Find JSON array
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        text = json_match.group()
    
    try:
        rankings = json.loads(text)
        if isinstance(rankings, list):
            return rankings
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    
    return []


def get_profile_options() -> dict:
    """Return all available options for building a user profile."""
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "secondary_interests": SECONDARY_INTERESTS,
        "methodologies": METHODOLOGIES
    }
