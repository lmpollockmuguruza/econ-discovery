"""
Gemini AI Processor for Economics Literature Discovery
Handles batch summarization, relevance scoring, and interest matching.
"""

import json
import re
from typing import Optional
import google.generativeai as genai


# User Interest Matrix Options
ACADEMIC_LEVELS = [
    "Undergraduate",
    "Graduate Student (Masters)",
    "PhD Student",
    "Postdoctoral Researcher",
    "Assistant Professor",
    "Associate/Full Professor",
    "Industry Economist",
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
    "Urban Economics",
    "Economic History",
    "Political Economy"
]

SECONDARY_INTERESTS = [
    "Causal Inference",
    "Machine Learning/AI",
    "Field Experiments (RCTs)",
    "Natural Experiments",
    "Structural Estimation",
    "Theory/Mechanism Design",
    "Policy Evaluation",
    "Inequality & Redistribution",
    "Climate & Energy",
    "Education",
    "Housing & Real Estate",
    "Trade & Globalization",
    "Monetary Policy",
    "Fiscal Policy",
    "Innovation & Technology",
    "Gender & Discrimination",
    "Crime & Law",
    "Health & Healthcare",
    "Immigration",
    "Political Economy"
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
    "Spatial Econometrics"
]


def create_user_profile(
    academic_level: str,
    primary_field: str,
    secondary_interests: list,
    preferred_methodology: list
) -> dict:
    """
    Create a structured user interest profile.
    
    Args:
        academic_level: User's academic/professional level
        primary_field: Main field of economics
        secondary_interests: List of secondary topics of interest
        preferred_methodology: List of preferred research methodologies
        
    Returns:
        Dictionary containing the user profile
    """
    return {
        "academic_level": academic_level,
        "primary_field": primary_field,
        "secondary_interests": secondary_interests,
        "preferred_methodology": preferred_methodology
    }


def build_ranking_prompt(user_profile: dict, papers: list) -> str:
    """
    Build the prompt for Gemini to rank and summarize papers.
    
    Args:
        user_profile: User's interest matrix
        papers: List of papers with abstracts
        
    Returns:
        Formatted prompt string
    """
    # Format user profile
    profile_text = f"""
USER PROFILE:
- Academic Level: {user_profile['academic_level']}
- Primary Field: {user_profile['primary_field']}
- Secondary Interests: {', '.join(user_profile['secondary_interests'])}
- Preferred Methodologies: {', '.join(user_profile['preferred_methodology'])}
"""
    
    # Format papers
    papers_text = "\n\n".join([
        f"PAPER {i+1} (ID: {p['id']}):\nTitle: {p['title']}\nJournal: {p['journal']}\nAbstract: {p['abstract'][:1500]}"
        for i, p in enumerate(papers)
    ])
    
    prompt = f"""You are an AI research assistant for an economist. Analyze the following papers based on the user's research profile.

{profile_text}

PAPERS TO ANALYZE:
{papers_text}

For EACH paper, provide:
1. A relevance score from 1-10 (10 = perfect match to user interests)
2. A 2-sentence summary where:
   - Sentence 1: Explain the core causal mechanism, theoretical contribution, or main finding
   - Sentence 2: Explain specifically why this paper matches (or doesn't match) the user's field, level, and methodological preferences

Return your analysis as a valid JSON array. Each object must have these exact keys:
- "paper_id": the paper ID provided
- "score": integer 1-10
- "contribution": first sentence (the what)
- "relevance_reason": second sentence (the why for this user)
- "key_methodology": the main methodology used (brief, 2-4 words)

Return ONLY the JSON array, no other text. Example format:
[
  {{"paper_id": "W123", "score": 8, "contribution": "This paper shows X causes Y.", "relevance_reason": "Directly relevant to your interest in Z.", "key_methodology": "Diff-in-Diff"}}
]
"""
    return prompt


def process_papers_with_gemini(
    api_key: str,
    user_profile: dict,
    papers: list,
    batch_size: int = 10
) -> list:
    """
    Process papers through Gemini for ranking and summarization.
    
    Args:
        api_key: Gemini API key
        user_profile: User's interest matrix
        papers: List of papers to process
        batch_size: Number of papers per API call
        
    Returns:
        List of papers with AI-generated rankings and summaries
    """
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    all_rankings = {}
    
    # Process in batches
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        prompt = build_ranking_prompt(user_profile, batch)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for consistent scoring
                    max_output_tokens=4096,
                )
            )
            
            # Parse JSON response
            rankings = parse_gemini_response(response.text)
            
            for ranking in rankings:
                paper_id = ranking.get("paper_id")
                if paper_id:
                    all_rankings[paper_id] = ranking
                    
        except Exception as e:
            print(f"Gemini API error for batch {i//batch_size + 1}: {e}")
            # Continue with other batches
            continue
    
    # Merge rankings back into papers
    enriched_papers = []
    for paper in papers:
        ranking = all_rankings.get(paper["id"], {})
        enriched = {
            **paper,
            "relevance_score": ranking.get("score", 5),
            "ai_contribution": ranking.get("contribution", "Summary not available."),
            "ai_relevance": ranking.get("relevance_reason", "Relevance analysis not available."),
            "ai_methodology": ranking.get("key_methodology", "Unknown")
        }
        enriched_papers.append(enriched)
    
    # Sort by relevance score (descending)
    enriched_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return enriched_papers


def parse_gemini_response(response_text: str) -> list:
    """
    Parse Gemini's JSON response, handling potential formatting issues.
    
    Args:
        response_text: Raw response from Gemini
        
    Returns:
        List of ranking dictionaries
    """
    # Clean up the response
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    
    # Try to find JSON array
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        text = json_match.group()
    
    try:
        rankings = json.loads(text)
        if isinstance(rankings, list):
            return rankings
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {text[:500]}...")
    
    return []


def get_profile_options() -> dict:
    """Return all available options for building a user profile."""
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "secondary_interests": SECONDARY_INTERESTS,
        "methodologies": METHODOLOGIES
    }


# Test functionality
if __name__ == "__main__":
    print("Processor module loaded successfully")
    print(f"Academic levels: {len(ACADEMIC_LEVELS)}")
    print(f"Primary fields: {len(PRIMARY_FIELDS)}")
    print(f"Secondary interests: {len(SECONDARY_INTERESTS)}")
    print(f"Methodologies: {len(METHODOLOGIES)}")
    
    # Test profile creation
    test_profile = create_user_profile(
        academic_level="PhD Student",
        primary_field="Labor Economics",
        secondary_interests=["Causal Inference", "Education"],
        preferred_methodology=["Difference-in-Differences", "Regression Discontinuity"]
    )
    print(f"\nTest profile: {json.dumps(test_profile, indent=2)}")
