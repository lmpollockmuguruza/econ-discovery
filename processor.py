"""
Gemini AI Processor for Literature Discovery
Handles batch summarization, relevance scoring, and interest matching.
"""

import json
import re
from typing import Optional, Tuple

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


def build_ranking_prompt(user_profile: dict, papers: list, start_index: int) -> str:
    """Build the prompt for Gemini to rank and summarize papers."""
    
    profile_text = f"""USER PROFILE:
- Academic Level: {user_profile['academic_level']}
- Primary Field: {user_profile['primary_field']}  
- Secondary Interests: {', '.join(user_profile['secondary_interests'])}
- Preferred Methodologies: {', '.join(user_profile['preferred_methodology'])}"""
    
    # Use simple numeric indices for reliable matching
    papers_text = ""
    for i, p in enumerate(papers):
        idx = start_index + i + 1  # 1-based index
        abstract = p.get('abstract', '')[:1000]  # Shorter to fit more
        papers_text += f"""
---
PAPER {idx}:
Title: {p.get('title', 'Unknown')}
Journal: {p.get('journal', 'Unknown')}
Abstract: {abstract}
"""
    
    prompt = f"""You are an expert research assistant helping an academic researcher find relevant papers.

{profile_text}

Analyze each paper and score its relevance to this specific researcher's interests.

{papers_text}
---

For each paper, return a JSON object with:
- "paper_num": the paper number (1, 2, 3, etc.)
- "score": relevance score from 1-10 where:
  - 9-10: Directly in their primary field AND uses their preferred methods
  - 7-8: Highly relevant to their interests
  - 5-6: Somewhat relevant, tangentially related
  - 3-4: Minimal relevance
  - 1-2: Not relevant to their profile
- "summary": One sentence explaining the paper's main contribution/finding
- "why_relevant": One sentence explaining specifically why this paper matches (or doesn't match) THIS user's interests, mentioning their specific field/interests
- "method": The main methodology (2-4 words)

Return ONLY a JSON array, no other text:
[{{"paper_num": 1, "score": 8, "summary": "...", "why_relevant": "...", "method": "..."}}, ...]"""

    return prompt


def process_papers_with_gemini(
    api_key: str,
    user_profile: dict,
    papers: list,
    batch_size: int = 5,
    progress_callback=None
) -> Tuple[list, list]:
    """
    Process papers through Gemini for ranking and summarization.
    
    Returns:
        Tuple of (enriched_papers, errors)
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai package not installed")
    
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    if not papers:
        return [], []
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Store results by original paper index
    results_by_index = {}
    errors = []
    
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(papers), batch_size)):
        batch = papers[i:i + batch_size]
        start_index = i  # 0-based start index for this batch
        
        if progress_callback:
            progress_callback(f"Processing batch {batch_num + 1}/{total_batches}...")
        
        prompt = build_ranking_prompt(user_profile, batch, start_index)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                )
            )
            
            # Check if response was blocked
            if not response.text:
                errors.append(f"Batch {batch_num + 1}: Empty response from API")
                continue
            
            # Parse the response
            rankings = parse_gemini_response(response.text)
            
            if not rankings:
                errors.append(f"Batch {batch_num + 1}: Could not parse JSON response")
                continue
            
            # Match results to papers using paper_num
            for ranking in rankings:
                paper_num = ranking.get("paper_num")
                if paper_num is not None:
                    # Convert 1-based paper_num to 0-based index
                    original_index = paper_num - 1
                    if 0 <= original_index < len(papers):
                        results_by_index[original_index] = ranking
                        
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "401" in error_msg:
                errors.append(f"Invalid API key. Please check your Gemini API key.")
            elif "QUOTA" in error_msg.upper() or "429" in error_msg:
                errors.append(f"API quota exceeded. Please try again later.")
            else:
                errors.append(f"Batch {batch_num + 1}: {error_msg[:100]}")
            continue
    
    # Build enriched papers list
    enriched_papers = []
    scored_count = 0
    
    for idx, paper in enumerate(papers):
        ranking = results_by_index.get(idx, {})
        
        has_ai_analysis = bool(ranking.get("summary"))
        if has_ai_analysis:
            scored_count += 1
        
        enriched = {
            **paper,
            "relevance_score": ranking.get("score", 5),
            "ai_contribution": ranking.get("summary", ""),
            "ai_relevance": ranking.get("why_relevant", ""),
            "ai_methodology": ranking.get("method", ""),
            "has_ai_analysis": has_ai_analysis
        }
        enriched_papers.append(enriched)
    
    # Add summary info
    if scored_count == 0 and len(papers) > 0:
        errors.append("AI analysis failed for all papers. Check your API key.")
    elif scored_count < len(papers):
        errors.append(f"AI analyzed {scored_count}/{len(papers)} papers. Some batches may have failed.")
    
    # Sort by relevance score (descending)
    enriched_papers.sort(key=lambda x: (x["relevance_score"], x.get("has_ai_analysis", False)), reverse=True)
    
    return enriched_papers, errors


def parse_gemini_response(response_text: str) -> list:
    """Parse Gemini's JSON response with robust error handling."""
    if not response_text:
        return []
    
    text = response_text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    # Try to find JSON array
    json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
    if json_match:
        text = json_match.group()
    
    # Clean up common issues
    text = text.replace('\n', ' ')
    text = re.sub(r',\s*]', ']', text)  # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)  # Remove trailing commas in objects
    
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        # Try to extract individual objects
        objects = []
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                objects.append(obj)
            except:
                continue
        return objects
    
    return []


def get_profile_options() -> dict:
    """Return all available options for building a user profile."""
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "secondary_interests": SECONDARY_INTERESTS,
        "methodologies": METHODOLOGIES
    }
