"""
Gemini AI Processor for Literature Discovery
Handles batch summarization, relevance scoring, and interest matching.

FIXED VERSION: Uses legacy google-generativeai SDK (more stable on Streamlit Cloud)
with updated model names and better error handling.
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

# Models to try in order (will use first one that works)
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

DEFAULT_MODEL = "gemini-2.0-flash"


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
    
    papers_text = ""
    for i, p in enumerate(papers):
        idx = start_index + i + 1
        abstract = p.get('abstract', '')[:800]
        papers_text += f"""
---
PAPER {idx}:
Title: {p.get('title', 'Unknown')}
Journal: {p.get('journal', 'Unknown')}
Abstract: {abstract}
"""
    
    prompt = f"""You are a research assistant helping find relevant papers.

{profile_text}

Analyze each paper's relevance to this researcher.

{papers_text}
---

IMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation, just the JSON.

For each paper return:
- paper_num: integer (1, 2, 3...)
- score: integer 1-10 (10=perfect match, 1=not relevant)
- summary: string (one sentence, main finding)
- why_relevant: string (one sentence, connection to user's interests)
- method: string (2-4 words, methodology used)

Example format:
[{{"paper_num": 1, "score": 8, "summary": "Finds X causes Y", "why_relevant": "Uses DiD which matches your methods", "method": "Difference-in-Differences"}}]

Your JSON array:"""

    return prompt


def get_sdk_info() -> dict:
    """Return information about SDK availability."""
    return {
        "available": GENAI_AVAILABLE,
        "default_model": DEFAULT_MODEL,
    }


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
    errors = []
    
    if not GENAI_AVAILABLE:
        return _fallback_processing(papers), ["google-generativeai package not installed. Run: pip install google-generativeai"]
    
    if not api_key:
        return _fallback_processing(papers), ["No Gemini API key provided"]
    
    if not papers:
        return [], []
    
    # Configure API
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        errors.append(f"Failed to configure API: {str(e)[:100]}")
        return _fallback_processing(papers), errors
    
    # Find a working model
    model = None
    working_model_name = None
    
    for model_name in MODELS_TO_TRY:
        try:
            model = genai.GenerativeModel(model_name)
            # Quick test
            test_response = model.generate_content("Say 'ok'")
            if test_response.text:
                working_model_name = model_name
                break
        except Exception as e:
            error_str = str(e)
            if "API_KEY" in error_str.upper() or "401" in error_str:
                errors.append("Invalid API key")
                return _fallback_processing(papers), errors
            elif "QUOTA" in error_str.upper() or "429" in error_str:
                errors.append("API quota exceeded. Wait a few minutes.")
                return _fallback_processing(papers), errors
            # Model not available, try next
            continue
    
    if not model or not working_model_name:
        errors.append("No working Gemini model found. Check your API key.")
        return _fallback_processing(papers), errors
    
    # Store results
    results_by_index = {}
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(papers), batch_size)):
        batch = papers[i:i + batch_size]
        start_index = i
        
        if progress_callback:
            progress_callback(f"Batch {batch_num + 1}/{total_batches} ({working_model_name})...")
        
        prompt = build_ranking_prompt(user_profile, batch, start_index)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                )
            )
            
            if not response.text:
                errors.append(f"Batch {batch_num + 1}: Empty response")
                continue
            
            rankings = parse_gemini_response(response.text)
            
            if not rankings:
                preview = response.text[:150].replace('\n', ' ')
                errors.append(f"Batch {batch_num + 1}: Could not parse response: {preview}...")
                continue
            
            # Match results
            for ranking in rankings:
                paper_num = ranking.get("paper_num")
                if paper_num is not None:
                    original_index = paper_num - 1
                    if 0 <= original_index < len(papers):
                        results_by_index[original_index] = ranking
                        
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "401" in error_msg:
                errors.append("Invalid API key")
                break
            elif "QUOTA" in error_msg.upper() or "429" in error_msg:
                errors.append("Quota exceeded. Try again later.")
                break
            else:
                errors.append(f"Batch {batch_num + 1}: {error_msg[:100]}")
            continue
    
    # Build results
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
    
    # Summary
    if scored_count == 0 and len(papers) > 0:
        errors.insert(0, f"⚠️ AI failed for all {len(papers)} papers")
    elif scored_count < len(papers):
        errors.insert(0, f"ℹ️ Analyzed {scored_count}/{len(papers)} papers")
    else:
        errors.insert(0, f"✓ Analyzed all {scored_count} papers with {working_model_name}")
    
    enriched_papers.sort(
        key=lambda x: (x.get("has_ai_analysis", False), x.get("relevance_score", 0)), 
        reverse=True
    )
    
    return enriched_papers, errors


def _fallback_processing(papers: list) -> list:
    """Fallback when AI isn't available."""
    return [
        {
            **paper,
            "relevance_score": 5,
            "ai_contribution": "",
            "ai_relevance": "",
            "ai_methodology": "",
            "has_ai_analysis": False
        }
        for paper in papers
    ]


def parse_gemini_response(response_text: str) -> list:
    """Parse Gemini's JSON response."""
    if not response_text:
        return []
    
    text = response_text.strip()
    
    # Remove markdown
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Find JSON array
    json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
    if json_match:
        text = json_match.group()
    
    # Clean
    text = text.replace('\n', ' ')
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)
    
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    
    # Fallback: extract individual objects
    objects = []
    pattern = r'\{\s*"paper_num"\s*:\s*\d+[^}]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            clean = re.sub(r',\s*}', '}', match)
            obj = json.loads(clean)
            objects.append(obj)
        except:
            continue
    
    return objects


def get_profile_options() -> dict:
    """Return all available options for building a user profile."""
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "secondary_interests": SECONDARY_INTERESTS,
        "methodologies": METHODOLOGIES
    }
