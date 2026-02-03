"""
Gemini AI Processor for Literature Discovery
Handles batch summarization, relevance scoring, and interest matching.

UPDATED: Uses new google-genai SDK (google-generativeai is deprecated)
Model: gemini-2.5-flash-lite (optimized for cost/latency)
"""

import json
import re
from typing import Optional, Tuple

# Try new SDK first, fall back to legacy
try:
    from google import genai
    from google.genai import types
    GENAI_SDK = "new"
    GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GENAI_SDK = "legacy"
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_SDK = None
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

# Model options - from cheapest/fastest to most capable
AVAILABLE_MODELS = {
    "gemini-2.5-flash-lite": "Fastest, lowest cost - good for MVP",
    "gemini-2.5-flash": "Balanced speed and quality",
    "gemini-2.0-flash": "Legacy stable option",
}

DEFAULT_MODEL = "gemini-2.5-flash-lite"


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
        idx = start_index + i + 1  # 1-based index
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
    """Return information about which SDK is being used."""
    return {
        "sdk": GENAI_SDK,
        "available": GENAI_AVAILABLE,
        "default_model": DEFAULT_MODEL,
        "available_models": AVAILABLE_MODELS
    }


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Validate the Gemini API key before processing."""
    if not GENAI_AVAILABLE:
        if GENAI_SDK is None:
            return False, "No Gemini SDK installed. Run: pip install google-genai"
        return False, "Gemini SDK import failed"
    
    if not api_key:
        return False, "No API key provided"
    
    if len(api_key) < 30:
        return False, "API key appears too short"
    
    try:
        if GENAI_SDK == "new":
            # New SDK: create client with API key
            client = genai.Client(api_key=api_key)
            # Test by listing models
            models = list(client.models.list())
            if not models:
                return False, "Could not retrieve model list"
            return True, f"API key validated (new SDK, {len(models)} models available)"
        else:
            # Legacy SDK
            genai_legacy.configure(api_key=api_key)
            models = list(genai_legacy.list_models())
            if not models:
                return False, "Could not retrieve model list"
            return True, f"API key validated (legacy SDK)"
            
    except Exception as e:
        error_str = str(e)
        if "API_KEY" in error_str.upper() or "401" in error_str:
            return False, "Invalid API key"
        elif "QUOTA" in error_str.upper() or "429" in error_str:
            return False, "API quota exceeded"
        else:
            return False, f"Validation error: {error_str[:100]}"


def _generate_with_new_sdk(client, model: str, prompt: str) -> str:
    """Generate content using the new google-genai SDK."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )
    )
    return response.text


def _generate_with_legacy_sdk(model_name: str, prompt: str) -> str:
    """Generate content using the legacy google-generativeai SDK."""
    model = genai_legacy.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config=genai_legacy.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )
    )
    return response.text


def process_papers_with_gemini(
    api_key: str,
    user_profile: dict,
    papers: list,
    batch_size: int = 5,
    model_name: str = DEFAULT_MODEL,
    progress_callback=None
) -> Tuple[list, list]:
    """
    Process papers through Gemini for ranking and summarization.
    
    Args:
        api_key: Gemini API key
        user_profile: User's research profile
        papers: List of papers to analyze
        batch_size: Papers per API call
        model_name: Gemini model to use
        progress_callback: Optional callback for progress updates
    
    Returns:
        Tuple of (enriched_papers, errors)
    """
    errors = []
    
    # Validate prerequisites
    if not GENAI_AVAILABLE:
        install_cmd = "pip install google-genai" if GENAI_SDK is None else "check import errors"
        return _fallback_processing(papers), [f"Gemini SDK not available. {install_cmd}"]
    
    if not api_key:
        return _fallback_processing(papers), ["No Gemini API key provided"]
    
    if not papers:
        return [], []
    
    # Validate API key first
    is_valid, validation_msg = validate_api_key(api_key)
    if not is_valid:
        errors.append(f"API Key Error: {validation_msg}")
        return _fallback_processing(papers), errors
    
    # Initialize client/model based on SDK
    try:
        if GENAI_SDK == "new":
            client = genai.Client(api_key=api_key)
        else:
            genai_legacy.configure(api_key=api_key)
            client = None
    except Exception as e:
        errors.append(f"Failed to initialize Gemini: {str(e)[:100]}")
        return _fallback_processing(papers), errors
    
    # Store results by original paper index
    results_by_index = {}
    total_batches = (len(papers) + batch_size - 1) // batch_size
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(papers), batch_size)):
        batch = papers[i:i + batch_size]
        start_index = i
        
        if progress_callback:
            progress_callback(f"Processing batch {batch_num + 1}/{total_batches}...")
        
        prompt = build_ranking_prompt(user_profile, batch, start_index)
        
        try:
            # Generate based on SDK version
            if GENAI_SDK == "new":
                response_text = _generate_with_new_sdk(client, model_name, prompt)
            else:
                # Legacy SDK - model names might differ
                legacy_model = model_name
                if model_name == "gemini-2.5-flash-lite":
                    legacy_model = "gemini-1.5-flash"  # Fallback for legacy
                response_text = _generate_with_legacy_sdk(legacy_model, prompt)
            
            if not response_text:
                errors.append(f"Batch {batch_num + 1}: Empty response from Gemini")
                continue
            
            # Parse the response
            rankings = parse_gemini_response(response_text)
            
            if not rankings:
                preview = response_text[:200].replace('\n', ' ')
                errors.append(f"Batch {batch_num + 1}: Could not parse JSON. Preview: {preview}...")
                continue
            
            # Match results to papers
            matched_count = 0
            for ranking in rankings:
                paper_num = ranking.get("paper_num")
                if paper_num is not None:
                    original_index = paper_num - 1
                    if 0 <= original_index < len(papers):
                        results_by_index[original_index] = ranking
                        matched_count += 1
            
            if matched_count == 0:
                errors.append(f"Batch {batch_num + 1}: Parsed {len(rankings)} items but none matched")
                        
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "401" in error_msg:
                errors.append("Invalid API key. Please check your Gemini API key.")
                break
            elif "QUOTA" in error_msg.upper() or "429" in error_msg:
                errors.append("API quota exceeded. Try again later or reduce batch size.")
                break
            elif "SAFETY" in error_msg.upper() or "blocked" in error_msg.lower():
                errors.append(f"Batch {batch_num + 1}: Content filtered by safety settings")
            elif "not found" in error_msg.lower() or "404" in error_msg:
                errors.append(f"Model '{model_name}' not found. Try 'gemini-2.0-flash' instead.")
                break
            else:
                errors.append(f"Batch {batch_num + 1} error: {error_msg[:150]}")
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
    
    # Summary
    if scored_count == 0 and len(papers) > 0:
        errors.insert(0, f"⚠️ AI analysis failed for ALL {len(papers)} papers. Check errors below.")
    elif scored_count < len(papers):
        errors.insert(0, f"ℹ️ AI analyzed {scored_count}/{len(papers)} papers successfully.")
    
    # Sort by relevance score
    enriched_papers.sort(
        key=lambda x: (x.get("has_ai_analysis", False), x.get("relevance_score", 0)), 
        reverse=True
    )
    
    return enriched_papers, errors


def _fallback_processing(papers: list) -> list:
    """Fallback when AI processing isn't available."""
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
    """Parse Gemini's JSON response with robust error handling."""
    if not response_text:
        return []
    
    text = response_text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    
    # Try to find JSON array
    json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
    if json_match:
        text = json_match.group()
    
    # Clean up common JSON issues
    text = text.replace('\n', ' ')
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)
    
    # First attempt: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass
    
    # Second attempt: find objects with paper_num
    objects = []
    pattern = r'\{\s*"paper_num"\s*:\s*\d+[^}]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            clean_match = re.sub(r',\s*}', '}', match)
            obj = json.loads(clean_match)
            objects.append(obj)
        except json.JSONDecodeError:
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
