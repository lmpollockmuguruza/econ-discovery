"""
AI Processor for Literature Discovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade semantic search with vector embeddings,
intelligent pre-ranking, and optimized LLM analysis.
"""

import json
import re
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SDK availability
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENHANCED USER PROFILE OPTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACADEMIC_LEVELS = [
    "Undergraduate",
    "Masters Student",
    "PhD Student (Early)",
    "PhD Student (ABD)",
    "Postdoctoral Fellow",
    "Assistant Professor",
    "Associate Professor",
    "Full Professor",
    "Industry Researcher",
    "Policy Analyst",
    "Think Tank Researcher",
    "Government Economist"
]

PRIMARY_FIELDS = [
    # Economics
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
    # Political Science
    "Political Economy",
    "Comparative Politics",
    "International Relations",
    "American Politics",
    "Political Theory",
    "Public Policy",
    "Political Methodology",
    "Security Studies",
    "Electoral Politics"
]

SECONDARY_INTERESTS = [
    # Methods
    "Causal Inference",
    "Machine Learning/AI",
    "Field Experiments (RCTs)",
    "Natural Experiments",
    "Structural Estimation",
    "Theory/Mechanism Design",
    "Survey Experiments",
    # Topics
    "Policy Evaluation",
    "Inequality & Redistribution",
    "Climate & Energy",
    "Education Policy",
    "Housing Markets",
    "Trade & Globalization",
    "Monetary Policy",
    "Fiscal Policy",
    "Innovation & Technology",
    "Gender & Discrimination",
    "Crime & Justice",
    "Health & Healthcare",
    "Immigration",
    "Democratic Institutions",
    "Voting & Elections",
    "Conflict & Security",
    "Media & Information",
    "Social Mobility",
    "Poverty & Welfare",
    "Regulation"
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
    "Network Analysis",
    "Bayesian Methods",
    "Qualitative Methods",
    "Case Studies",
    "Process Tracing"
]

REGIONAL_FOCUS = [
    "Global/Comparative",
    "United States",
    "European Union",
    "United Kingdom",
    "China",
    "India",
    "Latin America",
    "Sub-Saharan Africa",
    "Middle East & North Africa",
    "Southeast Asia",
    "Global South",
    "OECD Countries",
    "Emerging Markets"
]

# Prominent researchers by field for seed author suggestions
SEED_AUTHOR_SUGGESTIONS = {
    "Labor Economics": ["David Card", "Raj Chetty", "Lawrence Katz", "Claudia Goldin", "David Autor"],
    "Development Economics": ["Esther Duflo", "Abhijit Banerjee", "Michael Kremer", "Nathan Nunn"],
    "Public Economics": ["Emmanuel Saez", "Gabriel Zucman", "Raj Chetty", "Amy Finkelstein"],
    "Macroeconomics": ["Lawrence Summers", "Olivier Blanchard", "John Cochrane"],
    "Behavioral Economics": ["Richard Thaler", "Sendhil Mullainathan", "Stefano DellaVigna"],
    "Political Economy": ["Daron Acemoglu", "James Robinson", "Alberto Alesina"],
    "International Relations": ["Robert Keohane", "John Mearsheimer", "Beth Simmons"],
    "Comparative Politics": ["Theda Skocpol", "Robert Putnam", "Anna Grzymala-Busse"],
    "American Politics": ["Gary King", "Brandice Canes-Wrone", "Larry Bartels"],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class UserProfile:
    """Structured user research profile."""
    academic_level: str
    primary_field: str
    secondary_interests: List[str]
    preferred_methodology: List[str]
    regional_focus: str = "Global/Comparative"
    seed_authors: List[str] = field(default_factory=list)
    methodological_lean: float = 0.5  # 0=Qualitative, 1=Quantitative
    
    def to_text(self) -> str:
        """Convert profile to natural language for embedding."""
        method_desc = "quantitative causal inference" if self.methodological_lean > 0.6 else \
                      "qualitative and theoretical" if self.methodological_lean < 0.4 else \
                      "mixed methods"
        
        text = f"""
        Research Profile:
        - Career Stage: {self.academic_level}
        - Primary Field: {self.primary_field}
        - Research Interests: {', '.join(self.secondary_interests)}
        - Preferred Methods: {', '.join(self.preferred_methodology)}
        - Methodological Approach: {method_desc}
        - Regional Focus: {self.regional_focus}
        """
        if self.seed_authors:
            text += f"\n        - Follows work by: {', '.join(self.seed_authors)}"
        return text.strip()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "academic_level": self.academic_level,
            "primary_field": self.primary_field,
            "secondary_interests": self.secondary_interests,
            "preferred_methodology": self.preferred_methodology,
            "regional_focus": self.regional_focus,
            "seed_authors": self.seed_authors,
            "methodological_lean": self.methodological_lean
        }


@dataclass
class PaperAnalysis:
    """AI analysis result for a paper."""
    paper_id: str
    relevance_score: int
    summary: str
    why_relevant: str
    methodology: str
    topic_matches: List[str] = field(default_factory=list)
    method_matches: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AIConfig:
    """AI processing configuration."""
    EMBEDDING_MODEL = "models/embedding-001"
    GENERATION_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    DEFAULT_MODEL = "gemini-2.0-flash"
    
    BATCH_SIZE = 10  # Papers per LLM call
    TOP_K_FOR_LLM = 15  # Top semantic matches to send to LLM
    EMBEDDING_DIMENSION = 768
    
    TEMPERATURE = 0.15
    MAX_OUTPUT_TOKENS = 4096


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VECTOR OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns value between -1 and 1, where 1 is identical.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def get_embedding(text: str, model_name: str = AIConfig.EMBEDDING_MODEL) -> Optional[List[float]]:
    """
    Get embedding vector for text using Gemini embedding model.
    
    Returns None if embedding fails.
    """
    if not GENAI_AVAILABLE or not text:
        return None
    
    try:
        # Truncate text if too long (embedding model limit ~2048 tokens)
        text = text[:8000]
        
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document"
        )
        return result.get("embedding")
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def batch_get_embeddings(
    texts: List[str],
    model_name: str = AIConfig.EMBEDDING_MODEL
) -> List[Optional[List[float]]]:
    """
    Get embeddings for multiple texts efficiently.
    """
    embeddings = []
    for text in texts:
        embedding = get_embedding(text, model_name)
        embeddings.append(embedding)
    return embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEMANTIC PRE-RANKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def semantic_prerank_papers(
    profile: UserProfile,
    papers: List[dict],
    top_k: int = AIConfig.TOP_K_FOR_LLM,
    progress_callback: Optional[callable] = None
) -> Tuple[List[dict], List[float]]:
    """
    Pre-rank papers using semantic similarity to user profile.
    
    This reduces the number of papers sent to the LLM, saving tokens
    and improving relevance.
    
    Args:
        profile: User research profile
        papers: List of paper dictionaries
        top_k: Number of top papers to return
        progress_callback: Optional callback(message)
    
    Returns:
        Tuple of (ranked_papers, similarity_scores)
    """
    if not GENAI_AVAILABLE:
        logger.warning("Embeddings unavailable - returning unranked papers")
        return papers[:top_k], [0.5] * min(len(papers), top_k)
    
    if progress_callback:
        progress_callback("Computing profile embedding...")
    
    # Get profile embedding
    profile_embedding = get_embedding(profile.to_text())
    if not profile_embedding:
        logger.warning("Could not embed profile - returning unranked papers")
        return papers[:top_k], [0.5] * min(len(papers), top_k)
    
    # Score each paper
    scored_papers = []
    
    for i, paper in enumerate(papers):
        if progress_callback and i % 5 == 0:
            progress_callback(f"Embedding paper {i+1}/{len(papers)}...")
        
        # Create paper text for embedding
        paper_text = f"""
        Title: {paper.get('title', '')}
        Abstract: {paper.get('abstract', '')}
        Journal: {paper.get('journal', '')}
        """
        
        # Add concepts if available
        concepts = paper.get('concepts', [])
        if concepts:
            concept_names = [c.get('name', '') for c in concepts[:5]]
            paper_text += f"\nTopics: {', '.join(concept_names)}"
        
        # Get paper embedding
        paper_embedding = get_embedding(paper_text)
        
        if paper_embedding:
            similarity = cosine_similarity(profile_embedding, paper_embedding)
        else:
            similarity = 0.5  # Neutral score if embedding fails
        
        scored_papers.append((paper, similarity))
    
    # Sort by similarity (descending)
    scored_papers.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    ranked_papers = [p for p, _ in scored_papers[:top_k]]
    scores = [s for _, s in scored_papers[:top_k]]
    
    if progress_callback:
        progress_callback(f"Pre-ranked to top {len(ranked_papers)} papers")
    
    return ranked_papers, scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_analysis_prompt(profile: UserProfile, papers: List[dict]) -> str:
    """
    Build the prompt for Gemini to analyze and rank papers.
    
    Uses "Senior Research Editor" persona for high-quality analysis.
    """
    method_desc = "quantitative and causal inference methods" if profile.methodological_lean > 0.6 else \
                  "qualitative and theoretical approaches" if profile.methodological_lean < 0.4 else \
                  "both quantitative and qualitative methods"
    
    profile_section = f"""
═══════════════════════════════════════════════════════════════
RESEARCHER PROFILE
═══════════════════════════════════════════════════════════════
• Career Stage: {profile.academic_level}
• Primary Field: {profile.primary_field}
• Research Interests: {', '.join(profile.secondary_interests)}
• Preferred Methods: {', '.join(profile.preferred_methodology)}
• Methodological Preference: {method_desc}
• Regional Focus: {profile.regional_focus}
{"• Follows work by: " + ', '.join(profile.seed_authors) if profile.seed_authors else ""}
═══════════════════════════════════════════════════════════════
"""
    
    papers_section = "\nPAPERS TO ANALYZE:\n" + "─" * 60 + "\n"
    
    for i, paper in enumerate(papers, 1):
        abstract = paper.get('abstract', '')[:1200]  # Limit abstract length
        concepts = paper.get('concepts', [])
        concept_str = ', '.join([c.get('name', '') for c in concepts[:4]]) if concepts else "N/A"
        
        papers_section += f"""
[PAPER {i}]
Title: {paper.get('title', 'Unknown')}
Journal: {paper.get('journal', 'Unknown')} | Cited: {paper.get('cited_by_count', 0)}
Topics: {concept_str}
Abstract: {abstract}
─────────────────────────────────────────────────────────────
"""
    
    instruction = """
You are a Senior Research Editor at a top academic journal. Your task is to evaluate 
these papers for a researcher with the profile above.

For EACH paper, provide a JSON object with:
- paper_num: (integer) The paper number [1, 2, 3, ...]
- score: (integer 1-10) Relevance score where:
  • 9-10: Essential reading - directly advances their research agenda
  • 7-8: Highly relevant - strong methodological or topical match
  • 5-6: Moderately relevant - tangential connection
  • 3-4: Marginally relevant - weak connection
  • 1-2: Not relevant - outside their interests
- summary: (string) One clear sentence: the paper's main contribution/finding
- why_relevant: (string) One sentence explaining the specific connection to their profile
- method: (string) 2-4 words describing the methodology
- topic_matches: (array of strings) Which of their interests this paper touches
- method_matches: (array of strings) Which of their preferred methods this paper uses

CRITICAL: Return ONLY a valid JSON array. No markdown, no explanations, no preamble.

Example format:
[{"paper_num": 1, "score": 8, "summary": "Finds that X causes Y using a natural experiment.", "why_relevant": "Uses difference-in-differences on education policy, matching your labor economics focus.", "method": "Difference-in-Differences", "topic_matches": ["Education Policy", "Causal Inference"], "method_matches": ["Difference-in-Differences"]}]
"""
    
    return profile_section + papers_section + instruction


def build_synthesis_prompt(profile: UserProfile, top_papers: List[dict]) -> str:
    """
    Build prompt for synthesizing top paper recommendations.
    """
    papers_text = ""
    for i, paper in enumerate(top_papers[:3], 1):
        papers_text += f"""
Paper {i}: "{paper.get('title', 'Unknown')}"
- Summary: {paper.get('ai_contribution', 'N/A')}
- Why Relevant: {paper.get('ai_relevance', 'N/A')}
- Method: {paper.get('ai_methodology', 'N/A')}
"""
    
    return f"""
As a Senior Research Editor, write a brief synthesis (3-4 sentences) explaining why 
these three papers are the most important for a {profile.academic_level} focused on 
{profile.primary_field} with interests in {', '.join(profile.secondary_interests[:3])}.

{papers_text}

Focus on how these papers connect to each other and to the researcher's agenda.
Write in a professional, editorial voice. Be specific about methodological innovations
or empirical findings that make these essential reading.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_working_model() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Find a working Gemini model.
    
    Returns:
        Tuple of (model_instance, model_name, error_message)
    """
    if not GENAI_AVAILABLE:
        return None, None, "google-generativeai package not installed"
    
    for model_name in AIConfig.GENERATION_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            # Quick test
            test_response = model.generate_content("Reply with: OK")
            if test_response.text:
                return model, model_name, None
        except Exception as e:
            error_str = str(e).upper()
            if "API_KEY" in error_str or "401" in error_str:
                return None, None, "Invalid API key"
            elif "QUOTA" in error_str or "429" in error_str:
                return None, None, "API quota exceeded - wait a few minutes"
            # Try next model
            continue
    
    return None, None, "No working Gemini model found"


def parse_gemini_response(response_text: str) -> List[dict]:
    """
    Parse Gemini's JSON response with robust error handling.
    """
    if not response_text:
        return []
    
    text = response_text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()
    
    # Find JSON array
    json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', text)
    if json_match:
        text = json_match.group()
    
    # Clean common JSON issues
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


def process_papers_with_gemini(
    api_key: str,
    user_profile: UserProfile,
    papers: List[dict],
    use_semantic_prerank: bool = True,
    progress_callback: Optional[callable] = None
) -> Tuple[List[dict], List[str], Optional[str]]:
    """
    Process papers through semantic pre-ranking and Gemini analysis.
    
    Args:
        api_key: Gemini API key
        user_profile: User's research profile
        papers: List of paper dictionaries
        use_semantic_prerank: Whether to use embedding-based pre-ranking
        progress_callback: Optional callback(message)
    
    Returns:
        Tuple of (enriched_papers, status_messages, synthesis)
    """
    messages = []
    synthesis = None
    
    if not GENAI_AVAILABLE:
        messages.append("⚠️ AI SDK not available")
        return _fallback_processing(papers), messages, None
    
    if not api_key:
        messages.append("⚠️ No API key provided")
        return _fallback_processing(papers), messages, None
    
    if not papers:
        return [], ["No papers to process"], None
    
    # Configure API
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        messages.append(f"⚠️ API configuration failed: {str(e)[:50]}")
        return _fallback_processing(papers), messages, None
    
    # Step 1: Semantic pre-ranking (if enabled)
    if use_semantic_prerank and len(papers) > AIConfig.TOP_K_FOR_LLM:
        if progress_callback:
            progress_callback("🧠 Computing semantic similarity...")
        
        try:
            preranked_papers, similarity_scores = semantic_prerank_papers(
                user_profile,
                papers,
                top_k=AIConfig.TOP_K_FOR_LLM,
                progress_callback=progress_callback
            )
            
            # Store similarity scores
            for paper, score in zip(preranked_papers, similarity_scores):
                paper['semantic_similarity'] = score
            
            messages.append(f"📊 Pre-ranked {len(papers)} → top {len(preranked_papers)} by semantic match")
            papers_for_llm = preranked_papers
            
        except Exception as e:
            logger.warning(f"Semantic pre-ranking failed: {e}")
            messages.append(f"⚠️ Semantic ranking failed, using all papers")
            papers_for_llm = papers[:AIConfig.TOP_K_FOR_LLM]
    else:
        papers_for_llm = papers[:AIConfig.TOP_K_FOR_LLM]
    
    # Step 2: Find working model
    if progress_callback:
        progress_callback("🔍 Connecting to Gemini...")
    
    model, model_name, error = _find_working_model()
    if error:
        messages.append(f"⚠️ {error}")
        return _fallback_processing(papers), messages, None
    
    messages.append(f"🤖 Using {model_name}")
    
    # Step 3: Process papers with LLM
    results_by_index = {}
    total_batches = (len(papers_for_llm) + AIConfig.BATCH_SIZE - 1) // AIConfig.BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * AIConfig.BATCH_SIZE
        end_idx = min(start_idx + AIConfig.BATCH_SIZE, len(papers_for_llm))
        batch = papers_for_llm[start_idx:end_idx]
        
        if progress_callback:
            progress_callback(f"🔬 Analyzing batch {batch_num + 1}/{total_batches}...")
        
        prompt = build_analysis_prompt(user_profile, batch)
        
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=AIConfig.TEMPERATURE,
                    max_output_tokens=AIConfig.MAX_OUTPUT_TOKENS,
                )
            )
            
            if not response.text:
                messages.append(f"⚠️ Batch {batch_num + 1}: Empty response")
                continue
            
            rankings = parse_gemini_response(response.text)
            
            if not rankings:
                preview = response.text[:100].replace('\n', ' ')
                messages.append(f"⚠️ Batch {batch_num + 1}: Parse error")
                logger.warning(f"Could not parse: {preview}...")
                continue
            
            # Match results to papers
            for ranking in rankings:
                paper_num = ranking.get("paper_num")
                if paper_num is not None:
                    global_idx = start_idx + paper_num - 1
                    if 0 <= global_idx < len(papers_for_llm):
                        results_by_index[global_idx] = ranking
                        
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper() or "401" in error_msg:
                messages.append("⚠️ Invalid API key")
                break
            elif "QUOTA" in error_msg.upper() or "429" in error_msg:
                messages.append("⚠️ Quota exceeded - try again later")
                break
            else:
                messages.append(f"⚠️ Batch {batch_num + 1}: {str(e)[:50]}")
            continue
    
    # Step 4: Build enriched paper list
    enriched_papers = []
    analyzed_count = 0
    
    for idx, paper in enumerate(papers_for_llm):
        ranking = results_by_index.get(idx, {})
        has_analysis = bool(ranking.get("summary"))
        
        if has_analysis:
            analyzed_count += 1
        
        enriched = {
            **paper,
            "relevance_score": ranking.get("score", 5),
            "ai_contribution": ranking.get("summary", ""),
            "ai_relevance": ranking.get("why_relevant", ""),
            "ai_methodology": ranking.get("method", ""),
            "topic_matches": ranking.get("topic_matches", []),
            "method_matches": ranking.get("method_matches", []),
            "has_ai_analysis": has_analysis,
            "semantic_similarity": paper.get("semantic_similarity", 0.5)
        }
        enriched_papers.append(enriched)
    
    # Sort by relevance
    enriched_papers.sort(
        key=lambda x: (x.get("has_ai_analysis", False), x.get("relevance_score", 0)),
        reverse=True
    )
    
    # Step 5: Generate synthesis for top papers
    if analyzed_count >= 3:
        if progress_callback:
            progress_callback("📝 Generating synthesis...")
        
        try:
            synthesis_prompt = build_synthesis_prompt(user_profile, enriched_papers[:3])
            synth_response = model.generate_content(
                synthesis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                )
            )
            synthesis = synth_response.text.strip() if synth_response.text else None
        except Exception as e:
            logger.warning(f"Synthesis generation failed: {e}")
    
    # Final status
    if analyzed_count == 0:
        messages.insert(0, f"⚠️ Analysis failed for all {len(papers_for_llm)} papers")
    elif analyzed_count < len(papers_for_llm):
        messages.insert(0, f"✓ Analyzed {analyzed_count}/{len(papers_for_llm)} papers")
    else:
        messages.insert(0, f"✓ Successfully analyzed all {analyzed_count} papers")
    
    return enriched_papers, messages, synthesis


def _fallback_processing(papers: List[dict]) -> List[dict]:
    """Fallback when AI isn't available - returns papers with neutral scores."""
    return [
        {
            **paper,
            "relevance_score": 5,
            "ai_contribution": "",
            "ai_relevance": "",
            "ai_methodology": "",
            "topic_matches": [],
            "method_matches": [],
            "has_ai_analysis": False,
            "semantic_similarity": 0.5
        }
        for paper in papers
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_user_profile(
    academic_level: str,
    primary_field: str,
    secondary_interests: List[str],
    preferred_methodology: List[str],
    regional_focus: str = "Global/Comparative",
    seed_authors: Optional[List[str]] = None,
    methodological_lean: float = 0.5
) -> UserProfile:
    """Create a structured user profile."""
    return UserProfile(
        academic_level=academic_level,
        primary_field=primary_field,
        secondary_interests=secondary_interests,
        preferred_methodology=preferred_methodology,
        regional_focus=regional_focus,
        seed_authors=seed_authors or [],
        methodological_lean=methodological_lean
    )


@lru_cache(maxsize=1)
def get_profile_options() -> dict:
    """Return all available options for building a user profile."""
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "secondary_interests": SECONDARY_INTERESTS,
        "methodologies": METHODOLOGIES,
        "regional_focus": REGIONAL_FOCUS,
        "seed_author_suggestions": SEED_AUTHOR_SUGGESTIONS
    }


def get_sdk_info() -> dict:
    """Return information about SDK availability."""
    return {
        "available": GENAI_AVAILABLE,
        "default_model": AIConfig.DEFAULT_MODEL,
        "embedding_model": AIConfig.EMBEDDING_MODEL,
        "supports_embeddings": GENAI_AVAILABLE
    }


def get_suggested_authors(field: str) -> List[str]:
    """Get suggested seed authors for a field."""
    return SEED_AUTHOR_SUGGESTIONS.get(field, [])
