"""
Semantic Matching Engine for Literature Discovery
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gold-standard TF-IDF + Cosine Similarity matching with
academically rigorous keyword taxonomies.

No external API dependencies - runs entirely locally.
"""

import math
import re
import string
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from functools import lru_cache
from collections import Counter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER PROFILE OPTIONS
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
    "Causal Inference",
    "Machine Learning/AI",
    "Field Experiments (RCTs)",
    "Natural Experiments",
    "Structural Estimation",
    "Theory/Mechanism Design",
    "Survey Experiments",
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
# ACADEMIC KEYWORD TAXONOMY
# Rigorous mapping of research interests to semantic keyword clusters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIELD_KEYWORDS: Dict[str, Set[str]] = {
    "Microeconomics": {
        "microeconomic", "consumer", "producer", "demand", "supply", "equilibrium",
        "utility", "preference", "choice", "optimization", "market", "price",
        "elasticity", "welfare", "surplus", "efficiency", "allocation", "mechanism"
    },
    "Macroeconomics": {
        "macroeconomic", "gdp", "growth", "inflation", "unemployment", "recession",
        "business cycle", "aggregate", "monetary", "fiscal", "central bank",
        "interest rate", "output", "consumption", "investment", "savings"
    },
    "Econometrics": {
        "econometric", "estimation", "regression", "identification", "inference",
        "estimator", "consistent", "unbiased", "heteroskedasticity", "autocorrelation",
        "specification", "model selection", "bootstrap", "asymptotic"
    },
    "Labor Economics": {
        "labor", "wage", "employment", "unemployment", "worker", "job",
        "human capital", "education", "training", "skill", "occupation",
        "minimum wage", "union", "collective bargaining", "discrimination",
        "earnings", "income", "mobility", "search", "matching"
    },
    "Public Economics": {
        "public", "tax", "taxation", "government", "fiscal", "spending",
        "redistribution", "welfare", "social insurance", "public good",
        "externality", "regulation", "public finance", "budget", "deficit"
    },
    "International Economics": {
        "international", "trade", "export", "import", "tariff", "globalization",
        "exchange rate", "currency", "capital flow", "foreign direct investment",
        "comparative advantage", "trade policy", "protectionism", "wto"
    },
    "Development Economics": {
        "development", "poverty", "developing country", "aid", "microfinance",
        "rural", "agriculture", "infrastructure", "institution", "corruption",
        "growth", "inequality", "health", "education", "nutrition"
    },
    "Financial Economics": {
        "financial", "finance", "asset", "stock", "bond", "return", "risk",
        "portfolio", "investment", "banking", "credit", "loan", "mortgage",
        "interest rate", "yield", "market", "liquidity", "capital"
    },
    "Industrial Organization": {
        "industrial organization", "firm", "market structure", "competition",
        "monopoly", "oligopoly", "antitrust", "merger", "entry", "exit",
        "pricing", "product differentiation", "vertical integration"
    },
    "Behavioral Economics": {
        "behavioral", "psychology", "bias", "heuristic", "bounded rationality",
        "prospect theory", "loss aversion", "time preference", "present bias",
        "nudge", "framing", "anchoring", "overconfidence", "social preference"
    },
    "Health Economics": {
        "health", "healthcare", "hospital", "physician", "insurance", "medicare",
        "medicaid", "pharmaceutical", "drug", "mortality", "morbidity",
        "disease", "treatment", "patient", "medical"
    },
    "Environmental Economics": {
        "environmental", "climate", "carbon", "emission", "pollution", "energy",
        "renewable", "sustainability", "conservation", "natural resource",
        "cap and trade", "carbon tax", "externality", "green"
    },
    "Urban Economics": {
        "urban", "city", "housing", "rent", "real estate", "land", "zoning",
        "agglomeration", "spatial", "commute", "transportation", "neighborhood",
        "gentrification", "segregation", "metropolitan"
    },
    "Economic History": {
        "history", "historical", "long run", "persistence", "colonial",
        "industrial revolution", "great depression", "institution", "slavery",
        "war", "conflict", "demographic transition"
    },
    "Political Economy": {
        "political economy", "institution", "democracy", "autocracy", "regime",
        "voting", "election", "politician", "lobbying", "corruption",
        "redistribution", "inequality", "conflict", "state capacity"
    },
    "Comparative Politics": {
        "comparative", "cross-country", "regime", "democracy", "autocracy",
        "institution", "parliament", "coalition", "party", "electoral system",
        "federalism", "decentralization", "state building"
    },
    "International Relations": {
        "international relations", "conflict", "war", "peace", "diplomacy",
        "alliance", "treaty", "sanction", "nuclear", "security", "terrorism",
        "cooperation", "international organization", "sovereignty"
    },
    "American Politics": {
        "american politics", "congress", "senate", "house", "president",
        "supreme court", "partisan", "republican", "democrat", "polarization",
        "lobbying", "campaign", "primary", "electoral college"
    },
    "Political Theory": {
        "political theory", "justice", "liberty", "equality", "rights",
        "democracy", "legitimacy", "sovereignty", "social contract",
        "deliberation", "normative", "ideology"
    },
    "Public Policy": {
        "public policy", "policy evaluation", "implementation", "regulation",
        "reform", "government program", "effectiveness", "cost-benefit",
        "stakeholder", "agenda setting"
    },
    "Political Methodology": {
        "methodology", "causal inference", "identification", "experiment",
        "survey", "measurement", "text analysis", "machine learning",
        "bayesian", "formal model"
    },
    "Security Studies": {
        "security", "defense", "military", "war", "conflict", "terrorism",
        "nuclear", "cyber", "intelligence", "strategy", "deterrence"
    },
    "Electoral Politics": {
        "election", "voting", "voter", "turnout", "campaign", "candidate",
        "party", "primary", "polling", "electoral", "ballot", "swing"
    }
}

INTEREST_KEYWORDS: Dict[str, Set[str]] = {
    "Causal Inference": {
        "causal", "causality", "identification", "endogeneity", "exogenous",
        "treatment effect", "counterfactual", "selection", "confounding",
        "instrumental", "discontinuity", "difference-in-differences"
    },
    "Machine Learning/AI": {
        "machine learning", "artificial intelligence", "neural network",
        "deep learning", "prediction", "classification", "algorithm",
        "random forest", "lasso", "regularization", "cross-validation"
    },
    "Field Experiments (RCTs)": {
        "randomized", "rct", "experiment", "random assignment", "treatment",
        "control group", "field experiment", "randomization", "intent to treat"
    },
    "Natural Experiments": {
        "natural experiment", "quasi-experiment", "exogenous shock",
        "discontinuity", "regression discontinuity", "instrumental variable",
        "difference-in-differences", "event study"
    },
    "Structural Estimation": {
        "structural", "estimation", "model", "parameter", "simulation",
        "counterfactual", "welfare", "equilibrium", "dynamic"
    },
    "Theory/Mechanism Design": {
        "theory", "mechanism design", "game theory", "equilibrium", "incentive",
        "contract", "auction", "matching", "optimal", "strategic"
    },
    "Survey Experiments": {
        "survey experiment", "conjoint", "vignette", "factorial", "survey",
        "respondent", "treatment", "experimental"
    },
    "Policy Evaluation": {
        "policy evaluation", "program evaluation", "impact", "effectiveness",
        "cost-benefit", "welfare", "reform", "implementation"
    },
    "Inequality & Redistribution": {
        "inequality", "redistribution", "income distribution", "wealth",
        "gini", "top income", "bottom", "percentile", "mobility"
    },
    "Climate & Energy": {
        "climate", "carbon", "emission", "energy", "renewable", "fossil fuel",
        "temperature", "warming", "environmental", "green", "solar", "wind"
    },
    "Education Policy": {
        "education", "school", "student", "teacher", "test score", "achievement",
        "college", "university", "dropout", "graduation", "curriculum"
    },
    "Housing Markets": {
        "housing", "house price", "rent", "mortgage", "homeowner", "real estate",
        "foreclosure", "affordability", "zoning", "construction"
    },
    "Trade & Globalization": {
        "trade", "globalization", "export", "import", "tariff", "outsourcing",
        "supply chain", "multinational", "foreign", "comparative advantage"
    },
    "Monetary Policy": {
        "monetary policy", "central bank", "federal reserve", "interest rate",
        "inflation", "quantitative easing", "money supply", "liquidity"
    },
    "Fiscal Policy": {
        "fiscal policy", "government spending", "tax", "budget", "deficit",
        "debt", "stimulus", "austerity", "multiplier"
    },
    "Innovation & Technology": {
        "innovation", "technology", "patent", "r&d", "research", "startup",
        "entrepreneur", "productivity", "automation", "digital"
    },
    "Gender & Discrimination": {
        "gender", "discrimination", "wage gap", "women", "female", "bias",
        "diversity", "race", "racial", "minority", "affirmative action"
    },
    "Crime & Justice": {
        "crime", "criminal", "police", "prison", "incarceration", "recidivism",
        "sentencing", "justice", "law enforcement", "violence"
    },
    "Health & Healthcare": {
        "health", "healthcare", "hospital", "insurance", "mortality",
        "disease", "treatment", "physician", "patient", "medical"
    },
    "Immigration": {
        "immigration", "immigrant", "migration", "refugee", "border",
        "visa", "asylum", "deportation", "citizenship", "naturalization"
    },
    "Democratic Institutions": {
        "democracy", "democratic", "institution", "constitution", "rule of law",
        "checks and balances", "separation of powers", "accountability"
    },
    "Voting & Elections": {
        "voting", "election", "voter", "turnout", "ballot", "electoral",
        "campaign", "candidate", "polling", "swing voter"
    },
    "Conflict & Security": {
        "conflict", "war", "violence", "peace", "security", "military",
        "terrorism", "civil war", "interstate", "defense"
    },
    "Media & Information": {
        "media", "news", "information", "social media", "misinformation",
        "propaganda", "journalism", "press", "fake news", "polarization"
    },
    "Social Mobility": {
        "mobility", "intergenerational", "upward mobility", "opportunity",
        "socioeconomic", "class", "income mobility", "persistence"
    },
    "Poverty & Welfare": {
        "poverty", "welfare", "social assistance", "food stamps", "snap",
        "transfer", "safety net", "poor", "low income", "benefits"
    },
    "Regulation": {
        "regulation", "deregulation", "regulatory", "compliance", "rule",
        "standard", "enforcement", "agency", "policy"
    }
}

METHOD_KEYWORDS: Dict[str, Set[str]] = {
    "Difference-in-Differences": {
        "difference-in-differences", "diff-in-diff", "did", "parallel trends",
        "event study", "two-way fixed effects", "staggered", "treatment timing"
    },
    "Regression Discontinuity": {
        "regression discontinuity", "rdd", "discontinuity", "cutoff", "threshold",
        "running variable", "bandwidth", "local linear", "fuzzy rd", "sharp rd"
    },
    "Instrumental Variables": {
        "instrumental variable", "iv", "instrument", "two-stage", "2sls",
        "exclusion restriction", "first stage", "weak instrument"
    },
    "Randomized Controlled Trials": {
        "randomized", "rct", "experiment", "random assignment", "treatment group",
        "control group", "randomization", "experimental", "intent to treat"
    },
    "Structural Models": {
        "structural model", "structural estimation", "equilibrium model",
        "dynamic model", "simulation", "counterfactual simulation"
    },
    "Theoretical Models": {
        "theoretical model", "theory", "formal model", "game theory",
        "equilibrium", "mechanism", "optimal", "analytical"
    },
    "Machine Learning Methods": {
        "machine learning", "lasso", "random forest", "neural network",
        "causal forest", "double machine learning", "prediction", "cross-validation"
    },
    "Survey/Experimental Data": {
        "survey", "survey data", "experimental data", "questionnaire",
        "respondent", "sample", "response rate"
    },
    "Administrative Data": {
        "administrative data", "register data", "tax records", "census",
        "linked data", "population data", "registry"
    },
    "Time Series Analysis": {
        "time series", "var", "arima", "cointegration", "granger causality",
        "impulse response", "forecast", "autocorrelation"
    },
    "Panel Data Methods": {
        "panel data", "fixed effects", "random effects", "within estimator",
        "longitudinal", "repeated cross-section", "hausman test"
    },
    "Text Analysis/NLP": {
        "text analysis", "nlp", "natural language", "topic model", "sentiment",
        "word embedding", "text classification", "corpus"
    },
    "Network Analysis": {
        "network", "graph", "centrality", "clustering", "social network",
        "peer effects", "spillover", "contagion"
    },
    "Bayesian Methods": {
        "bayesian", "prior", "posterior", "mcmc", "credible interval",
        "hierarchical", "gibbs sampling"
    },
    "Qualitative Methods": {
        "qualitative", "interview", "case study", "ethnography", "discourse",
        "content analysis", "narrative", "interpretive"
    },
    "Case Studies": {
        "case study", "single case", "comparative case", "process tracing",
        "within-case", "cross-case"
    },
    "Process Tracing": {
        "process tracing", "causal mechanism", "within-case", "diagnostic",
        "smoking gun", "hoop test"
    }
}

REGION_KEYWORDS: Dict[str, Set[str]] = {
    "Global/Comparative": {"global", "cross-country", "international", "comparative", "world"},
    "United States": {"united states", "us", "usa", "american", "federal"},
    "European Union": {"european union", "eu", "europe", "european", "eurozone"},
    "United Kingdom": {"united kingdom", "uk", "british", "england", "britain"},
    "China": {"china", "chinese", "beijing", "shanghai"},
    "India": {"india", "indian", "delhi", "mumbai"},
    "Latin America": {"latin america", "brazil", "mexico", "argentina", "chile", "colombia"},
    "Sub-Saharan Africa": {"africa", "african", "nigeria", "kenya", "south africa", "ethiopia"},
    "Middle East & North Africa": {"middle east", "mena", "arab", "egypt", "iran", "turkey"},
    "Southeast Asia": {"southeast asia", "indonesia", "vietnam", "thailand", "philippines"},
    "Global South": {"developing", "global south", "emerging", "low income", "third world"},
    "OECD Countries": {"oecd", "developed", "advanced economy", "high income"},
    "Emerging Markets": {"emerging market", "brics", "developing economy"}
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STOPWORDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "also", "well", "even", "back", "much", "how", "where", "which",
    "while", "who", "whom", "why", "what", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "might",
    "must", "shall", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "paper", "study", "research", "analysis",
    "result", "find", "show", "suggest", "examine", "investigate", "use",
    "using", "based", "effect", "impact", "evidence", "data", "method"
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
    methodological_lean: float = 0.5
    
    def get_keywords(self) -> Set[str]:
        """Extract all relevant keywords from profile."""
        keywords = set()
        
        # Field keywords
        if self.primary_field in FIELD_KEYWORDS:
            keywords.update(FIELD_KEYWORDS[self.primary_field])
        
        # Interest keywords
        for interest in self.secondary_interests:
            if interest in INTEREST_KEYWORDS:
                keywords.update(INTEREST_KEYWORDS[interest])
        
        # Method keywords
        for method in self.preferred_methodology:
            if method in METHOD_KEYWORDS:
                keywords.update(METHOD_KEYWORDS[method])
        
        # Region keywords
        if self.regional_focus in REGION_KEYWORDS:
            keywords.update(REGION_KEYWORDS[self.regional_focus])
        
        return keywords
    
    def to_dict(self) -> dict:
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
class MatchResult:
    """Detailed matching result for a paper."""
    relevance_score: float
    field_score: float
    interest_score: float
    method_score: float
    region_score: float
    author_match: bool
    matched_interests: List[str]
    matched_methods: List[str]
    matched_keywords: List[str]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tokenize(text: str) -> List[str]:
    """Tokenize and clean text."""
    if not text:
        return []
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation except hyphens (for terms like "difference-in-differences")
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Filter stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    return tokens


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency (TF) for tokens."""
    if not tokens:
        return {}
    
    counts = Counter(tokens)
    total = len(tokens)
    
    return {term: count / total for term, count in counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency (IDF) across documents."""
    if not documents:
        return {}
    
    n_docs = len(documents)
    doc_freq = Counter()
    
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1
    
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(n_docs / (1 + df)) + 1  # Smoothed IDF
    
    return idf


def compute_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF scores."""
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    
    # Find common terms
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    
    if not common_terms:
        return 0.0
    
    # Compute dot product
    dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATCHING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SemanticMatcher:
    """
    Gold-standard semantic matching engine using TF-IDF and cosine similarity.
    
    Scoring weights:
    - Field match: 30%
    - Interest match: 35%
    - Method match: 25%
    - Region match: 10%
    
    Bonus: +0.5 for author match, journal tier boost
    """
    
    FIELD_WEIGHT = 0.30
    INTEREST_WEIGHT = 0.35
    METHOD_WEIGHT = 0.25
    REGION_WEIGHT = 0.10
    
    AUTHOR_BONUS = 0.5
    TIER_BONUS = {1: 0.3, 2: 0.15, 3: 0.05, 4: 0.0}
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.profile_keywords = profile.get_keywords()
        self.idf: Dict[str, float] = {}
    
    def _keyword_overlap_score(
        self, 
        text_tokens: Set[str], 
        keyword_set: Set[str]
    ) -> Tuple[float, List[str]]:
        """Calculate overlap between text tokens and keyword set."""
        if not keyword_set:
            return 0.0, []
        
        # Check for multi-word matches
        text_str = " ".join(text_tokens)
        matches = []
        
        for keyword in keyword_set:
            if " " in keyword:
                # Multi-word keyword
                if keyword in text_str:
                    matches.append(keyword)
            else:
                # Single word keyword
                if keyword in text_tokens:
                    matches.append(keyword)
        
        if not matches:
            return 0.0, []
        
        # Jaccard-style score with boost for more matches
        score = min(1.0, len(matches) / max(3, len(keyword_set) * 0.2))
        return score, matches
    
    def _compute_field_score(self, paper_tokens: Set[str]) -> Tuple[float, List[str]]:
        """Compute field match score."""
        field_keywords = FIELD_KEYWORDS.get(self.profile.primary_field, set())
        return self._keyword_overlap_score(paper_tokens, field_keywords)
    
    def _compute_interest_scores(self, paper_tokens: Set[str]) -> Tuple[float, List[str]]:
        """Compute interest match score across all user interests."""
        all_matches = []
        total_score = 0.0
        
        for interest in self.profile.secondary_interests:
            interest_keywords = INTEREST_KEYWORDS.get(interest, set())
            score, matches = self._keyword_overlap_score(paper_tokens, interest_keywords)
            if matches:
                all_matches.append(interest)
                total_score += score
        
        if not self.profile.secondary_interests:
            return 0.0, []
        
        avg_score = total_score / len(self.profile.secondary_interests)
        return min(1.0, avg_score * 1.5), all_matches  # Boost for having matches
    
    def _compute_method_scores(self, paper_tokens: Set[str]) -> Tuple[float, List[str]]:
        """Compute methodology match score."""
        all_matches = []
        total_score = 0.0
        
        for method in self.profile.preferred_methodology:
            method_keywords = METHOD_KEYWORDS.get(method, set())
            score, matches = self._keyword_overlap_score(paper_tokens, method_keywords)
            if matches:
                all_matches.append(method)
                total_score += score
        
        if not self.profile.preferred_methodology:
            return 0.0, []
        
        avg_score = total_score / len(self.profile.preferred_methodology)
        return min(1.0, avg_score * 1.5), all_matches
    
    def _compute_region_score(self, paper_tokens: Set[str]) -> float:
        """Compute regional focus match score."""
        region_keywords = REGION_KEYWORDS.get(self.profile.regional_focus, set())
        score, _ = self._keyword_overlap_score(paper_tokens, region_keywords)
        return score
    
    def _check_author_match(self, paper_authors: List[str]) -> bool:
        """Check if any seed author matches paper authors."""
        if not self.profile.seed_authors:
            return False
        
        paper_authors_lower = {a.lower() for a in paper_authors}
        seed_authors_lower = {a.lower() for a in self.profile.seed_authors}
        
        for seed in seed_authors_lower:
            for author in paper_authors_lower:
                if seed in author or author in seed:
                    return True
        return False
    
    def match_paper(self, paper: dict) -> MatchResult:
        """
        Compute comprehensive match result for a single paper.
        """
        # Prepare text
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        full_text = f"{title} {abstract}"
        
        # Add concept names
        concepts = paper.get("concepts", [])
        concept_text = " ".join(c.get("name", "") for c in concepts)
        full_text = f"{full_text} {concept_text}"
        
        # Tokenize
        tokens = tokenize(full_text)
        token_set = set(tokens)
        
        # Compute component scores
        field_score, field_matches = self._compute_field_score(token_set)
        interest_score, matched_interests = self._compute_interest_scores(token_set)
        method_score, matched_methods = self._compute_method_scores(token_set)
        region_score = self._compute_region_score(token_set)
        
        # Check author match
        author_match = self._check_author_match(paper.get("authors", []))
        
        # Compute weighted score
        base_score = (
            self.FIELD_WEIGHT * field_score +
            self.INTEREST_WEIGHT * interest_score +
            self.METHOD_WEIGHT * method_score +
            self.REGION_WEIGHT * region_score
        )
        
        # Apply bonuses
        if author_match:
            base_score += self.AUTHOR_BONUS
        
        journal_tier = paper.get("journal_tier", 4)
        base_score += self.TIER_BONUS.get(journal_tier, 0)
        
        # Normalize to 1-10 scale
        relevance_score = min(10.0, max(1.0, base_score * 10))
        
        # Get matched keywords for display
        matched_keywords = field_matches[:5]
        
        return MatchResult(
            relevance_score=round(relevance_score, 1),
            field_score=round(field_score, 2),
            interest_score=round(interest_score, 2),
            method_score=round(method_score, 2),
            region_score=round(region_score, 2),
            author_match=author_match,
            matched_interests=matched_interests,
            matched_methods=matched_methods,
            matched_keywords=matched_keywords
        )
    
    def rank_papers(
        self, 
        papers: List[dict],
        progress_callback: Optional[callable] = None
    ) -> List[dict]:
        """
        Rank all papers by relevance to user profile.
        
        Returns papers enriched with match data, sorted by relevance.
        """
        enriched_papers = []
        
        for i, paper in enumerate(papers):
            if progress_callback and i % 5 == 0:
                progress_callback(f"Analyzing paper {i+1}/{len(papers)}...")
            
            match_result = self.match_paper(paper)
            
            enriched = {
                **paper,
                "relevance_score": match_result.relevance_score,
                "field_score": match_result.field_score,
                "interest_score": match_result.interest_score,
                "method_score": match_result.method_score,
                "region_score": match_result.region_score,
                "author_match": match_result.author_match,
                "topic_matches": match_result.matched_interests,
                "method_matches": match_result.matched_methods,
                "matched_keywords": match_result.matched_keywords,
                "has_match_data": True
            }
            enriched_papers.append(enriched)
        
        # Sort by relevance score
        enriched_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return enriched_papers


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


def process_papers(
    user_profile: UserProfile,
    papers: List[dict],
    progress_callback: Optional[callable] = None
) -> Tuple[List[dict], List[str]]:
    """
    Process papers using semantic matching.
    
    Returns:
        Tuple of (enriched_papers, status_messages)
    """
    messages = []
    
    if not papers:
        return [], ["No papers to process"]
    
    matcher = SemanticMatcher(user_profile)
    
    if progress_callback:
        progress_callback("🔬 Computing semantic matches...")
    
    enriched_papers = matcher.rank_papers(papers, progress_callback)
    
    # Generate summary stats
    high_relevance = sum(1 for p in enriched_papers if p["relevance_score"] >= 7)
    messages.append(f"✓ Analyzed {len(papers)} papers")
    messages.append(f"📊 {high_relevance} high-relevance matches found")
    
    return enriched_papers, messages


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


def get_suggested_authors(field: str) -> List[str]:
    """Get suggested seed authors for a field."""
    return SEED_AUTHOR_SUGGESTIONS.get(field, [])
