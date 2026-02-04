"""
Econvery — Relevance Scoring Engine
===================================

A principled approach to matching academic papers to researcher profiles.

SCORING PHILOSOPHY
------------------
We use multiple independent signals, each interpretable on its own:

1. CONCEPT MATCH (Primary Signal)
   - OpenAlex extracts ML-classified concepts with confidence scores
   - We map user interests → OpenAlex concept names
   - This is our most reliable signal (pre-computed by ML, not keyword hacking)

2. KEYWORD MATCH (Secondary Signal)  
   - Searches title + abstract for domain-specific terms
   - Uses canonical terms + synonyms to catch variations
   - Weighted by term specificity (rare terms = stronger signal)

3. METHOD DETECTION
   - Looks for methodology-specific vocabulary
   - Methods are usually explicitly stated in abstracts
   
4. QUALITY SIGNALS
   - Journal tier (reliable proxy for quality)
   - Citation count (for papers with time to accumulate)

SCORE INTERPRETATION
--------------------
Final scores are calibrated to be meaningful:
- 8.5-10.0: Directly relevant — multiple strong matches
- 7.0-8.4:  Highly relevant — at least one strong match  
- 5.0-6.9:  Moderately relevant — partial matches
- 3.0-4.9:  Tangentially relevant — weak connections
- 1.0-2.9:  Low relevance — minimal overlap
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION OPTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACADEMIC_LEVELS = [
    "Undergraduate", "Masters Student", "PhD Student", "Postdoc",
    "Assistant Professor", "Associate Professor", "Full Professor",
    "Industry Researcher", "Policy Analyst", "Independent Researcher"
]

PRIMARY_FIELDS = [
    "Microeconomics", "Macroeconomics", "Econometrics", "Labor Economics",
    "Public Economics", "International Economics", "Development Economics",
    "Financial Economics", "Industrial Organization", "Behavioral Economics",
    "Health Economics", "Environmental Economics", "Urban Economics",
    "Economic History", "Political Economy", "Comparative Politics",
    "International Relations", "American Politics", "Public Policy",
    "Political Methodology"
]

RESEARCH_INTERESTS = [
    "Causal Inference", "Machine Learning", "Field Experiments",
    "Natural Experiments", "Structural Estimation", "Mechanism Design",
    "Policy Evaluation", "Inequality", "Climate and Energy", "Education",
    "Housing", "Trade", "Monetary Policy", "Fiscal Policy", "Innovation",
    "Gender", "Crime and Justice", "Health", "Immigration",
    "Elections and Voting", "Conflict and Security", "Social Mobility",
    "Poverty and Welfare", "Labor Markets", "Taxation", "Development"
]

METHODOLOGIES = [
    "Difference-in-Differences", "Regression Discontinuity",
    "Instrumental Variables", "Randomized Experiments", "Structural Models",
    "Machine Learning Methods", "Panel Data", "Time Series", "Text Analysis",
    "Synthetic Control", "Bunching Estimation", "Event Studies"
]

REGIONS = [
    "Global", "United States", "Europe", "United Kingdom", "China",
    "India", "Latin America", "Africa", "Middle East", "Southeast Asia"
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONCEPT MAPPING
# Maps user-facing interest names → OpenAlex concept names (case-insensitive)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTEREST_TO_CONCEPTS: Dict[str, List[str]] = {
    "Causal Inference": [
        "causal inference", "causality", "treatment effect", "instrumental variable",
        "regression discontinuity", "natural experiment", "randomized experiment",
        "econometrics", "identification"
    ],
    "Machine Learning": [
        "machine learning", "artificial intelligence", "deep learning", 
        "neural network", "prediction", "statistical learning", "data science"
    ],
    "Field Experiments": [
        "randomized controlled trial", "field experiment", "randomized experiment",
        "experimental economics", "rct", "experiment"
    ],
    "Natural Experiments": [
        "natural experiment", "quasi-experiment", "regression discontinuity",
        "instrumental variable", "difference in differences", "policy evaluation"
    ],
    "Structural Estimation": [
        "structural estimation", "structural model", "discrete choice",
        "demand estimation", "dynamic programming", "industrial organization"
    ],
    "Mechanism Design": [
        "mechanism design", "auction", "matching", "market design",
        "game theory", "information economics", "contract theory"
    ],
    "Policy Evaluation": [
        "policy evaluation", "program evaluation", "impact evaluation",
        "policy analysis", "cost-benefit analysis", "public policy"
    ],
    "Inequality": [
        "inequality", "income distribution", "wealth distribution",
        "economic inequality", "income inequality", "social mobility",
        "poverty", "redistribution", "gini"
    ],
    "Climate and Energy": [
        "climate change", "climate economics", "environmental economics",
        "energy economics", "carbon", "renewable energy", "emissions",
        "pollution", "sustainability"
    ],
    "Education": [
        "education economics", "education", "human capital", "school",
        "student achievement", "higher education", "returns to education",
        "teacher", "college"
    ],
    "Housing": [
        "housing", "real estate", "housing market", "rent", "mortgage",
        "urban economics", "housing policy", "homeownership", "zoning"
    ],
    "Trade": [
        "international trade", "trade policy", "globalization", "tariff",
        "trade agreement", "export", "import", "comparative advantage"
    ],
    "Monetary Policy": [
        "monetary policy", "central bank", "interest rate", "inflation",
        "money supply", "federal reserve", "monetary economics", "banking"
    ],
    "Fiscal Policy": [
        "fiscal policy", "government spending", "taxation", "public debt",
        "budget deficit", "stimulus", "public finance"
    ],
    "Innovation": [
        "innovation", "technological change", "patent", "r&d",
        "entrepreneurship", "productivity", "technology", "startup"
    ],
    "Gender": [
        "gender", "gender economics", "gender gap", "discrimination",
        "female labor", "wage gap", "women", "family economics"
    ],
    "Crime and Justice": [
        "crime", "criminal justice", "law enforcement", "prison",
        "incarceration", "policing", "recidivism", "law and economics"
    ],
    "Health": [
        "health economics", "healthcare", "public health", "mortality",
        "health insurance", "epidemiology", "medical", "hospital"
    ],
    "Immigration": [
        "immigration", "migration", "immigrant", "refugee",
        "labor migration", "international migration", "asylum"
    ],
    "Elections and Voting": [
        "election", "voting", "political economy", "voter turnout",
        "electoral", "democracy", "political participation", "campaign"
    ],
    "Conflict and Security": [
        "conflict", "war", "civil war", "political violence",
        "security", "peace", "military", "international relations"
    ],
    "Social Mobility": [
        "social mobility", "intergenerational mobility", "economic mobility",
        "income mobility", "opportunity", "inequality"
    ],
    "Poverty and Welfare": [
        "poverty", "welfare", "social protection", "transfer program",
        "food stamps", "social assistance", "safety net", "aid"
    ],
    "Labor Markets": [
        "labor market", "labor economics", "employment", "unemployment",
        "wage", "job search", "labor supply", "labor demand", "minimum wage"
    ],
    "Taxation": [
        "taxation", "tax policy", "income tax", "tax evasion",
        "optimal taxation", "tax incidence", "corporate tax", "public finance"
    ],
    "Development": [
        "development economics", "economic development", "poverty",
        "developing country", "foreign aid", "microfinance", "growth"
    ]
}

FIELD_TO_CONCEPTS: Dict[str, List[str]] = {
    "Microeconomics": ["microeconomics", "consumer behavior", "market", "game theory", "industrial organization", "welfare economics"],
    "Macroeconomics": ["macroeconomics", "economic growth", "business cycle", "monetary economics", "gdp", "inflation"],
    "Econometrics": ["econometrics", "statistical method", "causal inference", "estimation", "regression"],
    "Labor Economics": ["labor economics", "wage", "employment", "human capital", "labor market", "unemployment"],
    "Public Economics": ["public economics", "taxation", "public finance", "government", "welfare", "redistribution"],
    "International Economics": ["international economics", "international trade", "exchange rate", "globalization", "tariff"],
    "Development Economics": ["development economics", "poverty", "economic development", "foreign aid", "microfinance"],
    "Financial Economics": ["finance", "financial economics", "asset pricing", "banking", "stock market", "credit"],
    "Industrial Organization": ["industrial organization", "competition", "antitrust", "market structure", "monopoly"],
    "Behavioral Economics": ["behavioral economics", "psychology", "decision making", "bounded rationality", "bias"],
    "Health Economics": ["health economics", "healthcare", "health insurance", "medical", "mortality"],
    "Environmental Economics": ["environmental economics", "climate", "pollution", "energy", "carbon"],
    "Urban Economics": ["urban economics", "housing", "city", "real estate", "agglomeration", "rent"],
    "Economic History": ["economic history", "history", "historical economics", "long run"],
    "Political Economy": ["political economy", "institution", "democracy", "political economics", "voting"],
    "Comparative Politics": ["comparative politics", "regime", "democracy", "political system", "government"],
    "International Relations": ["international relations", "foreign policy", "diplomacy", "conflict", "war"],
    "American Politics": ["american politics", "congress", "election", "united states", "president"],
    "Public Policy": ["public policy", "policy analysis", "regulation", "government", "reform"],
    "Political Methodology": ["political methodology", "quantitative methods", "causal inference", "measurement"]
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KEYWORD DEFINITIONS WITH SYNONYMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class KeywordEntry:
    canonical: str
    synonyms: List[str]
    weight: float  # 0-1, higher = more diagnostic


METHOD_KEYWORDS: Dict[str, KeywordEntry] = {
    "Difference-in-Differences": KeywordEntry(
        "difference-in-differences",
        ["diff-in-diff", "did ", "difference in differences", "parallel trends", 
         "two-way fixed effects", "twfe", "staggered", "event study", "pretrend",
         "treated group", "control group", "treatment group"],
        1.0
    ),
    "Regression Discontinuity": KeywordEntry(
        "regression discontinuity",
        ["rdd", "rd design", "discontinuity", "sharp rd", "fuzzy rd",
         "running variable", "forcing variable", "cutoff", "threshold",
         "bandwidth", "local polynomial"],
        1.0
    ),
    "Instrumental Variables": KeywordEntry(
        "instrumental variable",
        ["iv ", " iv,", "instrument", "2sls", "two-stage", "tsls",
         "exclusion restriction", "first stage", "first-stage", 
         "weak instrument", "late", "local average treatment effect", "complier"],
        1.0
    ),
    "Randomized Experiments": KeywordEntry(
        "randomized",
        ["rct", "randomized controlled trial", "randomized trial", "randomised",
         "random assignment", "randomization", "field experiment", "lab experiment",
         "experimental", "treatment group", "control group", "intent to treat",
         "intention to treat", "randomly assigned", "random sample"],
        1.0
    ),
    "Structural Models": KeywordEntry(
        "structural model",
        ["structural estimation", "structural approach", "discrete choice model",
         "blp", "demand estimation", "supply estimation", "dynamic model",
         "counterfactual simulation", "estimated model", "model estimation"],
        1.0
    ),
    "Machine Learning Methods": KeywordEntry(
        "machine learning",
        ["lasso", "ridge regression", "elastic net", "random forest", 
         "gradient boosting", "neural network", "deep learning", "causal forest",
         "double ml", "cross-validation", "regularization", "prediction model",
         "xgboost", "boosted"],
        0.95
    ),
    "Panel Data": KeywordEntry(
        "panel data",
        ["fixed effects", "fixed effect", "random effects", "within estimator",
         "longitudinal", "panel regression", "individual fixed effects", 
         "time fixed effects", "entity fixed effects", "year fixed effects"],
        0.85
    ),
    "Time Series": KeywordEntry(
        "time series",
        ["var ", "vector autoregression", "arima", "cointegration",
         "granger causality", "impulse response", "forecast", "autoregressive"],
        0.85
    ),
    "Text Analysis": KeywordEntry(
        "text analysis",
        ["nlp", "natural language processing", "text mining", "topic model",
         "sentiment analysis", "word embedding", "text classification",
         "lda", "word2vec", "corpus"],
        0.95
    ),
    "Synthetic Control": KeywordEntry(
        "synthetic control",
        ["synthetic control method", "scm", "donor pool", "synthetic counterfactual",
         "abadie", "comparative case study"],
        1.0
    ),
    "Bunching Estimation": KeywordEntry(
        "bunching",
        ["bunching estimation", "bunching design", "kink", "notch",
         "excess mass", "missing mass"],
        1.0
    ),
    "Event Studies": KeywordEntry(
        "event study",
        ["event-study", "event window", "abnormal return", "announcement effect"],
        0.9
    )
}

INTEREST_KEYWORDS: Dict[str, KeywordEntry] = {
    "Causal Inference": KeywordEntry(
        "causal",
        ["causal effect", "causal inference", "causality", "causal identification",
         "identification strategy", "treatment effect", "causal impact",
         "endogeneity", "selection bias", "omitted variable", "confounding",
         "causal estimate", "causally"],
        1.0
    ),
    "Inequality": KeywordEntry(
        "inequality",
        ["income inequality", "wealth inequality", "economic inequality",
         "income distribution", "wealth distribution", "gini coefficient",
         "top 1%", "top income", "top 10%", "redistribution", "intergenerational",
         "income gap", "wage inequality", "earnings inequality"],
        1.0
    ),
    "Education": KeywordEntry(
        "education",
        ["school", "student", "teacher", "college", "university",
         "test score", "achievement gap", "graduation", "dropout",
         "returns to education", "human capital", "educational attainment",
         "classroom", "tuition", "enrollment", "academic"],
        0.9
    ),
    "Housing": KeywordEntry(
        "housing",
        ["house price", "home price", "rent", "rental", "mortgage",
         "homeownership", "housing market", "real estate", "zoning",
         "affordability", "eviction", "homelessness", "tenant", "landlord",
         "housing supply", "residential"],
        1.0
    ),
    "Health": KeywordEntry(
        "health",
        ["healthcare", "hospital", "physician", "doctor", "patient",
         "mortality", "morbidity", "life expectancy", "disease",
         "health insurance", "medicare", "medicaid", "aca", "obamacare",
         "medical", "clinical", "epidemic", "pandemic"],
        0.9
    ),
    "Immigration": KeywordEntry(
        "immigration",
        ["immigrant", "migration", "migrant", "refugee", "asylum",
         "foreign-born", "native-born", "undocumented", "visa", "border",
         "deportation", "naturalization", "citizenship"],
        1.0
    ),
    "Crime and Justice": KeywordEntry(
        "crime",
        ["criminal", "police", "policing", "prison", "incarceration",
         "recidivism", "sentencing", "arrest", "violence", "homicide",
         "theft", "robbery", "assault", "prosecution", "conviction"],
        1.0
    ),
    "Gender": KeywordEntry(
        "gender",
        ["female", "women", "woman", "male", "men", "sex difference",
         "gender gap", "wage gap", "discrimination", "motherhood",
         "child penalty", "fertility", "family", "maternity", "paternity"],
        0.9
    ),
    "Climate and Energy": KeywordEntry(
        "climate",
        ["climate change", "global warming", "carbon", "emissions",
         "greenhouse gas", "renewable", "energy", "fossil fuel",
         "carbon tax", "cap and trade", "electricity", "solar", "wind",
         "environmental", "pollution", "clean energy"],
        1.0
    ),
    "Trade": KeywordEntry(
        "trade",
        ["international trade", "tariff", "import", "export",
         "globalization", "trade policy", "trade war", "china shock",
         "offshoring", "outsourcing", "comparative advantage", "wto",
         "trade agreement", "trade liberalization"],
        1.0
    ),
    "Labor Markets": KeywordEntry(
        "labor",
        ["labour", "employment", "unemployment", "wage", "wages",
         "worker", "job", "hiring", "layoff", "minimum wage",
         "labor supply", "labor demand", "earnings", "workforce",
         "occupation", "employer", "employee"],
        0.85
    ),
    "Poverty and Welfare": KeywordEntry(
        "poverty",
        ["poor", "welfare", "social assistance", "transfer",
         "food stamp", "snap", "eitc", "safety net", "benefit",
         "low-income", "low income", "disadvantaged", "tanf"],
        1.0
    ),
    "Taxation": KeywordEntry(
        "tax",
        ["taxation", "income tax", "corporate tax", "tax rate",
         "tax evasion", "tax avoidance", "tax policy", "tax reform",
         "marginal tax", "progressive tax", "tax revenue", "taxpayer"],
        1.0
    ),
    "Monetary Policy": KeywordEntry(
        "monetary policy",
        ["central bank", "federal reserve", "fed ", "interest rate",
         "inflation", "money supply", "quantitative easing", "qe",
         "zero lower bound", "monetary transmission"],
        1.0
    ),
    "Fiscal Policy": KeywordEntry(
        "fiscal",
        ["government spending", "fiscal policy", "stimulus", "austerity",
         "deficit", "debt", "multiplier", "budget", "public spending"],
        1.0
    ),
    "Innovation": KeywordEntry(
        "innovation",
        ["patent", "r&d", "research and development", "invention",
         "entrepreneur", "startup", "technology", "productivity",
         "technological change", "creative destruction"],
        0.9
    ),
    "Development": KeywordEntry(
        "development",
        ["developing country", "developing world", "poor country",
         "foreign aid", "microfinance", "microcredit", "poverty reduction",
         "economic development", "third world", "global south"],
        0.9
    ),
    "Elections and Voting": KeywordEntry(
        "election",
        ["vote", "voting", "voter", "ballot", "electoral",
         "turnout", "campaign", "candidate", "polling", "poll",
         "democrat", "republican", "partisan"],
        1.0
    ),
    "Social Mobility": KeywordEntry(
        "mobility",
        ["intergenerational", "upward mobility", "downward mobility",
         "economic mobility", "income mobility", "opportunity",
         "social mobility", "class", "socioeconomic"],
        1.0
    ),
    "Conflict and Security": KeywordEntry(
        "conflict",
        ["war", "civil war", "violence", "military", "peace",
         "terrorism", "security", "battle", "casualty", "armed"],
        1.0
    ),
    "Field Experiments": KeywordEntry(
        "field experiment",
        ["rct", "randomized controlled", "randomized experiment",
         "randomization", "treatment arm", "control arm", "random assignment"],
        1.0
    ),
    "Natural Experiments": KeywordEntry(
        "natural experiment",
        ["quasi-experiment", "exogenous shock", "policy change",
         "reform", "plausibly exogenous", "exogenous variation"],
        1.0
    ),
    "Structural Estimation": KeywordEntry(
        "structural",
        ["structural model", "discrete choice", "demand estimation",
         "counterfactual", "structural estimation"],
        1.0
    ),
    "Mechanism Design": KeywordEntry(
        "mechanism design",
        ["auction", "matching market", "market design", "allocation mechanism"],
        1.0
    ),
    "Policy Evaluation": KeywordEntry(
        "policy evaluation",
        ["program evaluation", "impact evaluation", "effectiveness",
         "cost-benefit", "policy analysis", "intervention"],
        1.0
    ),
    "Machine Learning": KeywordEntry(
        "machine learning",
        ["ml ", "artificial intelligence", "ai ", "neural network",
         "deep learning", "prediction", "algorithm", "predictive"],
        0.95
    )
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class UserProfile:
    name: str
    academic_level: str
    primary_field: str
    interests: List[str]
    methods: List[str]
    region: str


@dataclass
class MatchScore:
    total: float
    concept_score: float
    keyword_score: float
    method_score: float
    quality_score: float
    matched_interests: List[str]
    matched_methods: List[str]
    explanation: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    text = " ".join(text.split())
    return text


def text_contains_any(text: str, terms: List[str]) -> bool:
    """Check if text contains any of the terms."""
    text_norm = normalize_text(text)
    for term in terms:
        term_norm = normalize_text(term)
        if term_norm in text_norm:
            return True
    return False


def count_keyword_matches(text: str, entry: KeywordEntry) -> Tuple[int, float]:
    """
    Count how many terms from a keyword entry appear in text.
    Returns (match_count, weighted_score).
    """
    text_norm = normalize_text(text)
    all_terms = [entry.canonical] + entry.synonyms
    
    matches = 0
    for term in all_terms:
        term_norm = normalize_text(term)
        if term_norm and term_norm in text_norm:
            matches += 1
    
    # More matches = stronger signal (but with diminishing returns)
    if matches == 0:
        return 0, 0.0
    elif matches == 1:
        score = 0.6 * entry.weight
    elif matches == 2:
        score = 0.8 * entry.weight
    else:
        score = 1.0 * entry.weight
    
    return matches, score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCORING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RelevanceScorer:
    
    TIER_SCORES = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.2}
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
        
        # Build concept targets
        self.target_concepts: Set[str] = set()
        for interest in profile.interests:
            if interest in INTEREST_TO_CONCEPTS:
                for c in INTEREST_TO_CONCEPTS[interest]:
                    self.target_concepts.add(normalize_text(c))
        
        if profile.primary_field in FIELD_TO_CONCEPTS:
            for c in FIELD_TO_CONCEPTS[profile.primary_field]:
                self.target_concepts.add(normalize_text(c))
        
        # Adaptive weights based on user emphasis
        n_interests = max(1, len(profile.interests))
        n_methods = max(1, len(profile.methods))
        
        # Base weights
        self.w_concept = 0.30
        self.w_interest = 0.30
        self.w_method = 0.25
        self.w_quality = 0.15
        
        # Adjust: more methods selected → weight methods higher
        if n_methods >= 3:
            self.w_method += 0.05
            self.w_interest -= 0.05
    
    def _score_concepts(self, paper: dict) -> Tuple[float, List[str]]:
        """Match OpenAlex concepts against user interests."""
        concepts = paper.get("concepts", [])
        if not concepts or not self.target_concepts:
            return 0.0, []
        
        matched = []
        weighted_sum = 0.0
        
        for concept in concepts:
            name = normalize_text(concept.get("name", ""))
            conf = concept.get("score", 0)
            
            for target in self.target_concepts:
                # Flexible matching: either contains the other
                if target in name or name in target:
                    matched.append(concept.get("name", ""))
                    weighted_sum += conf
                    break
        
        # Normalize: 1.5 cumulative confidence = perfect score
        score = min(1.0, weighted_sum / 1.2)
        return score, matched[:5]
    
    def _score_interest_keywords(self, text: str) -> Tuple[float, List[str]]:
        """Score interest keywords in title/abstract."""
        if not self.profile.interests:
            return 0.0, []
        
        matched = []
        scores = []
        
        for i, interest in enumerate(self.profile.interests):
            if interest not in INTEREST_KEYWORDS:
                scores.append(0.0)
                continue
            
            entry = INTEREST_KEYWORDS[interest]
            count, score = count_keyword_matches(text, entry)
            
            if count > 0:
                matched.append(interest)
            
            # Position weight: first = 1.0, second = 0.9, etc.
            pos_weight = max(0.6, 1.0 - i * 0.1)
            scores.append(score * pos_weight)
        
        if not scores:
            return 0.0, []
        
        # Take best score + bonus for multiple matches
        best = max(scores)
        avg = sum(scores) / len(scores)
        
        combined = best * 0.6 + avg * 0.4
        
        # Bonus for multiple strong matches
        if len(matched) >= 3:
            combined *= 1.25
        elif len(matched) >= 2:
            combined *= 1.1
        
        return min(1.0, combined), matched
    
    def _score_methods(self, text: str) -> Tuple[float, List[str]]:
        """Detect methodology matches."""
        if not self.profile.methods:
            return 0.0, []
        
        matched = []
        scores = []
        
        for i, method in enumerate(self.profile.methods):
            if method not in METHOD_KEYWORDS:
                scores.append(0.0)
                continue
            
            entry = METHOD_KEYWORDS[method]
            count, score = count_keyword_matches(text, entry)
            
            if count > 0:
                matched.append(method)
            
            pos_weight = max(0.5, 1.0 - i * 0.12)
            scores.append(score * pos_weight)
        
        if not scores:
            return 0.0, []
        
        best = max(scores)
        avg = sum(scores) / len(scores)
        combined = best * 0.7 + avg * 0.3
        
        if len(matched) >= 2:
            combined *= 1.15
        
        return min(1.0, combined), matched
    
    def _score_quality(self, paper: dict) -> float:
        """Score journal tier and citations."""
        tier = paper.get("journal_tier", 4)
        tier_score = self.TIER_SCORES.get(tier, 0.2)
        
        cites = paper.get("cited_by_count", 0)
        if cites >= 100:
            cite_score = 1.0
        elif cites >= 50:
            cite_score = 0.85
        elif cites >= 20:
            cite_score = 0.7
        elif cites >= 5:
            cite_score = 0.5
        else:
            cite_score = 0.3
        
        return tier_score * 0.6 + cite_score * 0.4
    
    def _build_explanation(self, matched_interests: List[str], matched_methods: List[str], paper: dict) -> str:
        parts = []
        
        if matched_interests:
            parts.append(f"{', '.join(matched_interests[:2])}")
        
        if matched_methods:
            methods_str = matched_methods[0] if len(matched_methods) == 1 else f"{matched_methods[0]} + more"
            parts.append(f"Uses {methods_str}")
        
        tier = paper.get("journal_tier", 4)
        if tier == 1:
            parts.append("Top journal")
        elif tier == 2:
            parts.append("Top field journal")
        
        return " · ".join(parts) if parts else "Related to your field"
    
    def score_paper(self, paper: dict) -> MatchScore:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {title} {title} {abstract}"  # Title 3x weight
        
        concept_score, matched_concepts = self._score_concepts(paper)
        keyword_score, matched_interests = self._score_interest_keywords(text)
        method_score, matched_methods = self._score_methods(text)
        quality_score = self._score_quality(paper)
        
        # Combine interest signals: best of concept OR keyword
        interest_combined = max(concept_score, keyword_score)
        # But if both are good, give a bonus
        if concept_score > 0.3 and keyword_score > 0.3:
            interest_combined = min(1.0, interest_combined * 1.15)
        
        # Weighted combination
        raw = (
            self.w_concept * concept_score +
            self.w_interest * keyword_score +
            self.w_method * method_score +
            self.w_quality * quality_score
        )
        
        # CALIBRATION to 1-10 scale
        # Designed so that:
        #   raw >= 0.45 → 8+ (excellent)
        #   raw >= 0.30 → 6.5+ (very good)
        #   raw >= 0.18 → 5+ (good)
        #   raw >= 0.10 → 3.5+ (some relevance)
        
        if raw >= 0.45:
            final = 8.0 + (raw - 0.45) / 0.55 * 2.0
        elif raw >= 0.30:
            final = 6.5 + (raw - 0.30) / 0.15 * 1.5
        elif raw >= 0.18:
            final = 5.0 + (raw - 0.18) / 0.12 * 1.5
        elif raw >= 0.10:
            final = 3.5 + (raw - 0.10) / 0.08 * 1.5
        else:
            final = 1.0 + raw / 0.10 * 2.5
        
        final = max(1.0, min(10.0, final))
        
        explanation = self._build_explanation(matched_interests, matched_methods, paper)
        
        return MatchScore(
            total=round(final, 1),
            concept_score=round(concept_score, 3),
            keyword_score=round(keyword_score, 3),
            method_score=round(method_score, 3),
            quality_score=round(quality_score, 3),
            matched_interests=matched_interests,
            matched_methods=matched_methods,
            explanation=explanation
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_profile(
    name: str,
    academic_level: str,
    primary_field: str,
    interests: List[str],
    methods: List[str],
    region: str
) -> UserProfile:
    return UserProfile(name, academic_level, primary_field, interests, methods, region)


def process_papers(profile: UserProfile, papers: List[dict]) -> Tuple[List[dict], str]:
    if not papers:
        return [], "No papers found."
    
    scorer = RelevanceScorer(profile)
    
    results = []
    for paper in papers:
        match = scorer.score_paper(paper)
        
        enriched = {
            **paper,
            "relevance_score": match.total,
            "matched_interests": match.matched_interests,
            "matched_methods": match.matched_methods,
            "match_explanation": match.explanation
        }
        results.append(enriched)
    
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    high = sum(1 for p in results if p["relevance_score"] >= 7.0)
    summary = f"Analyzed {len(papers)} papers · {high} highly relevant"
    
    return results, summary


@lru_cache(maxsize=1)
def get_profile_options() -> dict:
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "interests": RESEARCH_INTERESTS,
        "methods": METHODOLOGIES,
        "regions": REGIONS
    }
