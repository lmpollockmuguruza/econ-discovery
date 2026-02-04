"""
Econvery — Semantic Matching Engine
Gold-standard weighted keyword matching for academic paper discovery.
"""

import math
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTIONS
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
# WEIGHTED KEYWORD TAXONOMIES (abbreviated for brevity - full version has more)
# Format: keyword -> weight (1.0 = core, 0.7 = strong, 0.4 = related)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIELD_KEYWORDS = {
    "Microeconomics": {
        "microeconomic": 1.0, "consumer": 0.8, "demand": 0.9, "supply": 0.9,
        "equilibrium": 0.9, "utility": 1.0, "preference": 0.9, "choice": 0.8,
        "optimization": 0.8, "market": 0.6, "price": 0.7, "elasticity": 0.9,
        "welfare": 0.8, "efficiency": 0.8, "mechanism": 0.7, "incentive": 0.8,
        "game theory": 1.0, "nash": 0.8, "moral hazard": 0.9, "adverse selection": 0.9,
        "principal agent": 0.9, "asymmetric information": 0.9, "contract": 0.7
    },
    "Macroeconomics": {
        "macroeconomic": 1.0, "gdp": 1.0, "growth": 0.8, "inflation": 1.0,
        "unemployment": 0.9, "recession": 0.9, "business cycle": 1.0,
        "aggregate": 0.8, "monetary": 0.8, "fiscal": 0.8, "central bank": 0.9,
        "interest rate": 0.8, "consumption": 0.7, "investment": 0.7,
        "dsge": 1.0, "new keynesian": 1.0, "phillips curve": 0.9,
        "liquidity trap": 0.9, "quantitative easing": 0.9
    },
    "Econometrics": {
        "econometric": 1.0, "estimation": 0.9, "regression": 0.8, "identification": 1.0,
        "inference": 0.9, "estimator": 0.9, "consistent": 0.8, "unbiased": 0.8,
        "heteroskedasticity": 0.9, "specification": 0.7, "maximum likelihood": 0.9,
        "gmm": 1.0, "2sls": 1.0, "standard error": 0.8, "robust": 0.7,
        "endogeneity": 1.0, "exogeneity": 0.9
    },
    "Labor Economics": {
        "labor": 1.0, "wage": 1.0, "employment": 1.0, "unemployment": 0.9,
        "worker": 0.9, "job": 0.7, "human capital": 1.0, "education": 0.7,
        "skill": 0.9, "minimum wage": 1.0, "union": 0.9, "discrimination": 0.9,
        "earnings": 0.9, "mobility": 0.8, "monopsony": 1.0, "labor supply": 1.0,
        "labor demand": 1.0, "return to education": 1.0, "mincer": 0.9,
        "automation": 0.8, "gig economy": 0.8, "immigrant": 0.8
    },
    "Public Economics": {
        "public": 0.7, "tax": 1.0, "taxation": 1.0, "government": 0.7,
        "fiscal": 0.8, "redistribution": 1.0, "welfare": 0.8, "social insurance": 1.0,
        "public good": 1.0, "externality": 1.0, "public finance": 1.0,
        "optimal taxation": 1.0, "tax incidence": 1.0, "deadweight loss": 1.0,
        "mirrlees": 1.0, "evasion": 0.9, "compliance": 0.8
    },
    "International Economics": {
        "international": 0.7, "trade": 1.0, "export": 0.9, "import": 0.9,
        "tariff": 1.0, "globalization": 0.9, "exchange rate": 1.0,
        "fdi": 1.0, "comparative advantage": 1.0, "gravity model": 1.0,
        "heckscher ohlin": 1.0, "melitz": 0.9, "offshoring": 0.9
    },
    "Development Economics": {
        "development": 1.0, "poverty": 1.0, "developing": 0.9, "aid": 0.9,
        "microfinance": 0.9, "rural": 0.8, "agriculture": 0.8,
        "conditional cash transfer": 1.0, "randomized": 0.7, "rct": 0.7,
        "village": 0.7, "smallholder": 0.8, "property rights": 0.8
    },
    "Financial Economics": {
        "financial": 0.9, "finance": 0.9, "asset": 0.9, "stock": 0.8,
        "bond": 0.8, "return": 0.7, "risk": 0.8, "portfolio": 1.0,
        "banking": 0.9, "credit": 0.9, "mortgage": 0.9, "capm": 1.0,
        "efficient market": 1.0, "arbitrage": 0.9, "volatility": 0.9,
        "fama french": 1.0, "bubble": 0.8, "crisis": 0.8
    },
    "Industrial Organization": {
        "industrial organization": 1.0, "firm": 0.7, "market structure": 1.0,
        "competition": 0.9, "monopoly": 1.0, "oligopoly": 1.0, "antitrust": 1.0,
        "merger": 1.0, "entry": 0.8, "pricing": 0.8, "market power": 1.0,
        "markup": 0.9, "concentration": 0.9, "bertrand": 0.9, "cournot": 0.9,
        "collusion": 0.9, "platform": 0.8, "network effect": 0.9
    },
    "Behavioral Economics": {
        "behavioral": 1.0, "psychology": 0.8, "bias": 1.0, "heuristic": 1.0,
        "bounded rationality": 1.0, "prospect theory": 1.0, "loss aversion": 1.0,
        "present bias": 1.0, "hyperbolic": 0.9, "nudge": 1.0, "framing": 1.0,
        "anchoring": 1.0, "overconfidence": 0.9, "default": 0.8,
        "choice architecture": 1.0, "self control": 0.9
    },
    "Health Economics": {
        "health": 1.0, "healthcare": 1.0, "hospital": 0.9, "physician": 0.9,
        "insurance": 0.8, "medicare": 1.0, "medicaid": 1.0, "pharmaceutical": 0.9,
        "mortality": 1.0, "morbidity": 0.9, "aca": 0.9, "obamacare": 0.8
    },
    "Environmental Economics": {
        "environmental": 1.0, "climate": 1.0, "carbon": 1.0, "emission": 1.0,
        "pollution": 1.0, "energy": 0.8, "renewable": 0.9, "cap and trade": 1.0,
        "carbon tax": 1.0, "social cost of carbon": 1.0
    },
    "Urban Economics": {
        "urban": 1.0, "city": 1.0, "housing": 1.0, "rent": 0.9, "zoning": 1.0,
        "agglomeration": 1.0, "spatial": 0.9, "neighborhood": 0.9,
        "gentrification": 1.0, "segregation": 0.9, "rent control": 1.0
    },
    "Economic History": {
        "history": 0.8, "historical": 0.8, "persistence": 1.0, "colonial": 1.0,
        "industrial revolution": 1.0, "great depression": 1.0, "slavery": 0.9,
        "cliometric": 1.0, "path dependence": 0.9
    },
    "Political Economy": {
        "political economy": 1.0, "institution": 1.0, "democracy": 1.0,
        "autocracy": 1.0, "regime": 0.9, "corruption": 1.0, "state capacity": 1.0,
        "rent seeking": 1.0, "median voter": 1.0, "lobbying": 1.0
    },
    "Comparative Politics": {
        "comparative": 1.0, "regime": 1.0, "democratization": 1.0,
        "parliament": 0.9, "coalition": 0.9, "party": 0.8, "electoral system": 1.0,
        "proportional": 0.8, "federalism": 0.8
    },
    "International Relations": {
        "international relations": 1.0, "conflict": 1.0, "war": 1.0,
        "peace": 0.9, "diplomacy": 0.9, "alliance": 0.9, "sanction": 1.0,
        "nuclear": 0.9, "terrorism": 0.9, "cooperation": 0.9
    },
    "American Politics": {
        "congress": 1.0, "senate": 0.9, "president": 0.9, "supreme court": 0.9,
        "partisan": 1.0, "republican": 0.9, "democrat": 0.9, "polarization": 1.0,
        "filibuster": 0.9, "gerrymandering": 1.0
    },
    "Public Policy": {
        "public policy": 1.0, "policy evaluation": 1.0, "regulation": 0.9,
        "reform": 0.8, "effectiveness": 0.8, "cost benefit": 1.0,
        "evidence based": 0.9, "implementation": 0.8
    },
    "Political Methodology": {
        "methodology": 0.9, "causal inference": 1.0, "identification": 1.0,
        "experiment": 0.8, "measurement": 0.9, "formal model": 0.9
    }
}

INTEREST_KEYWORDS = {
    "Causal Inference": {
        "causal": 1.0, "causality": 1.0, "identification": 1.0, "endogeneity": 1.0,
        "treatment effect": 1.0, "counterfactual": 1.0, "selection": 0.7,
        "confounding": 0.9, "instrumental": 0.9, "difference in differences": 1.0,
        "potential outcome": 1.0, "average treatment effect": 1.0, "ate": 0.8,
        "late": 0.9, "exclusion restriction": 1.0, "parallel trends": 1.0,
        "event study": 0.9, "placebo": 0.9, "robustness": 0.7
    },
    "Machine Learning": {
        "machine learning": 1.0, "neural network": 1.0, "deep learning": 1.0,
        "prediction": 0.8, "random forest": 1.0, "lasso": 1.0, "regularization": 0.9,
        "cross validation": 1.0, "causal forest": 1.0, "double machine learning": 1.0
    },
    "Field Experiments": {
        "randomized": 1.0, "rct": 1.0, "randomized controlled trial": 1.0,
        "experiment": 0.8, "random assignment": 1.0, "treatment": 0.7,
        "control group": 0.9, "field experiment": 1.0, "intent to treat": 1.0,
        "compliance": 0.8, "attrition": 0.9, "spillover": 0.9
    },
    "Natural Experiments": {
        "natural experiment": 1.0, "quasi experiment": 1.0, "exogenous shock": 1.0,
        "regression discontinuity": 1.0, "instrumental variable": 1.0,
        "difference in differences": 1.0, "cutoff": 0.9, "threshold": 0.9,
        "lottery": 0.9, "policy change": 0.8
    },
    "Structural Estimation": {
        "structural": 1.0, "structural estimation": 1.0, "discrete choice": 1.0,
        "blp": 1.0, "demand estimation": 1.0, "counterfactual": 0.8,
        "simulation": 0.8, "dynamic discrete choice": 1.0
    },
    "Mechanism Design": {
        "mechanism design": 1.0, "incentive compatible": 1.0, "auction": 1.0,
        "matching": 0.9, "optimal mechanism": 1.0, "vickrey": 1.0,
        "gale shapley": 1.0, "market design": 1.0
    },
    "Policy Evaluation": {
        "policy evaluation": 1.0, "program evaluation": 1.0, "impact": 0.7,
        "effectiveness": 0.9, "cost benefit": 1.0, "welfare": 0.7,
        "external validity": 0.9, "heterogeneous effect": 0.9
    },
    "Inequality": {
        "inequality": 1.0, "redistribution": 0.9, "income distribution": 1.0,
        "wealth": 0.9, "gini": 1.0, "top income": 1.0, "mobility": 0.8,
        "intergenerational": 1.0, "piketty": 0.9, "saez": 0.9
    },
    "Climate and Energy": {
        "climate": 1.0, "climate change": 1.0, "carbon": 1.0, "emission": 1.0,
        "energy": 0.9, "renewable": 0.9, "fossil fuel": 0.9, "carbon tax": 1.0,
        "cap and trade": 1.0, "social cost of carbon": 1.0
    },
    "Education": {
        "education": 1.0, "school": 1.0, "student": 0.9, "teacher": 0.9,
        "test score": 1.0, "achievement": 0.9, "college": 0.9, "charter": 0.9,
        "voucher": 0.9, "class size": 1.0, "return to education": 1.0
    },
    "Housing": {
        "housing": 1.0, "house price": 1.0, "rent": 0.9, "mortgage": 1.0,
        "homeownership": 1.0, "zoning": 0.9, "affordability": 1.0,
        "rent control": 1.0, "eviction": 0.9
    },
    "Trade": {
        "trade": 1.0, "tariff": 1.0, "import": 0.9, "export": 0.9,
        "globalization": 1.0, "china shock": 1.0, "offshoring": 0.9,
        "trade war": 0.9, "comparative advantage": 0.9
    },
    "Monetary Policy": {
        "monetary policy": 1.0, "central bank": 1.0, "federal reserve": 1.0,
        "interest rate": 0.9, "inflation": 0.8, "quantitative easing": 1.0,
        "zero lower bound": 1.0, "forward guidance": 1.0
    },
    "Fiscal Policy": {
        "fiscal policy": 1.0, "government spending": 1.0, "stimulus": 1.0,
        "austerity": 1.0, "multiplier": 1.0, "deficit": 0.9, "debt": 0.8
    },
    "Innovation": {
        "innovation": 1.0, "patent": 1.0, "r&d": 1.0, "startup": 0.9,
        "entrepreneur": 0.9, "productivity": 0.9, "technology": 0.8,
        "creative destruction": 0.9
    },
    "Gender": {
        "gender": 1.0, "wage gap": 1.0, "women": 0.9, "discrimination": 0.9,
        "motherhood penalty": 1.0, "child penalty": 1.0, "fertility": 0.9,
        "parental leave": 0.9
    },
    "Crime and Justice": {
        "crime": 1.0, "criminal": 0.9, "police": 1.0, "prison": 1.0,
        "incarceration": 1.0, "recidivism": 1.0, "sentencing": 0.9,
        "deterrence": 0.9
    },
    "Health": {
        "health": 1.0, "mortality": 1.0, "life expectancy": 1.0,
        "disease": 0.9, "obesity": 0.9, "mental health": 0.9,
        "pandemic": 0.9, "vaccine": 0.9
    },
    "Immigration": {
        "immigration": 1.0, "immigrant": 1.0, "migration": 1.0,
        "refugee": 0.9, "border": 0.9, "undocumented": 1.0,
        "assimilation": 0.9
    },
    "Elections and Voting": {
        "election": 1.0, "voting": 1.0, "voter": 1.0, "turnout": 1.0,
        "campaign": 0.9, "polling": 0.9, "redistricting": 1.0,
        "gerrymandering": 1.0
    },
    "Conflict and Security": {
        "conflict": 1.0, "war": 1.0, "peace": 0.9, "terrorism": 1.0,
        "civil war": 1.0, "military": 0.9
    },
    "Social Mobility": {
        "mobility": 1.0, "intergenerational": 1.0, "upward mobility": 1.0,
        "opportunity": 0.9, "american dream": 0.8, "chetty": 0.8
    },
    "Poverty and Welfare": {
        "poverty": 1.0, "welfare": 1.0, "food stamps": 0.9, "snap": 0.9,
        "safety net": 1.0, "eitc": 1.0, "tanf": 0.9
    },
    "Labor Markets": {
        "labor market": 1.0, "employment": 0.9, "unemployment": 0.9,
        "hiring": 0.9, "vacancy": 0.9, "job search": 0.9, "wage": 0.8
    },
    "Taxation": {
        "tax": 1.0, "taxation": 1.0, "income tax": 1.0, "corporate tax": 1.0,
        "wealth tax": 1.0, "evasion": 0.9, "optimal taxation": 1.0
    },
    "Development": {
        "development": 1.0, "developing country": 0.9, "poverty": 0.8,
        "aid": 0.9, "microfinance": 0.9, "world bank": 0.8
    }
}

METHOD_KEYWORDS = {
    "Difference-in-Differences": {
        "difference in differences": 1.0, "diff in diff": 1.0, "did": 0.8,
        "parallel trends": 1.0, "event study": 0.9, "two way fixed effects": 1.0,
        "staggered": 1.0, "callaway santanna": 0.9, "sun abraham": 0.9,
        "goodman bacon": 0.9, "pretrend": 0.9
    },
    "Regression Discontinuity": {
        "regression discontinuity": 1.0, "rdd": 1.0, "discontinuity": 0.9,
        "cutoff": 1.0, "threshold": 0.9, "running variable": 1.0,
        "bandwidth": 1.0, "local linear": 0.9, "fuzzy": 0.9, "sharp": 0.9,
        "rdrobust": 0.9
    },
    "Instrumental Variables": {
        "instrumental variable": 1.0, "iv": 0.9, "instrument": 1.0,
        "two stage": 0.9, "2sls": 1.0, "exclusion restriction": 1.0,
        "first stage": 1.0, "weak instrument": 1.0, "late": 0.9
    },
    "Randomized Experiments": {
        "randomized": 1.0, "rct": 1.0, "random assignment": 1.0,
        "experiment": 0.8, "treatment": 0.7, "control": 0.6,
        "intent to treat": 1.0, "attrition": 0.9
    },
    "Structural Models": {
        "structural model": 1.0, "structural estimation": 1.0,
        "discrete choice": 1.0, "blp": 1.0, "demand estimation": 1.0,
        "counterfactual simulation": 1.0
    },
    "Machine Learning Methods": {
        "machine learning": 1.0, "lasso": 1.0, "random forest": 1.0,
        "causal forest": 1.0, "double machine learning": 1.0,
        "cross validation": 0.9
    },
    "Panel Data": {
        "panel data": 1.0, "fixed effects": 1.0, "random effects": 0.9,
        "within estimator": 0.9, "hausman": 1.0, "arellano bond": 1.0
    },
    "Time Series": {
        "time series": 1.0, "var": 0.9, "cointegration": 1.0,
        "granger causality": 1.0, "impulse response": 1.0
    },
    "Text Analysis": {
        "text analysis": 1.0, "nlp": 1.0, "topic model": 1.0,
        "sentiment": 0.9, "word embedding": 0.9
    },
    "Synthetic Control": {
        "synthetic control": 1.0, "donor pool": 0.9, "abadie": 1.0,
        "pre treatment fit": 0.9
    },
    "Bunching Estimation": {
        "bunching": 1.0, "kink": 0.9, "notch": 0.9, "saez": 0.9, "kleven": 0.9
    },
    "Event Studies": {
        "event study": 1.0, "abnormal return": 0.9, "announcement": 0.8
    }
}

REGION_KEYWORDS = {
    "Global": {"global": 1.0, "world": 0.8, "international": 0.7, "cross country": 0.9},
    "United States": {"united states": 1.0, "us": 0.9, "american": 0.9, "america": 0.8},
    "Europe": {"europe": 1.0, "european": 1.0, "eu": 0.9, "eurozone": 0.9},
    "United Kingdom": {"united kingdom": 1.0, "uk": 0.9, "british": 0.9, "britain": 0.9},
    "China": {"china": 1.0, "chinese": 1.0},
    "India": {"india": 1.0, "indian": 1.0},
    "Latin America": {"latin america": 1.0, "brazil": 0.9, "mexico": 0.9},
    "Africa": {"africa": 1.0, "african": 1.0, "sub saharan": 0.9},
    "Middle East": {"middle east": 1.0, "arab": 0.9},
    "Southeast Asia": {"southeast asia": 1.0, "indonesia": 0.9, "vietnam": 0.9}
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "also", "well", "even", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they",
    "paper", "study", "article", "research", "find", "show", "result"
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class UserProfile:
    name: str
    academic_level: str
    primary_field: str
    interests: List[str]
    methods: List[str]
    region: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATCHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def get_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def weighted_match(text: str, keywords: Dict[str, float]) -> Tuple[float, List[str]]:
    if not keywords:
        return 0.0, []
    
    text_lower = text.lower()
    tokens = tokenize(text)
    token_set = set(tokens)
    bigrams = set(get_ngrams(tokens, 2))
    trigrams = set(get_ngrams(tokens, 3))
    
    matched = []
    total_weight = 0.0
    max_weight = sum(keywords.values())
    
    for kw, weight in keywords.items():
        if " " in kw:
            if kw in text_lower:
                matched.append(kw)
                total_weight += weight
        else:
            if kw in token_set:
                matched.append(kw)
                total_weight += weight
    
    score = min(1.0, total_weight / (max_weight * 0.12)) if max_weight > 0 else 0.0
    return score, matched


class SemanticMatcher:
    
    FIELD_W = 0.25
    INTEREST_W = 0.35
    METHOD_W = 0.25
    REGION_W = 0.10
    
    TIER_BONUS = {1: 0.12, 2: 0.06, 3: 0.02, 4: 0.0}
    
    def __init__(self, profile: UserProfile):
        self.profile = profile
    
    def _score_field(self, text: str) -> Tuple[float, List[str]]:
        kw = FIELD_KEYWORDS.get(self.profile.primary_field, {})
        return weighted_match(text, kw)
    
    def _score_interests(self, text: str) -> Tuple[float, List[str]]:
        if not self.profile.interests:
            return 0.0, []
        
        all_matched = []
        scores = []
        
        for i, interest in enumerate(self.profile.interests):
            kw = INTEREST_KEYWORDS.get(interest, {})
            score, matches = weighted_match(text, kw)
            if matches:
                all_matched.append(interest)
            # Position weighting: first interest matters more
            weight = 1.0 - (i * 0.08)
            scores.append(score * max(0.5, weight))
        
        avg = sum(scores) / len(scores) if scores else 0.0
        
        # Boost for multiple matches
        if len(all_matched) >= 3:
            avg *= 1.25
        elif len(all_matched) >= 2:
            avg *= 1.12
        
        return min(1.0, avg), all_matched
    
    def _score_methods(self, text: str) -> Tuple[float, List[str]]:
        if not self.profile.methods:
            return 0.0, []
        
        all_matched = []
        scores = []
        
        for i, method in enumerate(self.profile.methods):
            kw = METHOD_KEYWORDS.get(method, {})
            score, matches = weighted_match(text, kw)
            if matches:
                all_matched.append(method)
            weight = 1.0 - (i * 0.12)
            scores.append(score * max(0.4, weight))
        
        avg = sum(scores) / len(scores) if scores else 0.0
        
        if len(all_matched) >= 2:
            avg *= 1.15
        
        return min(1.0, avg), all_matched
    
    def _score_region(self, text: str) -> float:
        kw = REGION_KEYWORDS.get(self.profile.region, {})
        score, _ = weighted_match(text, kw)
        return score
    
    def match_paper(self, paper: dict) -> dict:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        concepts = " ".join(c.get("name", "") for c in paper.get("concepts", []))
        
        # Title weighted more heavily
        full_text = f"{title} {title} {abstract} {concepts}"
        
        field_score, _ = self._score_field(full_text)
        interest_score, matched_interests = self._score_interests(full_text)
        method_score, matched_methods = self._score_methods(full_text)
        region_score = self._score_region(full_text)
        
        base = (
            self.FIELD_W * field_score +
            self.INTEREST_W * interest_score +
            self.METHOD_W * method_score +
            self.REGION_W * region_score
        )
        
        tier = paper.get("journal_tier", 4)
        tier_bonus = self.TIER_BONUS.get(tier, 0)
        
        cite_bonus = 0.0
        cites = paper.get("cited_by_count", 0)
        if cites >= 50:
            cite_bonus = 0.08
        elif cites >= 20:
            cite_bonus = 0.05
        elif cites >= 5:
            cite_bonus = 0.02
        
        final = base + tier_bonus + cite_bonus
        
        # Scale to 1-10 (generous)
        scaled = 1.0 + (final * 13.0)
        scaled = max(1.0, min(10.0, scaled))
        
        # Explanation
        parts = []
        if matched_interests:
            parts.append(f"{', '.join(matched_interests[:2])}")
        if matched_methods:
            parts.append(f"uses {', '.join(matched_methods[:2])}")
        if tier == 1:
            parts.append("top journal")
        elif tier == 2:
            parts.append("leading field journal")
        
        explanation = " · ".join(parts) if parts else "General relevance"
        
        return {
            **paper,
            "relevance_score": round(scaled, 1),
            "matched_interests": matched_interests,
            "matched_methods": matched_methods,
            "match_explanation": explanation
        }
    
    def rank_papers(self, papers: List[dict]) -> List[dict]:
        scored = [self.match_paper(p) for p in papers]
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored


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
    
    matcher = SemanticMatcher(profile)
    ranked = matcher.rank_papers(papers)
    
    high = sum(1 for p in ranked if p["relevance_score"] >= 7.0)
    
    return ranked, f"Analyzed {len(ranked)} papers. {high} highly relevant."


@lru_cache(maxsize=1)
def get_profile_options() -> dict:
    return {
        "academic_levels": ACADEMIC_LEVELS,
        "primary_fields": PRIMARY_FIELDS,
        "interests": RESEARCH_INTERESTS,
        "methods": METHODOLOGIES,
        "regions": REGIONS
    }
