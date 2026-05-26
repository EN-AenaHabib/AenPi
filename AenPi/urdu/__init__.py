"""
AenPi — Urdu-first NLP Library

Trainable models
    UrduNormalizer       : Roman Urdu spelling normalizer
    CodeSwitchDetector   : Urdu/English token-level switch detector
    UrduSentiment        : Urdu & Roman Urdu sentiment classifier
    UrduNER              : Urdu Named Entity Recognition
    UrduSummarizer       : Extractive micro-summarizer
    IntentRouter         : Lightweight offline intent classifier
    CarbonEstimator      : Carbon cost estimator vs LLM APIs
    ReduplicationDetector: Rule-based echo/full reduplication detector (T10)

Unified entry point
    Pipeline             : chain any combination of modules in one call
    Doc                  : the annotated result object a Pipeline returns

Function utilities
    preprocess, tokenize, remove_punctuation, remove_stopwords,
    get_stopwords, spell_correct, spell_correct_text, edit_distance,
    ngram_predict, NGramPredictor, GreenMetrics, green_metrics
"""

# ── Trainable / class-based modules ───────────────────────────────────────────
from .normalizer      import UrduNormalizer
from .code_switch     import CodeSwitchDetector
from .sentiment       import UrduSentiment
from .ner             import UrduNER
from .summarizer      import UrduSummarizer
from .intent_router   import IntentRouter
from .carbon          import CarbonEstimator
from .reduplication   import ReduplicationDetector, find_reduplications

# ── Unified entry point ───────────────────────────────────────────────────────
from .pipeline        import Pipeline, Doc

# ── Function-based utilities ──────────────────────────────────────────────────
from .preprocessor    import (
    preprocess, tokenize, remove_punctuation,
    remove_diacritics, remove_non_urdu, normalize_whitespace,
)
from .stopwords       import remove_stopwords, get_stopwords, URDU_STOPWORDS
from .spell_corrector import (
    edit_distance, spell_correct, spell_correct_text, load_vocabulary,
)
from .ngram           import NGramPredictor, ngram_predict
from .green_metrics   import GreenMetrics, green_metrics

__version__ = "0.2.0"
__author__  = "AenPi Contributors"

__all__ = [
    # class-based modules
    "UrduNormalizer",
    "CodeSwitchDetector",
    "UrduSentiment",
    "UrduNER",
    "UrduSummarizer",
    "IntentRouter",
    "CarbonEstimator",
    "ReduplicationDetector",
    "find_reduplications",
    # unified entry point
    "Pipeline",
    "Doc",
    # preprocessing
    "preprocess",
    "tokenize",
    "remove_punctuation",
    "remove_diacritics",
    "remove_non_urdu",
    "normalize_whitespace",
    # stopwords
    "remove_stopwords",
    "get_stopwords",
    "URDU_STOPWORDS",
    # spell correction
    "edit_distance",
    "spell_correct",
    "spell_correct_text",
    "load_vocabulary",
    # n-gram
    "NGramPredictor",
    "ngram_predict",
    # green metrics
    "GreenMetrics",
    "green_metrics",
]
