"""
AenPi.urdu - Urdu NLP Subpackage
Clean unified API for Urdu NLP tools
"""

# ================= CORE MODULES =================

from .preprocessor import preprocess, tokenize, remove_punctuation
from .stopwords import remove_stopwords, URDU_STOPWORDS , get_stopwords
from .stemmer import UrduStemmer
from .pos_tagger import UrduPOSTagger
from .normalizer import UrduNormalizer

from .spell_corrector import (
    spell_correct,
    spell_correct_text,
    edit_distance,
)

from .ngram import (
    NGramPredictor,
    ngram_predict,
)


# ================= NEW MODULES =================

from .code_switch import CodeSwitchDetector
from .sentiment import UrduSentiment
from .ner import UrduNER
from .summarizer import UrduSummarizer
from .intent_router import IntentRouter

# ================= EXTENDED MODULES =================

from .transliterator import (
    Transliterator,
    to_nastaliq,
    to_roman,
    transliterate,
)

from .reduplication import (
    ReduplicationDetector,
    find_reduplications,
)

from .textstats import (
    FreqDist,
    TextStats,
    freq_dist,
    concordance,
    collocations,
    lexical_diversity,
)

from .pipeline import (
    Pipeline,
    Doc,
    AVAILABLE_STAGES,
)



# ================= PUBLIC API =================

__all__ = [

    # Preprocessor
    "preprocess",
    "tokenize",
    "remove_punctuation",

    # Stopwords
    "remove_stopwords",
    "URDU_STOPWORDS",

    # Stemmer
    "UrduStemmer",

    # POS Tagger
    "UrduPOSTagger",

    # Normalizer
    "UrduNormalizer",

    # Spell Corrector
    "spell_correct",
    "spell_correct_text",
    "edit_distance",

    # N-Gram
    "NGramPredictor",
    "ngram_predict",

    # Code Switch
    "CodeSwitchDetector",

    # Sentiment
    "UrduSentiment",

    # NER
    "UrduNER",

    # Summarizer
    "UrduSummarizer",

    # Intent Router
    "IntentRouter",

    # Transliterator
    "Transliterator",
    "to_nastaliq",
    "to_roman",
    "transliterate",

    # Reduplication
    "ReduplicationDetector",
    "find_reduplications",

    # Text Statistics
    "FreqDist",
    "TextStats",
    "freq_dist",
    "concordance",
    "collocations",
    "lexical_diversity",

    # Pipeline
    "Pipeline",
    "Doc",
    "AVAILABLE_STAGES",
]



# ── auto-load saved models ────────────────────────────────────────────────────
import joblib
from pathlib import Path

_MODELS_DIR     = Path(__file__).parent / "models"
_router = None
_model  = None

try:
    _d      = joblib.load(_MODELS_DIR / "intent_router.joblib")
    _router = IntentRouter()
    _router.vectorizer = _d["vectorizer"]
    _router.clf        = _d["clf"]
    _router.classes_   = _d["classes_"]
    _router.is_fitted  = _d["is_fitted"]
    _router._examples  = _d["_examples"]
except Exception:
    pass

try:
    _d     = joblib.load(_MODELS_DIR / "urdu_sentiment.joblib")
    _model = UrduSentiment()
    _model.vectorizer = _d["vectorizer"]
    _model.clf        = _d["clf"]
    _model.classes_   = _d["classes_"]
    _model.is_fitted  = _d["is_fitted"]
except Exception:
    pass

# ── shortcut functions ────────────────────────────────────────────────────────

def sentiment(text: str) -> dict:
    if _model is None:
        raise RuntimeError("urdu_sentiment.joblib not found in AenPi/urdu/models/")
    return _model.predict(text)

def sentiment_batch(texts: list) -> list:
    if _model is None:
        raise RuntimeError("urdu_sentiment.joblib not found in AenPi/urdu/models/")
    return _model.predict_batch(texts)

def intent(text: str) -> dict:
    if _router is None:
        raise RuntimeError("intent_router.joblib not found in AenPi/urdu/models/")
    return _router.predict(text)

def intent_batch(texts: list) -> list:
    if _router is None:
        raise RuntimeError("intent_router.joblib not found in AenPi/urdu/models/")
    return _router.predict_batch(texts)

def intent_top(text: str, n: int = 3) -> list:
    if _router is None:
        raise RuntimeError("intent_router.joblib not found in AenPi/urdu/models/")
    return _router.top_n(text, n=n)
