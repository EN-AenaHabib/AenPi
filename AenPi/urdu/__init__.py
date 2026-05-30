"""
AenPi.urdu - Urdu NLP Subpackage
Clean unified API for Urdu NLP tools
"""

# ================= CORE MODULES =================
from AenPi.urdu.preprocessor import preprocess, tokenize, remove_punctuation
from AenPi.urdu.stopwords import remove_stopwords, URDU_STOPWORDS
from AenPi.urdu.stemmer import UrduStemmer
from AenPi.urdu.pos_tagger import UrduPOSTagger
from AenPi.urdu.normalizer import UrduNormalizer          # class, not function
from AenPi.urdu.spell_corrector import (
    spell_correct,
    spell_correct_text,
    edit_distance,
)
from AenPi.urdu.ngram import NGramPredictor, ngram_predict
from AenPi.urdu.green_metrics import green_metrics, GreenMetrics

# ================= NEW MODULES =================
from AenPi.urdu.code_switch import CodeSwitchDetector
from AenPi.urdu.sentiment import UrduSentiment
from AenPi.urdu.ner import UrduNER
from AenPi.urdu.summarizer import UrduSummarizer
from AenPi.urdu.intent_router import IntentRouter
from AenPi.urdu.carbon import CarbonEstimator

# ================= EXTENDED MODULES =================
from AenPi.urdu.transliterator import (
    Transliterator,
    to_nastaliq,
    to_roman,
    transliterate,
)
from AenPi.urdu.textstats import (
    TextStats,
    text_stats,
    readability_score,
)
from AenPi.urdu.pipeline import (
    UrduPipeline,
    pipeline,
)

# ================= PUBLIC API =================
__all__ = [
    # --- preprocessor ---
    "preprocess",
    "tokenize",
    "remove_punctuation",

    # --- stopwords ---
    "remove_stopwords",
    "URDU_STOPWORDS",

    # --- stemmer ---
    "UrduStemmer",

    # --- POS tagger ---
    "UrduPOSTagger",

    # --- normalizer (class) ---
    "UrduNormalizer",

    # --- spell corrector ---
    "spell_correct",
    "spell_correct_text",
    "edit_distance",

    # --- n-gram ---
    "NGramPredictor",
    "ngram_predict",

    # --- green metrics ---
    "green_metrics",
    "GreenMetrics",

    # --- code-switching ---
    "CodeSwitchDetector",

    # --- sentiment ---
    "UrduSentiment",

    # --- NER ---
    "UrduNER",

    # --- summarizer ---
    "UrduSummarizer",

    # --- intent router ---
    "IntentRouter",

    # --- carbon estimator ---
    "CarbonEstimator",

    # --- transliterator ---
    "Transliterator",
    "to_nastaliq",
    "to_roman",
    "transliterate",

    # --- reduplication ---
    "detect_reduplication",
    "normalize_reduplication",

    # --- text stats ---
    "TextStats",
    "text_stats",
    "readability_score",

    # --- pipeline ---
    "UrduPipeline",
    "pipeline",
]
