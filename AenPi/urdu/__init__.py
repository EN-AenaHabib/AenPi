"""
AenPi.urdu - Urdu NLP Subpackage
Clean unified API for Urdu NLP tools
"""

# ================= CORE MODULES =================

from AenPi.urdu.preprocessor import preprocess, tokenize, remove_punctuation
from AenPi.urdu.stopwords import remove_stopwords, URDU_STOPWORDS
from AenPi.urdu.stemmer import UrduStemmer
from AenPi.urdu.pos_tagger import UrduPOSTagger

# IMPORTANT: class (not function)
from AenPi.urdu.normalizer import UrduNormalizer

from AenPi.urdu.spell_corrector import (
    spell_correct,
    spell_correct_text,
    edit_distance
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

# ================= PUBLIC API =================

__all__ = [
    # core
    "preprocess",
    "tokenize",
    "remove_punctuation",
    "remove_stopwords",
    "URDU_STOPWORDS",
    "UrduStemmer",
    "UrduPOSTagger",

    # IMPORTANT CLASS
    "UrduNormalizer",

    # spell
    "spell_correct",
    "spell_correct_text",
    "edit_distance",

    # ngram
    "NGramPredictor",
    "ngram_predict",

    # metrics
    "green_metrics",
    "GreenMetrics",

    # new modules
    "CodeSwitchDetector",
    "UrduSentiment",
    "UrduNER",
    "UrduSummarizer",
    "IntentRouter",
    "CarbonEstimator"
]
