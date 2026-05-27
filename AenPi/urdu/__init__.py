"""
AenPi.urdu - Urdu NLP Subpackage
Exposes all Urdu NLP modules directly.

Usage:
    from AenPi import urdu
    urdu.preprocess("میرا نام احمد ہے")
    urdu.remove_stopwords("یہ ایک اچھا دن ہے")
    urdu.normalize("ﻣﯿﺮﺍ ﻧﺎﻡ")
    urdu.spell_correct("احمض", ["احمد", "محمد"])
    urdu.ngram_predict("میرا نام", n=2)
    urdu.green_metrics(my_func, args)
"""

# ===== OLD CORE IMPORTS =====
from AenPi.urdu.preprocessor import preprocess, tokenize, remove_punctuation
from AenPi.urdu.stopwords import remove_stopwords, URDU_STOPWORDS

# class renamed as normalizer (important)
from AenPi.urdu.normalizer import UrduNormalizer as normalizer

from AenPi.urdu.spell_corrector import spell_correct, spell_correct_text, edit_distance
from AenPi.urdu.ngram import NGramPredictor, ngram_predict
from AenPi.urdu.green_metrics import green_metrics, GreenMetrics

# ===== NEW MODULES =====
from AenPi.urdu.code_switch import CodeSwitchDetector
from AenPi.urdu.sentiment import UrduSentiment
from AenPi.urdu.ner import UrduNER
from AenPi.urdu.summarizer import UrduSummarizer
from AenPi.urdu.intent_router import IntentRouter
from AenPi.urdu.carbon import CarbonEstimator

# ===== PUBLIC API =====
__all__ = [
    # old
    "preprocess",
    "tokenize",
    "remove_punctuation",
    "remove_stopwords",
    "URDU_STOPWORDS",

    # IMPORTANT: this must match alias name
    "normalizer",

    "spell_correct",
    "spell_correct_text",
    "edit_distance",
    "NGramPredictor",
    "ngram_predict",
    "green_metrics",
    "GreenMetrics",

    # new
    "CodeSwitchDetector",
    "UrduSentiment",
    "UrduNER",
    "UrduSummarizer",
    "IntentRouter",
    "CarbonEstimator",
]
