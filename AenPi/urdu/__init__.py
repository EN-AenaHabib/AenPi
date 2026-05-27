"""
AenPi — Urdu-first NLP Library
"""

from .normalizer import UrduNormalizer
from .code_switch import CodeSwitchDetector
from .sentiment import UrduSentiment
from .ner import UrduNER
from .summarizer import UrduSummarizer
from .intent_router import IntentRouter
from .carbon import CarbonEstimator

from .spell_corrector import (
    spell_correct,
    spell_correct_text
)

from .preprocessor import preprocess
from .ngram import *
from .stopwords import *

from .green_metrics import *

__version__ = "0.1.0"
__author__ = "AenPi Contributors"

__all__ = [
    "UrduNormalizer",
    "CodeSwitchDetector",
    "UrduSentiment",
    "UrduNER",
    "UrduSummarizer",
    "IntentRouter",
    "CarbonEstimator",
    "spell_correct",
    "spell_correct_text",
    "preprocess"
]
