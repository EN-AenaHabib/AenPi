"""
AenPi — Urdu-first NLP Library
Modules:
    normalizer     : Roman Urdu spelling normalizer
    code_switch    : Urdu/English token-level switch detector
    sentiment      : Urdu & Roman Urdu sentiment classifier
    ner            : Urdu Named Entity Recognition
    summarizer     : Extractive micro-summarizer
    intent_router  : Lightweight offline intent classifier
    carbon         : Carbon cost estimator vs LLM APIs
"""

from .normalizer     import UrduNormalizer
from .code_switch    import CodeSwitchDetector
from .sentiment      import UrduSentiment
from .ner            import UrduNER
from .summarizer     import UrduSummarizer
from .intent_router  import IntentRouter
from .carbon         import CarbonEstimator

__version__ = "0.1.0"
__author__  = "AenPi Contributors"

__all__ = [
    "UrduNormalizer",
    "CodeSwitchDetector",
    "UrduSentiment",
    "UrduNER",
    "UrduSummarizer",
    "IntentRouter",
    "CarbonEstimator",
]
