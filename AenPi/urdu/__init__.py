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
from .ner import UrduNER , entities
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
