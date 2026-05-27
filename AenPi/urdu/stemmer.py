"""
stemmer.py — Urdu Rule-Based Stemmer / Lemmatizer
==================================================
T4 deliverable: stemmer.stem(word) → root form
Example: stemmer.stem("کھانے") → "کھا"

Approach:
  1. Suffix-stripping rules (longest match first)
  2. Verb conjugation normalization
  3. Noun/adjective inflection removal
  4. Validation against a known-roots list (optional)

Author : NuTech AI-23 | F23607023
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Suffix rules  (longest suffix first inside each group)
# Format: (suffix_to_strip, minimum_stem_length)
# ---------------------------------------------------------------------------

VERB_SUFFIXES = [
    # past/future/subjunctive conjugations
    ("ئیں گی", 2), ("ئیں گے", 2), ("یں گی", 2), ("یں گے", 2),
    ("ئیں گا", 2), ("یں گا", 2),
    ("ائیں", 2),  ("ئیں", 2),
    ("ئے گا", 2), ("ئے گی", 2), ("ئے گے", 2),
    ("تیں", 2),  ("تے", 2),  ("تا", 2),  ("تی", 2),
    ("ئیں", 2), ("ئے", 2),  ("ئی", 2),  ("ئا", 2),
    ("نا", 2),   ("نے", 2),  ("نی", 2),
    ("کر", 2),   ("کے", 2),
    ("یں", 2),   ("یا", 2),  ("یے", 2),
    ("اؤ", 2),   ("اؤں", 2),
    ("ا", 2),    ("ی", 2),   ("ے", 2),
]

NOUN_SUFFIXES = [
    ("وں", 2),   ("اں", 2),  ("یاں", 2),
    ("وں", 2),   ("ات", 2),
    ("گاہ", 3),  ("دان", 3), ("خانہ", 3),
    ("ی", 2),    ("ہ", 2),
]

ADJECTIVE_SUFFIXES = [
    ("ترین", 2), ("تر", 2),
    ("انہ", 2),  ("انہ", 2),
    ("ی", 2),    ("ہ", 2),
]

# Combined ordered list: try verb rules first, then noun, then adj
ALL_SUFFIX_RULES = VERB_SUFFIXES + NOUN_SUFFIXES + ADJECTIVE_SUFFIXES

# ---------------------------------------------------------------------------
# Small lexicon of known irregular roots  {surface_form: root}
# ---------------------------------------------------------------------------
IRREGULAR = {
    "ہے":   "ہو",
    "ہیں":  "ہو",
    "ہوں":  "ہو",
    "تھا":  "ہو",
    "تھی":  "ہو",
    "تھے":  "ہو",
    "گیا":  "جا",
    "گئی":  "جا",
    "گئے":  "جا",
    "آیا":  "آ",
    "آئی":  "آ",
    "آئے":  "آ",
    "دیا":  "دے",
    "دی":   "دے",
    "دیے":  "دے",
    "لیا":  "لے",
    "لی":   "لے",
    "لیے":  "لے",
    "کیا":  "کر",
    "کی":   "کر",
    "کیے":  "کر",
    "کھانے":"کھا",
    "کھایا":"کھا",
    "کھائی":"کھا",
    "کھائے":"کھا",
    "پیا":  "پی",
    "پئے":  "پی",
    "سونا": "سو",
    "سوئے": "سو",
    "بولنا":"بول",
    "پڑھنا":"پڑھ",
    "لکھنا":"لکھ",
    "چلنا": "چل",
    "دوڑنا":"دوڑ",
    "ہنسنا":"ہنس",
    "رونا": "رو",
    "بھاگنا":"بھاگ",
    "جاگنا":"جاگ",
    "ناچنا":"ناچ",
    "گانا": "گا",
    "سننا": "سن",
    "دیکھنا":"دیکھ",
    "سوچنا":"سوچ",
    "سمجھنا":"سمجھ",
    "بتانا":"بتا",
    "بنانا":"بنا",
    "اٹھنا":"اٹھ",
    "بیٹھنا":"بیٹھ",
    "کھلنا":"کھل",
    "کھلانا":"کھلا",
    "لڑنا": "لڑ",
    "مارنا":"مار",
    "بچنا": "بچ",
    "بچانا":"بچا",
}


class UrduStemmer:
    """
    Rule-based stemmer / lemmatizer for Urdu text.

    Usage
    -----
    >>> stemmer = UrduStemmer()
    >>> stemmer.stem("کھانے")
    'کھا'
    >>> stemmer.stem("کتابوں")
    'کتاب'
    """

    def __init__(self, min_stem_len: int = 2):
        self.min_stem_len = min_stem_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stem(self, word: str) -> str:
        """Return the stem / lemma of *word*."""
        word = word.strip()
        if not word:
            return word

        # 1. Irregular / lexicon lookup (highest priority)
        if word in IRREGULAR:
            return IRREGULAR[word]

        # 2. Suffix stripping
        result = self._strip_suffixes(word)
        return result if result else word

    def stem_sentence(self, sentence: str) -> list[str]:
        """Stem every whitespace-separated token in *sentence*."""
        return [self.stem(tok) for tok in sentence.split()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_suffixes(self, word: str) -> Optional[str]:
        """Try each suffix rule and return the first valid stem."""
        for suffix, min_len in ALL_SUFFIX_RULES:
            if word.endswith(suffix):
                candidate = word[: len(word) - len(suffix)]
                if len(candidate) >= max(self.min_stem_len, min_len):
                    return candidate
        return word  # no suffix matched → return as-is


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
def run_tests():
    stemmer = UrduStemmer()

    test_cases = [
        # (input,         expected_output)
        ("کھانے",        "کھا"),
        ("کتابوں",       "کتاب"),
        ("لکھنا",        "لکھ"),
        ("لڑکیاں",       "لڑک"),
        ("سوچتے",        "سوچ"),
        ("دیکھنا",       "دیکھ"),
        ("پڑھتی",        "پڑھ"),
        ("بولتا",        "بول"),
        ("گیا",          "جا"),
        ("ہیں",          "ہو"),
        ("کیا",          "کر"),
        ("استادوں",      "استاد"),
        ("بچانا",        "بچا"),
        ("ناچنا",        "ناچ"),
        ("پینے",         "پ"),     # edge case — very short stem
    ]

    print("=" * 55)
    print(f"{'Input':<20} {'Expected':<15} {'Got':<15} {'✓/✗'}")
    print("=" * 55)
    passed = 0
    for word, expected in test_cases:
        got = stemmer.stem(word)
        ok  = "✓" if got == expected else "✗"
        if got == expected:
            passed += 1
        print(f"{word:<20} {expected:<15} {got:<15} {ok}")

    print("=" * 55)
    print(f"Passed: {passed}/{len(test_cases)}")
    return passed, len(test_cases)


if __name__ == "__main__":
    run_tests()
