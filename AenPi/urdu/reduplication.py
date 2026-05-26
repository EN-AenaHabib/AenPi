"""
aenpi/reduplication.py
----------------------
Urdu Reduplication Detector  (proposal task T10)

Problem  : Urdu and Roman Urdu are full of reduplicated word pairs that
           carry meaning ("and such", "all kinds of", emphasis, plurality):

               chai-shai   ("tea and the like")
               kitab-vitab  ("books and stuff")
               garm garm   ("piping hot")
               jaldi jaldi  ("quickly, hurriedly")

           These pairs break tokenizers, stemmers and search indexes because
           the echo word ("shai", "vitab") is not a real dictionary word.

Solution : A purely rule-based, dictionary-free detector. No training data,
           no model download, no GPU. It recognises three patterns:

           1. full     — an exact repeat            : garm garm,  jaldi jaldi
           2. echo      — onset swapped for an echo  : chai-shai, kitab-vitab
                          former (v / w / sh  in Roman,
                          و / ش  in Nastaliq)
           3. rhyming   — onset swapped for any other consonant (optional,
                          off by default to keep precision high)

           Works on Roman Urdu and Nastaliq Urdu, and on both spaced
           ("kitab vitab") and hyphenated ("kitab-vitab") forms.

Usage
-----
    from AenPi.urdu import ReduplicationDetector

    redup = ReduplicationDetector()

    redup.detect("mujhe chai-shai pila do aur garm garm roti")
    # [{'text': 'chai-shai', 'base': 'chai', 'echo': 'shai',
    #   'type': 'echo', 'start': 2, 'end': 2},
    #  {'text': 'garm garm', 'base': 'garm', 'echo': 'garm',
    #   'type': 'full', 'start': 5, 'end': 6}]

    redup.has_reduplication("kitab vitab")     # True
    redup.reduplications("kitab-vitab le aao") # ['kitab-vitab']
"""

import re


# Characters that mark the text as Nastaliq (Arabic/Urdu Unicode block).
_URDU_RANGE = re.compile(r"[؀-ۿݐ-ݿ]")

# Roman vowels — used to find a word's onset (its leading consonant cluster).
_ROMAN_VOWELS = set("aeiou")


class ReduplicationDetector:
    """
    Rule-based detector for Urdu / Roman Urdu reduplication.

    Reduplication types
    --------------------
    "full"     : the word is repeated exactly        -> garm garm
    "echo"     : the second word copies the first but its onset is replaced
                 by a fixed "echo former"            -> chai-shai, kitab-vitab
    "rhyming"  : same as echo but the new onset is any consonant, not just a
                 known former                        -> off unless include_rhyming=True

    Parameters
    ----------
    roman_formers : iterable of str
        Onset replacements that signal an echo word in Roman Urdu.
        Default: ("v", "w", "sh").
    urdu_formers : iterable of str
        First-letter replacements that signal an echo word in Nastaliq Urdu.
        Default: ("و", "ش").
    include_rhyming : bool
        If True, also report rhyming pairs whose echo onset is not a known
        former (e.g. "ulta-pulta"). Default False, because arbitrary rhymes
        ("acha bacha") produce false positives.
    min_length : int
        Minimum length (in characters) a word must have to be considered.
        Default 3 — avoids spurious matches on tiny tokens.
    """

    def __init__(self,
                 roman_formers=("v", "w", "sh"),
                 urdu_formers=("و", "ش"),
                 include_rhyming: bool = False,
                 min_length: int = 3):
        self.roman_formers   = tuple(roman_formers)
        self.urdu_formers    = tuple(urdu_formers)
        self.include_rhyming = include_rhyming
        self.min_length      = min_length

    # ------------------------------------------------------------------
    # Optional fit() — kept for API symmetry with the trainable modules
    # so the detector can drop straight into Pipeline. It does nothing:
    # the detector is rule-based and ready to use on construction.
    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs):
        """No-op. The detector is rule-based and needs no training."""
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list:
        """
        Find every reduplicated pair in the text.

        Parameters
        ----------
        text : str
            Roman Urdu or Nastaliq Urdu sentence.

        Returns
        -------
        list of dicts, each with:
            text  : str  — the matched surface form ("chai-shai", "garm garm")
            base  : str  — the real/base word
            echo  : str  — the echo (or repeated) word
            type  : str  — "full", "echo", or "rhyming"
            start : int  — index of the first token in the whitespace split
            end   : int  — index of the last token (== start for hyphen forms)

        Example
        -------
        >>> ReduplicationDetector().detect("garm garm chai-shai")
        [{'text': 'garm garm', 'base': 'garm', 'echo': 'garm',
          'type': 'full', 'start': 0, 'end': 1},
         {'text': 'chai-shai', 'base': 'chai', 'echo': 'shai',
          'type': 'echo', 'start': 2, 'end': 2}]
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        tokens  = text.split()
        results = []

        # Pass 1 — hyphenated single tokens: "chai-shai", "kitab-vitab".
        for idx, tok in enumerate(tokens):
            parts = [p for p in re.split(r"[-–—]", tok) if p]
            if len(parts) == 2:
                match = self._classify_pair(parts[0], parts[1])
                if match:
                    base, echo, rtype = match
                    results.append({
                        "text":  tok,
                        "base":  base,
                        "echo":  echo,
                        "type":  rtype,
                        "start": idx,
                        "end":   idx,
                    })

        # Pass 2 — adjacent whitespace tokens: "garm garm", "kitab vitab".
        i = 0
        while i < len(tokens) - 1:
            a, b = tokens[i], tokens[i + 1]
            # Hyphenated tokens were already handled in pass 1.
            if "-" in a or "-" in b:
                i += 1
                continue
            match = self._classify_pair(a, b)
            if match:
                base, echo, rtype = match
                results.append({
                    "text":  f"{tokens[i]} {tokens[i + 1]}",
                    "base":  base,
                    "echo":  echo,
                    "type":  rtype,
                    "start": i,
                    "end":   i + 1,
                })
                i += 2          # consume both tokens of the pair
            else:
                i += 1

        # Return in left-to-right order of appearance.
        results.sort(key=lambda r: (r["start"], r["end"]))
        return results

    def has_reduplication(self, text: str) -> bool:
        """Return True if the text contains at least one reduplication."""
        return bool(self.detect(text))

    def reduplications(self, text: str) -> list:
        """Return just the matched surface forms, e.g. ['chai-shai']."""
        return [r["text"] for r in self.detect(text)]

    def count(self, text: str) -> int:
        """Return how many reduplicated pairs the text contains."""
        return len(self.detect(text))

    # ------------------------------------------------------------------
    # Core rule engine
    # ------------------------------------------------------------------

    def _classify_pair(self, first: str, second: str):
        """
        Decide whether (first, second) is a reduplicated pair.

        Returns
        -------
        (base, echo, type) tuple if it is a reduplication, else None.
        The returned base/echo preserve the original surface spelling.
        """
        a = self._clean(first)
        b = self._clean(second)
        if len(a) < self.min_length or len(b) < self.min_length:
            return None

        # 1. Full reduplication — exact repeat.
        if a == b:
            return (first, second, "full")

        # 2 & 3. Echo / rhyming — same rime, different onset.
        if self._is_urdu(a) or self._is_urdu(b):
            onset_a, rime_a = self._split_urdu(a)
            onset_b, rime_b = self._split_urdu(b)
            formers = self.urdu_formers
        else:
            onset_a, rime_a = self._split_roman(a)
            onset_b, rime_b = self._split_roman(b)
            formers = self.roman_formers

        same_rime    = rime_a and rime_a == rime_b
        different_on = onset_a != onset_b
        if not (same_rime and different_on):
            return None

        # Identify which word is the echo (the one carrying the echo former).
        if onset_b in formers and onset_a not in formers:
            return (first, second, "echo")
        if onset_a in formers and onset_b not in formers:
            # Echo written before the base — rare, but report base correctly.
            return (second, first, "echo")

        if self.include_rhyming:
            return (first, second, "rhyming")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(word: str) -> str:
        """Lowercase and strip surrounding punctuation for comparison."""
        word = word.strip().lower()
        # Drop leading/trailing punctuation but keep inner word characters.
        return re.sub(r"^[^\w؀-ۿ]+|[^\w؀-ۿ]+$", "", word)

    @staticmethod
    def _is_urdu(word: str) -> bool:
        """True if the word contains Nastaliq (Urdu-script) characters."""
        return bool(_URDU_RANGE.search(word))

    @staticmethod
    def _split_roman(word: str):
        """
        Split a Roman word into (onset, rime).
        onset = leading consonant cluster, rime = from the first vowel on.

        >>> _split_roman("chai")   # ('ch', 'ai')
        >>> _split_roman("vitab")  # ('v', 'itab')
        >>> _split_roman("aam")    # ('', 'aam')
        """
        i = 0
        while i < len(word) and word[i] not in _ROMAN_VOWELS:
            i += 1
        return word[:i], word[i:]

    @staticmethod
    def _split_urdu(word: str):
        """
        Split a Nastaliq word into (onset, rime).
        Urdu vowels are mostly implicit, so the onset is just the first
        letter and the rime is everything after it.

        >>> _split_urdu("کتاب")   # ('ک', 'تاب')
        >>> _split_urdu("وتاب")   # ('و', 'تاب')
        """
        return word[:1], word[1:]

    def __repr__(self):
        return (f"ReduplicationDetector(roman_formers={self.roman_formers}, "
                f"urdu_formers={self.urdu_formers}, "
                f"include_rhyming={self.include_rhyming})")


# ─── Convenience function ─────────────────────────────────────────────────────

def find_reduplications(text: str, include_rhyming: bool = False) -> list:
    """
    One-shot reduplication detection without building a detector first.

    >>> find_reduplications("chai-shai aur garm garm")
    [{'text': 'chai-shai', ...}, {'text': 'garm garm', ...}]
    """
    return ReduplicationDetector(include_rhyming=include_rhyming).detect(text)
