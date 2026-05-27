# -*- coding: utf-8 -*-
"""
AenPi.urdu.transliterator
--------------------------
Roman Urdu  Nastaliq (Urdu Arabic script) bi-directional transliterator.

Problem  : Most Urdu speakers type in Roman script ("mera naam Saqlain hai"),
           but downstream NLP models, search engines and traditional Urdu
           resources expect Nastaliq ("میرا نام ثقلین ہے").
Solution : Deterministic phoneme-based mapping using the longest-match
           greedy algorithm over multi-character Roman digraphs (kh, gh,
           sh, ch, th, ph, zh ...). Works both directions.

Author   : NuTech AI-23 | Saqlain (T8 in proposal)

Usage
-----
    from AenPi.urdu.transliterator import Transliterator

    trans = Transliterator()

    # Roman  Nastaliq
    print(trans.to_nastaliq("mera naam Saqlain hai"))
    # → 'میرا نام ثقلین ہے'

    # Nastaliq  Roman
    print(trans.to_roman("میرا نام ثقلین ہے"))
    # → 'mera naam saqlain hai'
"""

import re


class Transliterator:
    """
    Bi-directional Roman ↔ Nastaliq transliterator for Urdu.

    Two main public methods
    -----------------------
    to_nastaliq(text) : Roman Urdu  →  Urdu Arabic script
    to_roman(text)    : Urdu Arabic script  →  Roman Urdu
    """

    # ------------------------------------------------------------------
    # Roman  → Nastaliq mappings  (longest-first matching)
    # ------------------------------------------------------------------
    ROMAN_TO_NASTALIQ = [
        # ----- multi-character digraphs (MUST come first) -----
        ("chh", "چھ"),
        ("ghh", "گھ"),
        ("kh",  "خ"),
        ("gh",  "غ"),
        ("sh",  "ش"),
        ("ch",  "چ"),
        ("th",  "تھ"),
        ("ph",  "پھ"),
        ("dh",  "دھ"),
        ("zh",  "ژ"),
        ("aa",  "ا"),
        ("ee",  "ی"),
        ("oo",  "و"),
        ("ai",  "ی"),
        ("ay",  "ے"),
        ("au",  "و"),
        ("ng",  "نگ"),

        # ----- single consonants -----
        ("b", "ب"),
        ("p", "پ"),
        ("t", "ت"),
        ("j", "ج"),
        ("d", "د"),
        ("r", "ر"),
        ("z", "ز"),
        ("s", "س"),
        ("f", "ف"),
        ("q", "ق"),
        ("k", "ک"),
        ("g", "گ"),
        ("l", "ل"),
        ("m", "م"),
        ("n", "ن"),
        ("v", "و"),
        ("w", "و"),
        ("h", "ہ"),
        ("y", "ی"),
        ("x", "کس"),
        ("c", "ک"),

        # ----- vowels -----
        ("a", "ا"),
        ("e", "ے"),
        ("i", "ی"),
        ("o", "و"),
        ("u", "و"),
    ]

    # ------------------------------------------------------------------
    # Nastaliq → Roman mappings
    # ------------------------------------------------------------------
    NASTALIQ_TO_ROMAN = {
        "ا": "a",  "آ": "aa", "ب": "b",  "پ": "p",  "ت": "t",
        "ٹ": "t",  "ث": "s",  "ج": "j",  "چ": "ch", "ح": "h",
        "خ": "kh", "د": "d",  "ڈ": "d",  "ذ": "z",  "ر": "r",
        "ڑ": "r",  "ز": "z",  "ژ": "zh", "س": "s",  "ش": "sh",
        "ص": "s",  "ض": "z",  "ط": "t",  "ظ": "z",  "ع": "a",
        "غ": "gh", "ف": "f",  "ق": "q",  "ک": "k",  "گ": "g",
        "ل": "l",  "م": "m",  "ن": "n",  "ں": "n",  "و": "o",
        "ؤ": "o",  "ہ": "h",  "ھ": "h",  "ء": "",   "ی": "i",
        "ئ": "i",  "ے": "e",  "ۓ": "e",
    }

    # Common whole-word overrides (improves accuracy for high-frequency words)
    WORD_OVERRIDES_ROMAN_TO_NASTALIQ = {
        "hai":     "ہے",
        "hain":    "ہیں",
        "tha":     "تھا",
        "thi":     "تھی",
        "the":     "تھے",
        "mera":    "میرا",
        "meri":    "میری",
        "mere":    "میرے",
        "tera":    "تیرا",
        "tum":     "تم",
        "tu":      "تو",
        "hum":     "ہم",
        "aap":     "آپ",
        "main":    "میں",
        "wo":      "وہ",
        "ye":      "یہ",
        "kya":     "کیا",
        "kyu":     "کیوں",
        "kyun":    "کیوں",
        "kaise":   "کیسے",
        "kab":     "کب",
        "kahan":   "کہاں",
        "kaun":    "کون",
        "kuch":    "کچھ",
        "aur":     "اور",
        "ya":      "یا",
        "lekin":   "لیکن",
        "nahi":    "نہیں",
        "nahin":   "نہیں",
        "haan":    "ہاں",
        "ji":      "جی",
        "naam":    "نام",
        "ghar":    "گھر",
        "kaam":    "کام",
        "din":     "دن",
        "raat":    "رات",
        "saal":    "سال",
        "pani":    "پانی",
        "khana":   "کھانا",
        "achha":   "اچھا",
        "acha":    "اچھا",
        "bura":    "برا",
        "bhai":    "بھائی",
        "behan":   "بہن",
        "ammi":    "امی",
        "abbu":    "ابو",
        "saqlain": "ثقلین",
        "pakistan":"پاکستان",
        "urdu":    "اردو",
        "allah":   "اللہ",
        "salam":   "سلام",
        "shukria": "شکریہ",
    }

    WORD_OVERRIDES_NASTALIQ_TO_ROMAN = {
        v: k for k, v in WORD_OVERRIDES_ROMAN_TO_NASTALIQ.items()
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def to_nastaliq(self, text: str) -> str:
        """
        Convert Roman Urdu to Nastaliq (Urdu Arabic script).

        Parameters
        ----------
        text : str
            Roman Urdu input, any casing.

        Returns
        -------
        str
            Nastaliq output, words separated by single spaces.

        Example
        -------
        >>> Transliterator().to_nastaliq("mera naam")
        'میرا نام'
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        words = text.lower().split()
        out = []
        for w in words:
            stripped = re.sub(r"[^a-z]", "", w)
            if not stripped:
                out.append(w)
                continue
            if stripped in self.WORD_OVERRIDES_ROMAN_TO_NASTALIQ:
                out.append(self.WORD_OVERRIDES_ROMAN_TO_NASTALIQ[stripped])
            else:
                out.append(self._roman_word_to_nastaliq(stripped))
        return " ".join(out)

    def to_roman(self, text: str) -> str:
        """
        Convert Nastaliq (Urdu Arabic script) to Roman Urdu.

        Parameters
        ----------
        text : str
            Urdu Arabic-script input.

        Returns
        -------
        str
            Roman-Urdu transliteration (lowercase).

        Example
        -------
        >>> Transliterator().to_roman("میرا نام")
        'mera naam'
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        words = text.split()
        out = []
        for w in words:
            if w in self.WORD_OVERRIDES_NASTALIQ_TO_ROMAN:
                out.append(self.WORD_OVERRIDES_NASTALIQ_TO_ROMAN[w])
            else:
                out.append(self._nastaliq_word_to_roman(w))
        return " ".join(out)

    def transliterate(self, text: str, direction: str = "auto") -> str:
        """
        Auto-detect script and transliterate to the opposite.

        Parameters
        ----------
        text      : str   – input text
        direction : str   – "auto" | "to_nastaliq" | "to_roman"

        Returns
        -------
        str
        """
        if direction == "to_nastaliq":
            return self.to_nastaliq(text)
        if direction == "to_roman":
            return self.to_roman(text)

        # auto-detect: if any char in Urdu Unicode range, treat as Nastaliq
        if re.search(r"[؀-ۿ]", text):
            return self.to_roman(text)
        return self.to_nastaliq(text)

    def batch(self, texts: list, direction: str = "auto") -> list:
        """Transliterate a list of strings in one call."""
        return [self.transliterate(t, direction=direction) for t in texts]

    # ------------------------------------------------------------------
    # Internals — greedy longest-match conversion
    # ------------------------------------------------------------------
    def _roman_word_to_nastaliq(self, word: str) -> str:
        result = []
        i = 0
        n = len(word)
        while i < n:
            matched = False
            # try longest sequence first (max 3 chars)
            for length in (3, 2, 1):
                chunk = word[i:i + length]
                for src, dst in self.ROMAN_TO_NASTALIQ:
                    if len(src) == length and chunk == src:
                        result.append(dst)
                        i += length
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                # unknown character (digit / leftover) — skip
                i += 1
        return "".join(result)

    def _nastaliq_word_to_roman(self, word: str) -> str:
        result = []
        for ch in word:
            if ch in self.NASTALIQ_TO_ROMAN:
                result.append(self.NASTALIQ_TO_ROMAN[ch])
            elif ch.isspace():
                result.append(" ")
            # silently drop diacritics / unknown marks
        roman = "".join(result)
        # collapse triple-vowel artefacts (aaa → aa)
        roman = re.sub(r"(.)\1{2,}", r"\1\1", roman)
        return roman

    def __repr__(self):
        return "Transliterator(Roman ↔ Nastaliq, AenPi T8)"


# ─── Module-level convenience functions ──────────────────────────────────────

_default = Transliterator()


def to_nastaliq(text: str) -> str:
    """Convenience wrapper: Roman Urdu → Nastaliq."""
    return _default.to_nastaliq(text)


def to_roman(text: str) -> str:
    """Convenience wrapper: Nastaliq → Roman Urdu."""
    return _default.to_roman(text)


def transliterate(text: str, direction: str = "auto") -> str:
    """Convenience wrapper: auto-detect script and convert."""
    return _default.transliterate(text, direction=direction)
