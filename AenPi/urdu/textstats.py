# -*- coding: utf-8 -*-
"""
AenPi.urdu.textstats
---------------------
NLTK-style text statistics utilities for Urdu corpora.

Provides
--------
    FreqDist        : frequency distribution over tokens
    TextStats       : concordance, collocations, dispersion, hapaxes
    freq_dist()     : module-level convenience for FreqDist
    concordance()   : module-level concordance one-shot
    collocations()  : module-level top-bigram one-shot

Author : NuTech AI-23 | Saqlain (T11 in proposal)

Quick start
-----------
    from AenPi.urdu.textstats import TextStats, FreqDist

    text = "میرا نام احمد ہے میرا گھر لاہور میں ہے"

    stats = TextStats(text)

    stats.freq_dist().most_common(3)
    # → [('میرا', 2), ('ہے', 2), ('نام', 1)]

    stats.concordance("میرا", width=30)
    # → ['میرا نام احمد ہے میرا گھر ...', ' ... ہے میرا گھر لاہور میں ہے']

    stats.collocations(top_n=5)
    # → [('میرا', 'نام'), ('نام', 'احمد'), ...]
"""

import math
import re
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
#  FreqDist  —  thin wrapper around collections.Counter (NLTK-compatible API)
# ──────────────────────────────────────────────────────────────────────────────
class FreqDist(Counter):
    """
    Frequency distribution of tokens.

    Mirrors the most-used parts of NLTK's ``nltk.FreqDist`` API so existing
    teaching material works without modification.

    Example
    -------
    >>> fd = FreqDist(["میرا", "نام", "میرا"])
    >>> fd.freq("میرا")
    0.6666666666666666
    >>> fd.most_common(1)
    [('میرا', 2)]
    """

    def N(self) -> int:
        """Total number of token occurrences (with repetition)."""
        return sum(self.values())

    def B(self) -> int:
        """Number of distinct tokens (bin count)."""
        return len(self)

    def freq(self, token) -> float:
        """Relative frequency of *token* (0.0 if unseen)."""
        n = self.N()
        return self[token] / n if n else 0.0

    def hapaxes(self) -> list:
        """Tokens occurring exactly once."""
        return [tok for tok, c in self.items() if c == 1]

    def max(self):
        """Single most common token, or None if empty."""
        return self.most_common(1)[0][0] if self else None

    def tabulate(self, top_n: int = 10) -> None:
        """Pretty-print the top-N tokens as a 2-column table."""
        for tok, c in self.most_common(top_n):
            print(f"{tok:<20} {c}")

    def __repr__(self):
        return f"FreqDist({self.B()} types, {self.N()} tokens)"


# ──────────────────────────────────────────────────────────────────────────────
#  TextStats  —  concordance, collocations, dispersion, type-token ratio …
# ──────────────────────────────────────────────────────────────────────────────
class TextStats:
    """
    NLTK-style text statistics for Urdu (works for any whitespace-tokenized
    language really).

    Parameters
    ----------
    text : str | list[str]
        Either raw whitespace-separated text or a pre-tokenized list.
    """

    def __init__(self, text):
        if isinstance(text, str):
            self.tokens = text.split()
        elif isinstance(text, (list, tuple)):
            self.tokens = list(text)
        else:
            raise TypeError(
                f"text must be str or list, got {type(text).__name__}"
            )
        self._freq = None  # lazy-built FreqDist

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------
    def freq_dist(self) -> FreqDist:
        """Return a cached FreqDist over the tokens."""
        if self._freq is None:
            self._freq = FreqDist(self.tokens)
        return self._freq

    def most_common(self, top_n: int = 10) -> list:
        """Top-N tokens as (token, count) tuples."""
        return self.freq_dist().most_common(top_n)

    def hapaxes(self) -> list:
        """Tokens occurring exactly once."""
        return self.freq_dist().hapaxes()

    # ------------------------------------------------------------------
    # Concordance  (KWIC — Key Word In Context)
    # ------------------------------------------------------------------
    def concordance(self, word: str, width: int = 40, lines: int = 25) -> list:
        """
        Return up-to *lines* concordance lines for *word*.

        A concordance line is a snippet of text showing the target word in
        the middle, surrounded by ``width`` characters of context on each
        side.

        Returns
        -------
        list[str]
        """
        joined = " ".join(self.tokens)
        results = []
        # iterate over every occurrence
        for m in re.finditer(rf"(?<!\S){re.escape(word)}(?!\S)", joined):
            start = max(0, m.start() - width)
            end = min(len(joined), m.end() + width)
            snippet = joined[start:end].replace("\n", " ").strip()
            results.append(snippet)
            if len(results) >= lines:
                break
        return results

    # ------------------------------------------------------------------
    # Collocations  —  most informative bigrams (PMI ranked)
    # ------------------------------------------------------------------
    def collocations(
        self,
        top_n: int = 20,
        min_count: int = 2,
        window: int = 1,
    ) -> list:
        """
        Return top-N collocations ranked by Pointwise Mutual Information.

        Parameters
        ----------
        top_n     : int   – how many bigrams to return
        min_count : int   – ignore bigrams seen fewer times
        window    : int   – distance considered "adjacent" (default 1)

        Returns
        -------
        list[tuple[str, str]]
        """
        # token frequencies
        unigrams = self.freq_dist()
        total = unigrams.N()
        if total == 0:
            return []

        # bigram frequencies (within window)
        bigrams = Counter()
        toks = self.tokens
        for i, w in enumerate(toks):
            for j in range(i + 1, min(i + 1 + window, len(toks))):
                bigrams[(w, toks[j])] += 1

        # PMI = log( p(x,y) / (p(x) * p(y)) )
        scored = []
        for (a, b), c in bigrams.items():
            if c < min_count:
                continue
            p_ab = c / total
            p_a = unigrams[a] / total
            p_b = unigrams[b] / total
            if p_a == 0 or p_b == 0:
                continue
            pmi = math.log(p_ab / (p_a * p_b))
            scored.append(((a, b), pmi))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in scored[:top_n]]

    # ------------------------------------------------------------------
    # N-grams (generic)
    # ------------------------------------------------------------------
    def ngrams(self, n: int = 2) -> list:
        """All n-grams in order (as tuples)."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return [
            tuple(self.tokens[i:i + n])
            for i in range(len(self.tokens) - n + 1)
        ]

    def ngram_freq(self, n: int = 2) -> FreqDist:
        """FreqDist over n-grams."""
        return FreqDist(self.ngrams(n))

    # ------------------------------------------------------------------
    # Lexical diversity & dispersion
    # ------------------------------------------------------------------
    def lexical_diversity(self) -> float:
        """Type-token ratio: distinct tokens / total tokens."""
        total = len(self.tokens)
        return len(set(self.tokens)) / total if total else 0.0

    def dispersion(self, words: list) -> dict:
        """
        For each *word* return the list of token positions where it occurs.
        Equivalent to NLTK's dispersion_plot data.

        Returns
        -------
        dict[str, list[int]]
        """
        out = {w: [] for w in words}
        for i, tok in enumerate(self.tokens):
            if tok in out:
                out[tok].append(i)
        return out

    def count(self, word: str) -> int:
        """Number of times *word* appears."""
        return self.freq_dist()[word]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def vocabulary(self) -> set:
        """Set of unique tokens."""
        return set(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return (
            f"TextStats(tokens={len(self.tokens)}, "
            f"vocab={len(set(self.tokens))})"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Module-level convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────
def freq_dist(text) -> FreqDist:
    """One-shot FreqDist over a string or token list."""
    return TextStats(text).freq_dist()


def concordance(text, word: str, width: int = 40, lines: int = 25) -> list:
    """One-shot concordance lookup."""
    return TextStats(text).concordance(word, width=width, lines=lines)


def collocations(text, top_n: int = 20, min_count: int = 2) -> list:
    """One-shot collocation extraction."""
    return TextStats(text).collocations(top_n=top_n, min_count=min_count)


def lexical_diversity(text) -> float:
    """One-shot type-token ratio."""
    return TextStats(text).lexical_diversity()
