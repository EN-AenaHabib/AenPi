"""
AenPi - Urdu POS Tagger
Automatically downloads and trains on the UPOS (Universal Dependencies Urdu) dataset.
No manual setup required. Just import and use.
"""

import os
import pickle
import re
import urllib.request
import json
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_DIR, "aenpi_pos_model.pkl")

# ── Universal Dependencies Urdu treebank (raw CoNLL-U files) ───────────────────
_UD_URLS = [
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/master/ur_udtb-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/master/ur_udtb-ud-dev.conllu",
]

# ── UPOS tagset (used by UD) ────────────────────────────────────────────────────
UPOS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
    "CONJ", "PART", "NUM", "PUNCT", "X", "PROPN", "AUX",
    "CCONJ", "SCONJ", "INTJ",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _download_conllu(url: str) -> str:
    """Fetch a CoNLL-U file from the web and return its text."""
    req = urllib.request.Request(url, headers={"User-Agent": "AenPi-NLP/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def _parse_conllu(text: str):
    """
    Parse CoNLL-U text into a list of sentences.
    Each sentence is a list of (word, upos) tuples.
    """
    sentences, current = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if current:
                sentences.append(current)
                current = []
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        idx, word, _lemma, upos = parts[0], parts[1], parts[2], parts[3]
        # skip multi-word tokens (e.g. "1-2")
        if "-" in idx or "." in idx:
            continue
        if upos and upos != "_":
            current.append((word, upos))
    if current:
        sentences.append(current)
    return sentences


def _load_corpus():
    sentences = []
    for url in _UD_URLS:
        try:
            print(f"  Downloading: {url.split('/')[-1]}")
            text = _download_conllu(url)
            sents = _parse_conllu(text)
            sentences.extend(sents)
            print(f"    → {len(sents)} sentences loaded")
        except Exception as e:
            print(f"  Warning: could not load {url}: {e}")
    return sentences


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _word_features(sentence, i):
    word = sentence[i]
    features = {
        "word":        word,
        "suffix2":     word[-2:] if len(word) >= 2 else word,
        "suffix3":     word[-3:] if len(word) >= 3 else word,
        "prefix2":     word[:2]  if len(word) >= 2 else word,
        "prefix3":     word[:3]  if len(word) >= 3 else word,
        "is_first":    i == 0,
        "is_last":     i == len(sentence) - 1,
        "word_len":    len(word),
        "has_digit":   any(c.isdigit() for c in word),
        "is_punct":    all(not c.isalpha() for c in word),
        "prev_word":   sentence[i - 1] if i > 0 else "<START>",
        "next_word":   sentence[i + 1] if i < len(sentence) - 1 else "<END>",
        "prev2_word":  sentence[i - 2] if i > 1 else "<START>",
        "next2_word":  sentence[i + 2] if i < len(sentence) - 2 else "<END>",
    }
    return features


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Averaged Perceptron tagger  (no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

class _AveragedPerceptron:
    def __init__(self):
        self.weights   = defaultdict(lambda: defaultdict(float))
        self._totals   = defaultdict(lambda: defaultdict(float))
        self._stamps   = defaultdict(lambda: defaultdict(int))
        self._i        = 0

    def predict(self, features):
        scores = defaultdict(float)
        for feat, val in features.items():
            if not val or feat not in self.weights:
                continue
            w = self.weights[feat]
            for label, weight in w.items():
                scores[label] += weight * (1 if val is True else val if isinstance(val, (int, float)) else 1)
        return max(scores, key=scores.get) if scores else "NOUN"

    def update(self, truth, guess, features):
        self._i += 1
        for feat in features:
            self._update_feat(truth, feat,  1.0)
            self._update_feat(guess, feat, -1.0)

    def _update_feat(self, label, feat, delta):
        self._totals[feat][label] += (self._i - self._stamps[feat][label]) * self.weights[feat][label]
        self._stamps[feat][label]  = self._i
        self.weights[feat][label] += delta

    def average_weights(self):
        """Call once after training to finalise averaged weights."""
        for feat, weights in self.weights.items():
            for label in weights:
                total = self._totals[feat][label]
                total += (self._i - self._stamps[feat][label]) * weights[label]
                self.weights[feat][label] = total / self._i if self._i else 0


class _PerceptronTagger:
    def __init__(self):
        self.model    = _AveragedPerceptron()
        self.tag_dict = {}   # word → most-frequent-tag (for unambiguous words)
        self.classes  = set()

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, sentences, epochs=5):
        import random
        # build word→tag frequency table
        freq = defaultdict(lambda: defaultdict(int))
        for sent in sentences:
            for word, tag in sent:
                freq[word][tag] += 1
                self.classes.add(tag)

        # shortcut dict: only for words that always get the same tag
        for word, tag_counts in freq.items():
            top_tag, top_cnt = max(tag_counts.items(), key=lambda x: x[1])
            total = sum(tag_counts.values())
            if top_cnt / total >= 0.98 and total >= 5:
                self.tag_dict[word] = top_tag

        print(f"  Training on {len(sentences)} sentences for {epochs} epochs …")
        for epoch in range(epochs):
            correct = total_words = 0
            random.shuffle(sentences)
            for sent in sentences:
                words = [w for w, _ in sent]
                tags  = [t for _, t in sent]
                prev, prev2 = "<START>", "<START2>"
                for i, (word, truth) in enumerate(zip(words, tags)):
                    guess = self.tag_dict.get(word)
                    if not guess:
                        feats = self._features(words, i, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(truth, guess, feats)
                    correct     += guess == truth
                    total_words += 1
                    prev2, prev  = prev, guess
            acc = correct / total_words * 100
            print(f"    Epoch {epoch + 1}/{epochs}  accuracy: {acc:.2f}%")
        self.model.average_weights()

    # ── inference ─────────────────────────────────────────────────────────────

    def tag(self, words):
        tags = []
        prev, prev2 = "<START>", "<START2>"
        for i, word in enumerate(words):
            tag = self.tag_dict.get(word)
            if not tag:
                feats = self._features(words, i, prev, prev2)
                tag   = self.model.predict(feats)
            tags.append(tag)
            prev2, prev = prev, tag
        return list(zip(words, tags))

    # ── features ──────────────────────────────────────────────────────────────

    def _features(self, words, i, prev, prev2):
        word = words[i]
        return {
            "bias":          True,
            "word":          word.lower(),
            "suffix2":       word[-2:],
            "suffix3":       word[-3:] if len(word) >= 3 else word,
            "suffix4":       word[-4:] if len(word) >= 4 else word,
            "prefix2":       word[:2],
            "prefix3":       word[:3] if len(word) >= 3 else word,
            "word_len_bin":  min(len(word), 10),
            "has_digit":     any(c.isdigit() for c in word),
            "is_punct":      all(not c.isalpha() for c in word),
            "prev_tag":      prev,
            "prev2_tag":     prev2,
            "prev_tag+word": f"{prev}+{word.lower()}",
            "prev_word":     words[i - 1].lower() if i > 0 else "<START>",
            "next_word":     words[i + 1].lower() if i < len(words) - 1 else "<END>",
            "prev2_word":    words[i - 2].lower() if i > 1 else "<START>",
            "next2_word":    words[i + 2].lower() if i < len(words) - 2 else "<END>",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Tokeniser  (simple Urdu-aware split)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str):
    """Split Urdu text on whitespace and separate leading/trailing punctuation."""
    tokens = []
    for raw in text.split():
        # pull off leading punctuation
        m = re.match(r'^([^\w\u0600-\u06FF]+)(.*)', raw, re.UNICODE)
        if m:
            tokens.append(m.group(1))
            raw = m.group(2)
        if not raw:
            continue
        # pull off trailing punctuation
        m = re.match(r'(.*[^\W\d_\u0600-\u06FF])([^\w\u0600-\u06FF]+)$', raw, re.UNICODE)
        if m:
            tokens.append(m.group(1))
            tokens.append(m.group(2))
        else:
            tokens.append(raw)
    return [t for t in tokens if t.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Public API
# ══════════════════════════════════════════════════════════════════════════════

class UrduPOSTagger:
    """
    AenPi Urdu POS Tagger.

    On first use the model is automatically trained from the Universal
    Dependencies Urdu treebank and cached to disk.  Every subsequent call
    just loads the cached model — no internet connection needed.

    Usage
    -----
    from pos_tagger import UrduPOSTagger
    tagger = UrduPOSTagger()
    print(tagger.tag_sentence("میں کتاب پڑھتا ہوں"))
    """

    def __init__(self, model_path: str = _MODEL, retrain: bool = False):
        self._tagger = _PerceptronTagger()
        if not retrain and os.path.exists(model_path):
            self._load(model_path)
        else:
            self._train_and_save(model_path)

    # ── public methods ────────────────────────────────────────────────────────

    def tag_sentence(self, sentence: str):
        """
        Tag a single Urdu sentence string.

        Returns a list of (word, POS-tag) tuples and also prints them.

        Example
        -------
        >>> tagger.tag_sentence("میں کتاب پڑھتا ہوں")
        [('میں', 'PRON'), ('کتاب', 'NOUN'), ('پڑھتا', 'VERB'), ('ہوں', 'AUX')]
        """
        tokens = _tokenize(sentence)
        if not tokens:
            return []
        tagged = self._tagger.tag(tokens)
        print(tagged)
        return tagged

    def tag_tokens(self, tokens: list):
        """Tag a pre-tokenised list of Urdu words."""
        return self._tagger.tag(tokens)

    # ── internal ──────────────────────────────────────────────────────────────

    def _train_and_save(self, model_path: str):
        print("[AenPi] First run — downloading corpus and training POS model …")
        sentences = _load_corpus()
        if not sentences:
            raise RuntimeError(
                "Could not download training data. "
                "Check your internet connection and try again."
            )
        self._tagger.train(sentences, epochs=5)
        self._save(model_path)
        print(f"[AenPi] Model saved to: {model_path}")

    def _save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weights":   dict(self._tagger.model.weights),
                    "tag_dict":  self._tagger.tag_dict,
                    "classes":   self._tagger.classes,
                },
                f,
                protocol=4,
            )

    def _load(self, path: str):
        print(f"[AenPi] Loading cached POS model from: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._tagger.model.weights = defaultdict(
            lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in data["weights"].items()}
        )
        self._tagger.tag_dict = data["tag_dict"]
        self._tagger.classes  = data["classes"]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Quick demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tagger = UrduPOSTagger()
    print(tagger.tag_sentence("میں کتاب پڑھتا ہوں"))
    print(tagger.tag_sentence("وہ بازار جاتا ہے"))
    print(tagger.tag_sentence("اسلام آباد پاکستان کا دارالحکومت ہے"))
