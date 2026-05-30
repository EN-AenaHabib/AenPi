"""
AenPi – Urdu POS Tagger
=======================
CRF-based tagger trained on Universal Dependencies Urdu Treebank.
~92% token accuracy on held-out test set.

SETUP
-----
Put BOTH files in your project folder:
  • pos_tagger.py
  • aenpi_pos_crf.pkl.gz   ← pre-trained model, no training needed

REQUIRES
--------
  pip install sklearn-crfsuite

USAGE
-----
  from pos_tagger import UrduPOSTagger
  tagger = UrduPOSTagger()
  print(tagger.tag_sentence("میں کتاب پڑھتا ہوں"))
  # [('میں', 'PRON'), ('کتاب', 'NOUN'), ('پڑھتا', 'VERB'), ('ہوں', 'AUX')]
"""

import os, re, gzip, pickle, urllib.request

# ── model lives next to this file ─────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "aenpi_pos_crf.pkl.gz")

# ── UD corpus URLs  (only used if model file is missing) ──────────────────────
_UD_BASE  = "https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/master/"
_UD_FILES = ["ur_udtb-ud-train.conllu", "ur_udtb-ud-dev.conllu"]


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENISER
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str):
    """Split Urdu text, separating punctuation from words."""
    tokens = []
    for raw in text.split():
        m = re.match(r'^([^\w\u0600-\u06FF]+)(.*)', raw, re.UNICODE)
        if m:
            tokens.append(m.group(1))
            raw = m.group(2)
        if not raw:
            continue
        m = re.match(r'(.+[\w\u0600-\u06FF])([^\w\u0600-\u06FF]+)$', raw, re.UNICODE)
        if m:
            tokens += [m.group(1), m.group(2)]
        else:
            tokens.append(raw)
    return [t for t in tokens if t.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES  (must match exactly what was used during training)
# ══════════════════════════════════════════════════════════════════════════════

def _features(sent, i):
    word = sent[i]
    f = {
        "bias":      "1",
        "word":      word,
        "suffix1":   word[-1:],
        "suffix2":   word[-2:],
        "suffix3":   word[-3:],
        "suffix4":   word[-4:] if len(word) >= 4 else word,
        "prefix1":   word[:1],
        "prefix2":   word[:2],
        "prefix3":   word[:3],
        "prefix4":   word[:4] if len(word) >= 4 else word,
        "len":       str(min(len(word), 15)),
        "has_digit": str(any(c.isdigit() for c in word)),
        "is_punct":  str(not any(c.isalpha() for c in word)),
        "is_first":  str(i == 0),
        "is_last":   str(i == len(sent) - 1),
    }
    if i > 0:
        pw = sent[i - 1]
        f["prev_word"]    = pw
        f["prev_suffix2"] = pw[-2:]
        f["prev_prefix2"] = pw[:2]
    else:
        f["BOS"] = "1"
    if i > 1:
        f["prev2_word"] = sent[i - 2]
    if i < len(sent) - 1:
        nw = sent[i + 1]
        f["next_word"]    = nw
        f["next_suffix2"] = nw[-2:]
        f["next_prefix2"] = nw[:2]
    else:
        f["EOS"] = "1"
    if i < len(sent) - 2:
        f["next2_word"] = sent[i + 2]
    return f


def _sent_features(tokens):
    return [_features(tokens, i) for i in range(len(tokens))]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-TRAIN  (fallback – only runs if .pkl.gz is missing)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "AenPi-NLP/1.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")


def _parse_conllu(text):
    sents, cur = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        idx, word, lemma, upos = parts[0], parts[1], parts[2], parts[3]
        if "-" in idx or "." in idx:
            continue
        if upos and upos != "_":
            cur.append((word, lemma, upos))
    if cur:
        sents.append(cur)
    return sents


def _build_and_save(model_path):
    try:
        import sklearn_crfsuite
    except ImportError:
        raise ImportError(
            "sklearn-crfsuite is required.\n"
            "Install it with:  pip install sklearn-crfsuite"
        )

    print("[AenPi] Model not found – downloading corpus and training CRF …")
    sentences = []
    for fname in _UD_FILES:
        print(f"  Downloading {fname} …")
        sentences.extend(_parse_conllu(_fetch(_UD_BASE + fname)))
    print(f"  Loaded {len(sentences)} sentences.")

    def s2f(sent):
        toks = [w for w, _, _ in sent]
        return [_features(toks, i) for i in range(len(toks))]

    X = [s2f(s) for s in sentences]
    y = [[t for _, _, t in s] for s in sentences]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.05, c2=0.05,
        max_iterations=200, all_possible_transitions=True,
    )
    print("  Training CRF (~1-2 min) …")
    crf.fit(X, y)
    print("  Done. Saving model …")

    with gzip.open(model_path, "wb") as fh:
        pickle.dump(crf, fh, protocol=4)
    print(f"  Model saved → {model_path}")
    return crf


def _load_model(model_path):
    with gzip.open(model_path, "rb") as fh:
        return pickle.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class UrduPOSTagger:
    """
    AenPi Urdu Part-of-Speech Tagger (CRF-based, ~92% accuracy).

    Parameters
    ----------
    model_path : str, optional
        Path to aenpi_pos_crf.pkl.gz. Defaults to the folder this file lives in.
    retrain : bool, optional
        Force retraining from scratch (default False).

    POS Tags (Universal Dependencies UPOS standard)
    ------------------------------------------------
    NOUN   – common noun       (کتاب، گھر)
    PROPN  – proper noun       (احمد، پاکستان)
    VERB   – main verb         (پڑھنا، جانا)
    AUX    – auxiliary verb    (ہے، ہوں، تھا)
    ADJ    – adjective         (اچھا، بڑا)
    ADV    – adverb            (بہت، یہاں)
    PRON   – pronoun           (میں، وہ، آپ)
    DET    – determiner        (یہ، وہ)
    ADP    – postposition      (میں، پر، سے، نے، کا)
    CCONJ  – coord. conj.      (اور، یا)
    SCONJ  – subord. conj.     (کہ، اگر)
    PART   – particle          (بھی، ہی، نہ)
    NUM    – numeral           (ایک، دو، 3)
    PUNCT  – punctuation       (۔ ، !)
    INTJ   – interjection      (واہ، اوہ)
    X      – other / foreign
    """

    def __init__(self, model_path: str = _MODEL, retrain: bool = False):
        if retrain or not os.path.exists(model_path):
            self._crf = _build_and_save(model_path)
        else:
            self._crf = _load_model(model_path)

    # ── primary method ────────────────────────────────────────────────────────

    def tag_sentence(self, sentence: str):
        """
        Tag a single Urdu sentence string.

        Prints and returns a list of (word, POS-tag) tuples.

        Example
        -------
        >>> tagger.tag_sentence("میں کتاب پڑھتا ہوں")
        [('میں', 'PRON'), ('کتاب', 'NOUN'), ('پڑھتا', 'VERB'), ('ہوں', 'AUX')]
        """
        tokens = _tokenize(sentence)
        if not tokens:
            return []
        tags   = self._crf.predict([_sent_features(tokens)])[0]
        result = list(zip(tokens, tags))
        print(result)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def tag_tokens(self, tokens: list):
        """Tag a pre-tokenised list of words. Returns (word, tag) list."""
        if not tokens:
            return []
        tags = self._crf.predict([_sent_features(tokens)])[0]
        return list(zip(tokens, tags))

    def tag_batch(self, sentences: list):
        """
        Tag many sentences in one efficient batch call.
        Returns a list of (word, tag) lists. Does not print.

        Example
        -------
        >>> tagger.tag_batch(["میں گھر جاتا ہوں", "وہ کتاب پڑھتی ہے"])
        """
        tokenised = [_tokenize(s) for s in sentences]
        all_tags  = self._crf.predict([_sent_features(t) for t in tokenised])
        return [list(zip(t, g)) for t, g in zip(tokenised, all_tags)]


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tagger = UrduPOSTagger()
    print(tagger.tag_sentence("میں کتاب پڑھتا ہوں"))
    print(tagger.tag_sentence("وہ بازار جاتا ہے"))
    print(tagger.tag_sentence("اسلام آباد پاکستان کا دارالحکومت ہے"))
    print(tagger.tag_sentence("آج موسم بہت اچھا ہے"))
    print(tagger.tag_sentence("احمد نے گھر خریدا"))
