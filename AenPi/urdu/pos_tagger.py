# -*- coding: utf-8 -*-
"""
AenPi Urdu POS Tagger (CRF-based, production ready)
Author: NuTech AI-23 | F23607023

Location: AenPi/urdu/pos_tagger.py
"""

import os
import gzip
import pickle
import urllib.request

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aenpi_pos_crf.pkl.gz")

# Replace this with your real GitHub release link later
MODEL_URL = "https://github.com/YOUR_USERNAME/AenPi/releases/download/v1/aenpi_pos_crf.pkl.gz"


# ---------------------------------------------------------------------
# Feature extraction (CRF standard)
# ---------------------------------------------------------------------
def _char_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word) - n + 1)]


def _features(sent, i):
    word = sent[i]

    feats = {
        "bias": 1.0,
        "word": word,
        "lower": word.lower(),
        "prefix2": word[:2],
        "prefix3": word[:3],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "len": len(word),
        "isdigit": word.isdigit(),
        "is_first": i == 0,
        "is_last": i == len(sent) - 1,
        "prev_word": sent[i - 1] if i > 0 else "<START>",
        "next_word": sent[i + 1] if i < len(sent) - 1 else "<END>",
    }

    for bg in _char_ngrams(word, 2):
        feats[f"bg={bg}"] = True

    for tg in _char_ngrams(word, 3):
        feats[f"tg={tg}"] = True

    return feats


def _sent_features(sent):
    return [_features(sent, i) for i in range(len(sent))]


# ---------------------------------------------------------------------
# Download model
# ---------------------------------------------------------------------
def _download_model():
    print("[AenPi] Downloading pretrained Urdu POS model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ---------------------------------------------------------------------
# POS TAGGER CLASS
# ---------------------------------------------------------------------
class UrduPOSTagger:
    """
    CRF-based Urdu POS Tagger (AenPi)

    Usage:
        from AenPi.urdu.pos_tagger import UrduPOSTagger
        tagger = UrduPOSTagger()
        print(tagger.tag_sentence("میں کتاب پڑھتا ہوں"))
    """

    def __init__(self):
        self.model = None
        self.labels = []

        if not os.path.exists(MODEL_PATH):
            try:
                _download_model()
            except Exception:
                raise FileNotFoundError(
                    "Model not found and download failed. "
                    "Please upload aenpi_pos_crf.pkl.gz in urdu folder."
                )

        self._load_model()

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------
    def _load_model(self):
        with gzip.open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.labels = data.get("labels", [])

        print("[AenPi] Urdu POS model loaded successfully")

    # ---------------------------------------------------------
    # Tag tokens
    # ---------------------------------------------------------
    def tag(self, tokens):
        feats = _sent_features(tokens)
        preds = self.model.predict([feats])[0]
        return list(zip(tokens, preds))

    def tag_sentence(self, sentence):
        return self.tag(sentence.split())
