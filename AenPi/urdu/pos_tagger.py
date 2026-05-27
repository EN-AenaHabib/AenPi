# -*- coding: utf-8 -*-
"""
AenPi Urdu POS Tagger (CRF-based, production ready)

- No training at runtime
- Loads pretrained model automatically
- Falls back to download if missing
"""

import os
import gzip
import pickle
import urllib.request

import sklearn_crfsuite


# ─────────────────────────────────────────────────────────────
# Model location
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aenpi_pos_crf.pkl.gz")

# optional GitHub fallback (you can change later)
MODEL_URL = "https://github.com/YOUR_USERNAME/AenPi/releases/download/v1/aenpi_pos_crf.pkl.gz"


# ─────────────────────────────────────────────────────────────
# Feature extraction (CRF standard)
# ─────────────────────────────────────────────────────────────
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
    }

    # character ngrams
    for bg in _char_ngrams(word, 2):
        feats[f"bg={bg}"] = True

    for tg in _char_ngrams(word, 3):
        feats[f"tg={tg}"] = True

    # context (VERY important for CRF)
    feats["prev_word"] = sent[i-1] if i > 0 else "<START>"
    feats["next_word"] = sent[i+1] if i < len(sent)-1 else "<END>"

    return feats


def _sent_features(sent):
    return [_features(sent, i) for i in range(len(sent))]


# ─────────────────────────────────────────────────────────────
# Download model if missing
# ─────────────────────────────────────────────────────────────
def _download_model():
    print("[AenPi] Downloading pretrained POS model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────
class UrduPOSTagger:

    def __init__(self):
        self.model = None

        # auto-load model
        if not os.path.exists(MODEL_PATH):
            try:
                _download_model()
            except:
                raise FileNotFoundError(
                    "POS model not found and download failed."
                )

        self._load_model()

    # ─────────────────────────────────────────────
    # LOAD MODEL
    # ─────────────────────────────────────────────
    def _load_model(self):
        with gzip.open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        print("[AenPi] POS model loaded successfully")

    # ─────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────
    def tag(self, tokens):
        feats = _sent_features(tokens)
        preds = self.model.predict([feats])[0]
        return list(zip(tokens, preds))

    def tag_sentence(self, sentence):
        return self.tag(sentence.split())
