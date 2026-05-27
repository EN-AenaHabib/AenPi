# -*- coding: utf-8 -*-
"""
AenPi.urdu — POS Tagger Module
CRF-based Urdu POS Tagger
"""

import os
import re
import pickle
import urllib.request
from typing import Sequence

# ---------------------------
# Lazy dependency loader
# ---------------------------
def _require(pkg: str, pip_name: str = ""):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        raise ImportError(
            f"{pip_name or pkg} is required. Install using pip."
        )


# ---------------------------
# CoNLL-U reader
# ---------------------------
def read_conllu(path: str):
    sentences, current = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#"):
                continue

            if line == "":
                if current:
                    sentences.append(current)
                    current = []
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue

            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue

            word = cols[1]
            upos = cols[3]

            if upos == "_":
                continue

            current.append((word, upos))

    if current:
        sentences.append(current)

    return sentences


# ---------------------------
# Feature extraction
# ---------------------------
def _char_ngrams(word: str, n: int):
    return [word[i:i+n] for i in range(len(word)-n+1)]


def word_features(sent, i):
    word = sent[i]

    feats = {
        "word": word,
        "lower": word.lower(),
        "len": len(word),
        "isdigit": word.isdigit(),
    }

    for bg in _char_ngrams(word, 2):
        feats[f"bg={bg}"] = True

    for tg in _char_ngrams(word, 3):
        feats[f"tg={tg}"] = True

    return feats


def sent_to_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent):
    return [t for _, t in sent]


def sent_to_tokens(sent):
    return [w for w, _ in sent]


# ---------------------------
# POS TAGGER CLASS
# ---------------------------
class UrduPOSTagger:

    def __init__(self):
        self.model = None
        self.labels = []

    # -----------------------
    # TRAIN
    # -----------------------
    def train(self, train_path, dev_path=""):
        crf = _require("sklearn_crfsuite", "sklearn-crfsuite").CRF

        train_sents = read_conllu(train_path)

        X = [sent_to_features(sent_to_tokens(s)) for s in train_sents]
        y = [sent_to_labels(s) for s in train_sents]

        self.labels = list(set(t for s in train_sents for _, t in s))

        self.model = crf(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100
        )

        self.model.fit(X, y)

        print("Training complete")

    # -----------------------
    # PREDICT
    # -----------------------
    def tag(self, tokens):
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        feats = sent_to_features(tokens)
        pred = self.model.predict([feats])[0]

        return list(zip(tokens, pred))

    def tag_sentence(self, sentence):
        return self.tag(sentence.split())

    # -----------------------
    # SAVE / LOAD
    # -----------------------
    def save(self, path="pos_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path="pos_model.pkl"):
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj
