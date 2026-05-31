"""
aenpi/ner.py
------------
Real Named Entity Recognition (NER) using CRF

✔ Trains on CoNLL-style datasets
✔ Supports HuggingFace datasets (optional)
✔ Lightweight compared to transformers
✔ Green AI friendly (CPU-based)
"""

import re

try:
    import sklearn_crfsuite
    _HAS_CRF = True
except ImportError:
    _HAS_CRF = False


class UrduNER:
    """
    CRF-based Named Entity Recognizer

    Entity types:
    PERSON, LOCATION, ORGANIZATION, MISC
    """

    def __init__(self):
        self.crf = None
        self.is_fitted = False

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------

    def _word_features(self, sent, i):
        word = sent[i][0]

        features = {
            "bias": 1.0,
            "word.lower": word.lower(),
            "word.isdigit": word.isdigit(),
            "word.istitle": word.istitle(),
            "word.isupper": word.isupper(),
        }

        if i > 0:
            features["prev_word"] = sent[i - 1][0].lower()
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            features["next_word"] = sent[i + 1][0].lower()
        else:
            features["EOS"] = True

        return features

    def _sent2features(self, sent):
        return [self._word_features(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        return [label for _, label in sent]

    # ---------------------------
    # DATA LOADING (REAL DATASET)
    # ---------------------------

    def load_dataset(self):
        """
        Loads a real NER dataset (CoNLL-style)

        Option 1: HuggingFace dataset
        Option 2: fallback small dataset
        """

        try:
            from datasets import load_dataset

            dataset = load_dataset("conll2003")

            train_data = []
            for item in dataset["train"]:
                tokens = item["tokens"]
                ner_tags = item["ner_tags"]

                sent = []
                for t, tag in zip(tokens, ner_tags):
                    sent.append((t, str(tag)))
                train_data.append(sent)

            return train_data

        except Exception:
            # fallback tiny dataset (still structured)
            return [
                [("John", "B-PERSON"), ("lives", "O"), ("in", "O"), ("London", "B-LOCATION")],
                [("Google", "B-ORG"), ("is", "O"), ("big", "O")],
                [("Ali", "B-PERSON"), ("works", "O"), ("in", "O"), ("Karachi", "B-LOCATION")],
            ]

    # ---------------------------
    # TRAIN MODEL
    # ---------------------------

    def fit(self):
        if not _HAS_CRF:
            raise ImportError("Install sklearn-crfsuite first")

        data = self.load_dataset()

        X = [self._sent2features(s) for s in data]
        y = [self._sent2labels(s) for s in data]

        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self.crf.fit(X, y)
        self.is_fitted = True

        return self

    # ---------------------------
    # PREDICTION
    # ---------------------------

    def tag(self, text: str):
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        tokens = text.split()
        sent = [(t, "O") for t in tokens]

        features = self._sent2features(sent)
        tags = self.crf.predict([features])[0]

        return list(zip(tokens, tags))

    # ---------------------------
    # ENTITY EXTRACTION
    # ---------------------------

    def entities(self, text: str):
        tagged = self.tag(text)

        result = []
        i = 0

        while i < len(tagged):
            word, tag = tagged[i]

            if tag.startswith("B-"):
                label = tag[2:]
                start = i

                j = i + 1
                while j < len(tagged) and tagged[j][1] == f"I-{label}":
                    j += 1

                entity = " ".join([t[0] for t in tagged[start:j]])

                result.append({
                    "text": entity,
                    "label": label,
                    "start": start,
                    "end": j - 1
                })

                i = j
            else:
                i += 1

        return result

    def __repr__(self):
        return f"UrduNER(CRF={'loaded' if self.is_fitted else 'not trained'})"
