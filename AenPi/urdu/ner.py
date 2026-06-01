"""
ner.py — Urdu Named Entity Recognition
=======================================
Lightweight CRF-based NER using WikiANN Urdu (ur) dataset.
~20K sentences, clean PER/LOC/ORG labels, train/val/test splits.
Zero legal/licensing issues. Works offline after first download.
using Wkin dataset
Usage (Inference)
-----------------
    from ner import UrduNER
    ner = UrduNER()
    print(ner.tag("محمد علی لاہور چلے گئے۔"))
    print(ner.get_entities("وزیر اعظم نے اسلام آباد میں تقریر کی"))

Usage (Training)
----------------
    ner = UrduNER()
    ner.fit()          # trains on full WikiANN ur split (~20K sentences)
    ner.fit(limit=5000) # or a quick subset
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import joblib

# ── Label Map ─────────────────────────────────────────────────────────────────
# WikiANN label order (integer index → BIO string):
# 0=O  1=B-PER  2=I-PER  3=B-ORG  4=I-ORG  5=B-LOC  6=I-LOC
WIKIANN_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

# ── Urdu Unicode & digit patterns ─────────────────────────────────────────────
_URDU_RE  = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_DIGIT_RE = re.compile(r"[\d\u0660-\u0669\u06F0-\u06F9]")


# ── Feature Extractor ──────────────────────────────────────────────────────────
def _features(tokens: List[str], i: int) -> dict:
    """Rich Urdu-aware CRF feature function (window ±2)."""
    w = tokens[i]
    n = len(tokens)

    if not w:
        return {"bias": 1.0}

    feats: dict = {
        "bias":      1.0,
        "word":      w,
        "len":       min(len(w), 20),
        "suffix1":   w[-1:],
        "suffix2":   w[-2:],
        "suffix3":   w[-3:],
        "prefix1":   w[:1],
        "prefix2":   w[:2],
        "prefix3":   w[:3],
        "is_urdu":   bool(_URDU_RE.search(w)),
        "is_digit":  bool(_DIGIT_RE.fullmatch(w)),
        "has_digit": bool(_DIGIT_RE.search(w)),
        "is_punct":  unicodedata.category(w[0]).startswith("P"),
        "is_first":  i == 0,
        "is_last":   i == n - 1,
        "rel_pos":   round(i / max(n - 1, 1), 1),
    }

    # Previous context (window = 2)
    if i >= 1:
        p1 = tokens[i - 1]
        feats.update({
            "prev1":         p1,
            "prev1_suffix2": p1[-2:],
            "prev1_is_urdu": bool(_URDU_RE.search(p1)),
            "bigram_prev":   f"{p1}|{w}",
        })
    else:
        feats["prev1"] = "<START>"

    if i >= 2:
        p2 = tokens[i - 2]
        feats.update({"prev2": p2, "prev2_suffix2": p2[-2:]})
    else:
        feats["prev2"] = "<START2>"

    # Next context (window = 2)
    if i < n - 1:
        n1 = tokens[i + 1]
        feats.update({
            "next1":         n1,
            "next1_suffix2": n1[-2:],
            "next1_is_urdu": bool(_URDU_RE.search(n1)),
            "bigram_next":   f"{w}|{n1}",
        })
    else:
        feats["next1"] = "<END>"

    if i < n - 2:
        n2 = tokens[i + 2]
        feats.update({"next2": n2, "next2_suffix2": n2[-2:]})
    else:
        feats["next2"] = "<END2>"

    return feats


def _sentence_features(tokens: List[str]) -> List[dict]:
    return [_features(tokens, i) for i in range(len(tokens))]


# ── BIO Span Merger ────────────────────────────────────────────────────────────
def _merge_spans(tagged: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Converts flat BIO token list into grouped entity dict.
    Fixed: flush() now always resets buf/etype to prevent duplication.
    """
    entities: Dict[str, List[str]] = {}
    buf: List[str] = []
    etype: Optional[str] = None

    def flush() -> None:
        nonlocal buf, etype
        if etype and buf:
            entities.setdefault(etype, []).append(" ".join(buf))
        buf, etype = [], None  # always reset — fixes silent duplication bug

    for token, label in tagged:
        if label == "O":
            flush()
        elif label.startswith("B-"):
            flush()
            etype = label[2:]
            buf = [token]
        elif label.startswith("I-"):
            t = label[2:]
            if t == etype:
                buf.append(token)
            else:
                # I-tag mismatch → treat as new entity start
                flush()
                etype, buf = t, [token]

    flush()  # flush final entity
    return entities


# ── Main Class ─────────────────────────────────────────────────────────────────
class UrduNER:
    """
    Lightweight Urdu NER using CRF + WikiANN (ur) dataset.
    Trains in ~2 min on CPU. Model file is ~5–15 MB.
    """

    _DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "urdu_ner_crf.joblib",
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path: str = model_path or self._DEFAULT_MODEL
        self._crf = None
        self._label_list: List[str] = WIKIANN_LABELS

        if os.path.exists(self.model_path):
            self._load()

    def _load(self) -> None:
        try:
            payload = joblib.load(self.model_path)
            # Support both plain CRF saves and dict payload with label list
            if isinstance(payload, dict):
                self._crf = payload["crf"]
                self._label_list = payload.get("label_list", WIKIANN_LABELS)
            else:
                self._crf = payload
            print(f"✅ Model loaded from: {self.model_path}")
        except Exception as exc:
            raise RuntimeError(f"❌ Cannot load model at {self.model_path}: {exc}") from exc

    def tokenize(self, text: str) -> List[str]:
        """
        Urdu-aware tokenizer.
        - Splits on zero-width non-joiners (common in Urdu text)
        - Isolates punctuation attached to words like 'لاہور۔'
        """
        text = text.replace("\u200c", " ").replace("\u200d", " ")
        return re.findall(r"[\w\u0600-\u06FF]+|[^\w\s]", text, re.UNICODE)

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, limit: Optional[int] = None) -> None:
        """
        Train CRF on WikiANN Urdu (ur).
        Dataset: ~20K sentences, ~163K tokens, 3 entity types.
        Full training takes ~2–3 min on CPU.

        Args:
            limit: cap number of training sentences (None = use all).
        """
        try:
            import sklearn_crfsuite
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install dependencies: pip install sklearn-crfsuite datasets"
            ) from e

        print("📥 Loading WikiANN Urdu dataset (lightweight, ~20K sentences)...")
        ds = load_dataset("unimelb-nlp/wikiann", "ur")

        # WikiANN has proper ClassLabel features — read label names directly
        # This is the correct way; never hardcode index→label mapping
        label_list: List[str] = ds["train"].features["ner_tags"].feature.names
        self._label_list = label_list
        print(f"   Labels detected: {label_list}")

        train_split = ds["train"]
        if limit:
            train_split = train_split.select(range(min(limit, len(train_split))))

        print(f"   Training on {len(train_split)} sentences...")

        X_train, y_train = [], []
        skipped = 0

        for item in train_split:
            tokens = item["tokens"]
            tags   = item["ner_tags"]

            # Validate lengths match before appending
            if not tokens or len(tokens) != len(tags):
                skipped += 1
                continue

            X_train.append(_sentence_features(tokens))
            # Map integer indices → label strings using dataset's own ClassLabel
            y_train.append([label_list[t] for t in tags])

        if skipped:
            print(f"   ⚠️  Skipped {skipped} malformed samples.")

        print(f"🚀 Training CRF (lbfgs, max_iter=150)...")
        self._crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.05,       # L1 — slightly lower promotes recall
            c2=0.01,       # L2 — keeps weights stable
            max_iterations=150,  # enough for WikiANN to converge
            all_possible_transitions=True,
            verbose=False,
        )
        self._crf.fit(X_train, y_train)

        # Evaluate on validation split
        self._evaluate(ds["validation"], label_list)

        # Save both CRF and label list together
        joblib.dump({"crf": self._crf, "label_list": label_list}, self.model_path)
        print(f"✅ Model saved to: {self.model_path}")

    def _evaluate(self, val_split, label_list: List[str]) -> None:
        """Quick F1 report on validation set after training."""
        try:
            from sklearn_crfsuite import metrics as crf_metrics
        except ImportError:
            return

        X_val, y_val = [], []
        for item in val_split:
            tokens = item["tokens"]
            tags   = item["ner_tags"]
            if not tokens or len(tokens) != len(tags):
                continue
            X_val.append(_sentence_features(tokens))
            y_val.append([label_list[t] for t in tags])

        y_pred = self._crf.predict(X_val)
        entity_labels = [l for l in label_list if l != "O"]

        print("\n📊 Validation Results:")
        print(crf_metrics.flat_classification_report(
            y_val, y_pred, labels=entity_labels, digits=3
        ))

    # ── Inference ──────────────────────────────────────────────────────────────
    def _check_loaded(self) -> None:
        if self._crf is None:
            raise RuntimeError(
                "Model not loaded. Run .fit() to train or provide a model file."
            )

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Tag a single Urdu string. Returns list of (token, label) tuples."""
        self._check_loaded()
        tokens = self.tokenize(text)
        if not tokens:
            return []
        preds = self._crf.predict([_sentence_features(tokens)])[0]
        # Safe fallback: unknown prediction index → "O"
        return [(t, p if p in self._label_list else "O") for t, p in zip(tokens, preds)]

    def tag_batch(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """Tag multiple strings efficiently in one CRF call."""
        self._check_loaded()

        tokenized = [self.tokenize(t) for t in texts]
        # Only process non-empty; preserve original indices
        active = [(i, toks) for i, toks in enumerate(tokenized) if toks]

        if not active:
            return [[] for _ in texts]

        indices, batches = zip(*active)
        feats   = [_sentence_features(toks) for toks in batches]
        all_pred = self._crf.predict(feats)

        out: List[List[Tuple[str, str]]] = [[] for _ in texts]
        for orig_i, toks, preds in zip(indices, batches, all_pred):
            out[orig_i] = [
                (t, p if p in self._label_list else "O")
                for t, p in zip(toks, preds)
            ]
        return out

    def get_entities(self, text: str) -> Dict[str, List[str]]:
        """Return grouped entity dict for a single text."""
        return _merge_spans(self.tag(text))

    def get_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Return grouped entity dicts for multiple texts."""
        return [_merge_spans(tagged) for tagged in self.tag_batch(texts)]

    def save(self, path: str) -> None:
        """Explicitly save model to a custom path."""
        self._check_loaded()
        joblib.dump({"crf": self._crf, "label_list": self._label_list}, path)
        print(f"✅ Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model from a custom path."""
        self.model_path = path
        self._load()

    def __repr__(self) -> str:
        status = "loaded" if self._crf else "not loaded"
        return f"UrduNER(model='{self.model_path}', status={status})"


# ── Quick Sanity Check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ner = UrduNER(model_path="urdu_ner_crf.joblib")

    # Train (remove limit=None for full dataset)
    ner.fit(limit=None)

    test_sentences = [
        "جسٹس فائز عیسی نے لاہور اور اسلام آباد ہائی کورٹ میں سماعت مکمل کی",
        "محمد علی جناح نے پاکستان مسلم لیگ کی قیادت کی",
        "وزیر اعظم نے اقوام متحدہ کے اجلاس میں شرکت کی",
    ]

    print("\n" + "=" * 50)
    print("🔎 INFERENCE TEST")
    print("=" * 50)

    for sent in test_sentences:
        print(f"\n📝 {sent}")
        print("   Tokens:")
        for word, tag in ner.tag(sent):
            marker = f"  ← {tag}" if tag != "O" else ""
            print(f"     {word:20}{marker}")
        print("   Entities:", ner.get_entities(sent))

    # Colab download hook
    try:
        from google.colab import files
        files.download("urdu_ner_crf.joblib")
    except ImportError:
        print("\n💡 Local run — model saved to urdu_ner_crf.joblib")
