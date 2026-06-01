"""
urdu_ner.py — Urdu Named Entity Recognition
============================================
Hybrid Inference & Training Module. Fully decoupled.
Optimized for zero-network execution if a model file is found locally,
with built-in memory-safe dataset streaming capabilities for training.

Usage (Inference)
-----------------
    from urdu_ner import UrduNER

    ner = UrduNER()
    print(ner.tag("محمد علی لاہور چلے گئے۔"))
    print(ner.get_entities("وزیر اعظم نے اسلام آباد میں تقریر کی"))

Usage (Training in Colab)
------------------------
    ner = UrduNER()
    ner.fit(limit=15000) # Streams safely, will not crash RAM
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import joblib

# ── Label Map ────────────────────────────────────────────────────────────────
# Updated to match the specific structural indices used by the Urdu-Legal dataset
LABEL_MAP: Dict[str, str] = {
    "0": "O",
    "1": "B-PERSON",
    "2": "I-PERSON",
    "3": "B-LOCATION",
    "4": "I-LOCATION",
    "5": "B-ORG",
    "6": "I-ORG",
    "7": "B-DATE",
    "8": "I-DATE",
    "9": "B-LEGAL_ACTION",
    "10": "I-LEGAL_ACTION",
}

# Inverse lookup map for data alignment tasks during training loops
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ── Urdu Unicode Ranges & Punctuation Matchers ─────────────────────────────────
_URDU_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_DIGIT_RE   = re.compile(r"[\d\u0660-\u0669\u06F0-\u06F9]")   # ASCII + Arabic-Indic


# ── Feature Extractor ────────────────────────────────────────────────────────
def _features(tokens: List[str], i: int) -> dict:
    """
    Rich, Urdu-aware feature engine for sequence modelling.
    Note: Do not mutate keys or structures without executing a full training run.
    """
    w  = tokens[i]
    n  = len(tokens)

    # ── Current token geometry ───────────────────────────────────
    feats: dict = {
        "bias":         1.0,
        "word":         w,
        "len":          min(len(w), 20),
        "suffix1":      w[-1:],
        "suffix2":      w[-2:],
        "suffix3":      w[-3:],
        "prefix1":      w[:1],
        "prefix2":      w[:2],
        "prefix3":      w[:3],
        "is_urdu":      bool(_URDU_RANGE.search(w)),
        "is_digit":     bool(_DIGIT_RE.fullmatch(w)),
        "has_digit":    bool(_DIGIT_RE.search(w)),
        "is_punct":     unicodedata.category(w[0]).startswith("P") if w else False,
        "all_upper":    w.isupper(), 
        "is_first":     i == 0,
        "is_last":      i == n - 1,
        "rel_pos":      round(i / max(n - 1, 1), 1),
    }

    # ── Historical context markers (window = 2) ───────────────────
    if i >= 1:
        p1 = tokens[i - 1]
        feats.update({
            "prev1":         p1,
            "prev1_suffix2": p1[-2:],
            "prev1_is_urdu": bool(_URDU_RANGE.search(p1)),
        })
    else:
        feats["prev1"] = "<START>"

    if i >= 2:
        p2 = tokens[i - 2]
        feats.update({
            "prev2":         p2,
            "prev2_suffix2": p2[-2:],
        })
    else:
        feats["prev2"] = "<START2>"

    # ── Future context markers (window = 2) ───────────────────────
    if i < n - 1:
        n1 = tokens[i + 1]
        feats.update({
            "next1":         n1,
            "next1_suffix2": n1[-2:],
            "next1_is_urdu": bool(_URDU_RANGE.search(n1)),
        })
    else:
        feats["next1"] = "<END>"

    if i < n - 2:
        n2 = tokens[i + 2]
        feats.update({
            "next2":         n2,
            "next2_suffix2": n2[-2:],
        })
    else:
        feats["next2"] = "<END2>"

    # ── Bigram configurations ────────────────────────────────────
    if i >= 1:
        feats["bigram_prev"] = f"{tokens[i-1]}|{w}"
    if i < n - 1:
        feats["bigram_next"] = f"{w}|{tokens[i+1]}"

    return feats


def _sentence_features(tokens: List[str]) -> List[dict]:
    return [_features(tokens, i) for i in range(len(tokens))]


# ── Entity Span Merger ───────────────────────────────────────────────────────
def _merge_spans(tagged: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Converts a sequence of flat BIO tuples into clustered string groupings."""
    entities: Dict[str, List[str]] = {}
    buf: List[str] = []
    etype: Optional[str] = None

    def flush():
        if etype and buf:
            entities.setdefault(etype, []).append(" ".join(buf))

    for token, label in tagged:
        if label == "O":
            flush()
            buf, etype = [], None
        elif label.startswith("B-"):
            flush()
            etype = label[2:]
            buf = [token]
        elif label.startswith("I-"):
            t = label[2:]
            if t == etype:
                buf.append(token)
            else:
                flush()
                etype, buf = t, [token]

    flush()
    return entities


# ── Main Processing Class ────────────────────────────────────────────────────
class UrduNER:
    """
    Inference & training wrapper logic optimized for high efficiency Urdu NER.
    Uses structural token splitting to split attached punctuation natively.
    """

    _DEFAULT_MODEL = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "urdu_ner_crf.joblib",
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path: str = model_path or self._DEFAULT_MODEL
        self._crf = None
        
        # Load automatically if artifact is found in runtime path
        if os.path.exists(self.model_path):
            self._load()

    def _load(self) -> None:
        try:
            self._crf = joblib.load(self.model_path)
            print(f"✅ Model file successfully mapped from: {self.model_path}")
        except Exception as exc:
            raise RuntimeError(f"❌ Unreadable artifact at {self.model_path}: {exc}") from exc

    def tokenize(self, text: str) -> List[str]:
        """Punctuation-aware tokenizer isolating attachments like 'لاہور۔'"""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def fit(self, limit: int = 15000) -> None:
        """
        Memory safe streaming fit implementation. Reads elements directly
        from source data chunks sequentially to safely operate within low RAM bounds.
        """
        import sklearn_crfsuite
        from datasets import load_dataset

        print(f"📥 Commencing streaming pipeline on cheemasohail/Urdu-Legal_ner_corpora...")
        ds = load_dataset("cheemasohail/Urdu-Legal_ner_corpora", split="train", streaming=True)
        
        X_train, y_train = [], []
        processed_count = 0

        for item in ds:
            if 'tokens' in item and 'ner_tags' in item:
                # Isolate specific sample tokens and clean indices
                tokens = item['tokens']
                tags = [str(t) for t in item['ner_tags']]
                
                X_train.append(_sentence_features(tokens))
                y_train.append(tags)
                processed_count += 1
                
            if processed_count >= limit:
                break
        
        print(f"🚀 Initializing L-BFGS Optimization over {processed_count} sequence arrays...")
        self._crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.01,
            max_iterations=60,
            all_possible_transitions=True,
            verbose=False
        )
        self._crf.fit(X_train, y_train)
        
        # Serialize immediately to runtime memory target
        joblib.dump(self._crf, self.model_path)
        print(f"✅ Training step complete. Binary saved cleanly to: {self.model_path}")

    def tag(self, text: str) -> List[Tuple[str, str]]:
        """Process a raw text string context into typed IOB tokens."""
        if not self._crf:
            raise RuntimeError("CRF Core unmapped. Please execute the .fit() method or place weights file.")
        
        tokens = self.tokenize(text)
        if not tokens:
            return []
            
        preds = self._crf.predict([_sentence_features(tokens)])[0]
        return [(t, LABEL_MAP.get(p, p)) for t, p in zip(tokens, preds)]

    def tag_batch(self, texts: List[str]) -> List[List[Tuple[str, str]]]:
        """Process multiple strings through vector pipelines simultaneously."""
        if not self._crf:
            raise RuntimeError("CRF Core unmapped.")
            
        split_texts = [self.tokenize(t) for t in texts]
        active_items = [(idx, tokens) for idx, tokens in enumerate(split_texts) if tokens]

        if not active_items:
            return [[] for _ in texts]

        indices, processing_batches = zip(*active_items)
        features_matrix = [_sentence_features(tokens) for tokens in processing_batches]
        predictions_matrix = self._crf.predict(features_matrix)

        out: List[List[Tuple[str, str]]] = [[] for _ in texts]
        for original_idx, tokens, labels in zip(indices, processing_batches, predictions_matrix):
            out[original_idx] = [(t, LABEL_MAP.get(l, l)) for t, l in zip(tokens, labels)]
        return out

    def get_entities(self, text: str) -> Dict[str, List[str]]:
        return _merge_spans(self.tag(text))

    def get_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        return [_merge_spans(tagged_output) for tagged_output in self.tag_batch(texts)]

    def __repr__(self) -> str:
        return f"UrduNER(mapped_path='{self.model_path}')"


# ── Immediate Verification Runtime Execution Block ────────────────────────────
if __name__ == "__main__":
    # 1. Initialize instance to local current folder execution target
    ner = UrduNER(model_path="urdu_ner_crf.joblib")
    
    # 2. Trigger low-RAM profile extraction model training
    ner.fit(limit=12000)
    
    # 3. Execution Pipeline Sanity Checks
    sample_phrase = "جسٹس فائز عیسی نے لاہور اور اسلام آباد ہائی کورٹ میں سماعت مکمل کی"
    print("\n" + "="*40 + "\n🔎 VERIFICATION TEST RESULTS\n" + "="*40)
    
    # Test flat token mapping output
    print("\n--- Raw Word Tokens Mapping Profile ---")
    for word, tag in ner.tag(sample_phrase):
        print(f"{word:18} -> {tag}")
        
    # Test dictionary entity grouping matrix parsing layout
    print("\n--- Parsed Grouped Structured Entity Map ---")
    print(ner.get_entities(sample_phrase))
    
    # 4. Trigger Automatic Download Mechanism For Your Google Colab Browser Session
    try:
        from google.colab import files
        print("\n📥 Initiating secure binary bridge file download...")
        files.download("urdu_ner_crf.joblib")
    except ImportError:
        print("\n💡 Standalone environment running outside Colab. Output stored to local repository folder.")
