"""
aenpi/summarizer.py
-------------------
Urdu Extractive Micro-Summarizer

Problem  : Summarization for Urdu almost always uses a generative LLM,
           which hallucinates, wastes energy, and requires an API call.
           Most production use cases only need the KEY sentences, not
           rewritten prose.

Solution : Score each sentence by three signals:
           1. TF-IDF information density
           2. Position weight (first/last sentences matter more in Urdu news)
           3. Named-entity overlap (sentences with more entities rank higher)
           Return top-N sentences in original order with scores.

Dataset  : mirfan899/usummary (HuggingFace) — Urdu news summaries
           Used to tune position weights and validate output quality.
"""

import re
import math
from collections import Counter


class UrduSummarizer:
    """
    Extractive summarizer for Urdu news and articles.

    Returns the most important sentences with interpretable scores.
    No LLM. No generation. No hallucinations by design.

    Score components
    ----------------
    tf_idf_score   : How information-dense is this sentence?
    position_score : First and last sentences are usually most important.
    entity_score   : Sentences mentioning people/places rank higher.
    final_score    : Weighted combination of the three.
    """

    def __init__(self,
                 tfidf_weight:    float = 0.5,
                 position_weight: float = 0.3,
                 entity_weight:   float = 0.2):
        self.tfidf_weight    = tfidf_weight
        self.position_weight = position_weight
        self.entity_weight   = entity_weight
        self.idf_table       = {}
        self.is_fitted       = False

    # ------------------------------------------------------------------
    # Training (builds IDF table from Urdu news corpus)
    # ------------------------------------------------------------------

    def fit(self):
        """
        Build IDF table from Urdu news corpus.
        This improves TF-IDF scoring for Urdu-specific vocabulary.
        """
        print("Loading Urdu news summary dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("mirfan899/usummary", split="train",
                             trust_remote_code=True)
            corpus_texts = []
            for row in ds:
                for key in ("article", "text", "document", "body"):
                    if row.get(key):
                        corpus_texts.append(str(row[key]))
                        break
        except Exception as e:
            print(f"Dataset load failed ({e}). Using term-frequency only.")
            self.is_fitted = True
            return self

        print(f"Building IDF table from {len(corpus_texts):,} documents...")
        self.idf_table = self._build_idf(corpus_texts)
        self.is_fitted = True
        print(f"IDF table built: {len(self.idf_table):,} terms.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def summarize(self, text: str, n: int = 3) -> list:
        """Extract the $N$ most important sentences from Urdu text."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        n = min(n, len(sentences))

        total = len(sentences)
        scored = []

        # Build document-level TF for IDF scoring
        all_words = [w for s in sentences for w in self._tokenize(s)]
        doc_tf    = Counter(all_words)
        doc_len   = max(len(all_words), 1)

        for idx, sent in enumerate(sentences):
            words          = self._tokenize(sent)
            tf_idf_score   = self._tfidf_score(words, doc_tf, doc_len)
            position_score = self._position_score(idx, total)
            entity_score   = self._entity_score(sent)

            final = (
                self.tfidf_weight    * tf_idf_score  +
                self.position_weight * position_score +
                self.entity_weight   * entity_score
            )

            scored.append({
                "sentence":       sent.strip(),
                "position":       idx,
                "score":          round(final, 4),
                "tf_idf_score":   round(tf_idf_score,   4),
                "position_score": round(position_score, 4),
                "entity_score":   round(entity_score,   4),
            })

        # Sort by score, pick top-N
        top = sorted(scored, key=lambda x: x["score"], reverse=True)[:n]

        # Return in original document order
        top.sort(key=lambda x: x["position"])
        return top

    def summarize_text(self, text: str, n: int = 3) -> str:
        """Return the top-$N$ sentences joined as a single string."""
        results = self.summarize(text, n=n)
        return " ".join(r["sentence"] for r in results)

    def keyword_summary(self, text: str, top_k: int = 10) -> list:
        """Return the top-$K$ most important words (keywords) from the text."""
        words  = self._tokenize(text)
        tf     = Counter(words)
        length = max(len(words), 1)
        scores = {}
        for word in tf:
            tfidf = self._word_tfidf(word, tf[word], length)
            scores[word] = tfidf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _tfidf_score(self, words: list, doc_tf: Counter, doc_len: int) -> float:
        if not words:
            return 0.0
        total = sum(self._word_tfidf(w, doc_tf[w], doc_len) for w in words)
        return total / len(words)

    def _word_tfidf(self, word: str, tf_count: int, doc_len: int) -> float:
        tf  = tf_count / doc_len
        idf = self.idf_table.get(word, math.log(10))
        return tf * idf

    def _position_score(self, idx: int, total: int) -> float:
        if total == 1:
            return 1.0
        if idx == 0:
            return 1.0
        if idx == total - 1:
            return 0.8
        rel = idx / (total - 1)
        return 1.0 - 0.6 * rel

    def _entity_score(self, sentence: str) -> float:
        """
        Reward sentences containing proper nouns/entities.
        Uses UrduNER sequence tagger dynamically if available, with a 
        safe capitalization fallback for Romanized Urdu text.
        """
        entity_count = 0
        words = sentence.split()
        if not words:
            return 0.0

        # Clean cross-module integration with UrduNER
        try:
            from .ner import UrduNER
            ner = UrduNER()
            if ner._crf is not None:
                # Count discovered entities from our CRF pipeline
                found_entities = ner.get_entities(sentence)
                entity_count += len(found_entities)
        except Exception:
            pass  # Fail gracefully if NER model file isn't generated yet

        # Capitalization heuristic (crucial context signal for Roman Urdu proper nouns)
        for w in words:
            if w and w[0].isupper():
                entity_count += 0.5

        return min(entity_count / len(words) * 3, 1.0)

    # ------------------------------------------------------------------
    # IDF builder & Text helpers
    # ------------------------------------------------------------------

    def _build_idf(self, documents: list) -> dict:
        N   = len(documents)
        df  = Counter()
        for doc in documents:
            words = set(self._tokenize(doc))
            df.update(words)
        idf = {}
        for word, count in df.items():
            idf[word] = math.log(N / (count + 1)) + 1
        return idf

    def _split_sentences(self, text: str) -> list:
        text = re.sub(r"([۔؟!\.\?])", r"\1\n", text)
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        return [s for s in sentences if len(s.split()) >= 3]

    def _tokenize(self, text: str) -> list:
        _UR_STOPWORDS = {
            "hai","hain","tha","thi","the","ko","ka","ki","ke","se",
            "mein","ne","par","aur","ya","kya","nahi","agar","lekin",
            "phir","toh","bhi","hi","sirf","bas","aap","tum","hum",
            "wo","ye","yeh","woh","iska","uska","apna","mera","tera",
            "tak","pe","jo","jab","jahan","jaise","kyun","kaun","kaise",
            "sab","kuch","yahan","wahan","ab","abhi","pehle","baad",
            "is","are","was","were","to","of","in","on","at","by","for"
        }
        text  = text.lower()
        text  = re.sub(r"[^\w\s]", " ", text)
        words = text.split()
        return [w for w in words if w not in _UR_STOPWORDS and len(w) > 1]

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"UrduSummarizer(tfidf={self.tfidf_weight}, "
                f"position={self.position_weight}, "
                f"entity={self.entity_weight}, {status})")
