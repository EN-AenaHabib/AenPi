"""
aenpi/code_switch.py
--------------------
Token-level Urdu / English Code-Switch Detector

Problem  : Pakistani social media mixes Urdu and English mid-sentence.
           "Mujhe yeh project deadline tak finish karna hai."
           No lightweight offline tool labels each token as UR / EN / MIX.

Solution : Character-trigram features + Logistic Regression.
           Trains in seconds on CPU. Returns structured span output.

Dataset  : community-datasets/roman_urdu  (HuggingFace, Apache 2.0)
           + english words bootstrapped from a built-in frequency list

Usage
-----
    from aenpi import CodeSwitchDetector

    detector = CodeSwitchDetector()
    detector.fit()

    result = detector.detect("mujhe yeh project finish karna hai")
    print(result)
    # [("mujhe","UR"),("yeh","UR"),("project","EN"),
    #  ("finish","EN"),("karna","UR"),("hai","UR")]

    spans = detector.spans("mujhe yeh project finish karna hai")
    # [{"text":"mujhe yeh","lang":"UR","start":0,"end":1},
    #  {"text":"project finish","lang":"EN","start":2,"end":3}, ...]
"""

import re
import pickle
import os
from collections import Counter


# ---------------------------------------------------------------------------
# Built-in English seed vocabulary (top-500 common English words that appear
# in code-switched Roman Urdu text).  No external download needed.
# ---------------------------------------------------------------------------
_EN_SEED = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","shall","should","may","might",
    "must","can","could","i","you","he","she","it","we","they","me","him",
    "her","us","them","my","your","his","its","our","their","this","that",
    "these","those","and","but","or","nor","for","yet","so","in","on","at",
    "to","of","with","by","from","up","about","into","through","during",
    "before","after","above","below","between","out","off","over","under",
    "again","further","then","once","here","there","when","where","why",
    "how","all","both","each","few","more","most","other","some","such",
    "no","not","only","same","than","too","very","just","now","also",
    "project","work","job","time","day","year","way","man","woman","child",
    "thing","life","hand","part","place","case","week","company","system",
    "program","question","government","number","night","point","home","water",
    "room","mother","area","money","story","fact","month","lot","right",
    "study","book","eye","job","word","business","issue","side","kind",
    "head","house","service","friend","father","power","hour","game","line",
    "end","among","never","last","long","little","own","old","right","big",
    "high","different","small","large","next","early","young","important",
    "public","private","real","best","free","able","need","want","seem",
    "feel","try","leave","call","keep","let","begin","show","hear","play",
    "run","move","live","believe","hold","bring","happen","write","provide",
    "sit","stand","lose","pay","meet","include","continue","set","learn",
    "change","lead","understand","watch","follow","stop","create","speak",
    "read","spend","grow","open","walk","win","offer","remember","love",
    "consider","appear","buy","wait","serve","die","send","expect","build",
    "stay","fall","cut","reach","kill","remain","suggest","raise","pass",
    "sell","require","report","decide","pull","finish","deadline","complete",
    "start","end","process","result","plan","level","area","field","type",
    "post","list","name","form","data","page","code","file","test","team",
    "user","account","email","phone","online","website","internet","social",
    "media","video","photo","music","film","app","mobile","laptop","computer",
    "software","hardware","network","server","database","api","model","class",
}

# ---------------------------------------------------------------------------
# Urdu-specific Roman spellings that are never English
# ---------------------------------------------------------------------------
_UR_SEED = {
    "hai","hain","tha","thi","the","ko","ka","ki","ke","se","me","mein",
    "ne","par","aur","ya","kya","kia","nahi","nhi","agar","lekin","lkn",
    "phir","phr","toh","to","bhi","hi","sirf","bas","abhi","ab","kab",
    "kahan","kyun","kaun","kaise","kitna","sab","kuch","yahan","wahan",
    "aap","tum","hum","mein","wo","ye","yeh","woh","iska","uska","inki",
    "unki","apna","apni","mera","tera","hamara","tumhara","unka","yaar",
    "bhai","behen","amma","abba","dost","ghar","zindagi","dil","pyar",
    "karo","karna","karte","karein","hoga","hogi","honge","chahiye",
    "theek","acha","achha","bura","zyada","thoda","bohat","bohot","bahut",
    "bilkul","zaroor","shayad","lagta","lagti","samjha","pata","maloom",
    "mazay","maza","khush","dukhi","pareshan","thaka","seedha","zyada",
}


class CodeSwitchDetector:
    """
    Token-level language identifier for Urdu–English code-switched text.

    Labels
    ------
    "UR"  : Urdu (Roman script)
    "EN"  : English
    "MIX" : Ambiguous / mixed

    Approach
    --------
    1. Hard-coded seed lexicons for both languages.
    2. Character trigram overlap with each seed vocabulary.
    3. Script heuristics (digits, English-only letter combos).
    4. Trained logistic regression on labeled Roman Urdu corpus.
    """

    def __init__(self):
        self.clf        = None
        self.is_fitted  = False
        self._en_vocab  = set(_EN_SEED)
        self._ur_vocab  = set(_UR_SEED)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train on the Roman Urdu sentiment corpus.
        We derive pseudo-labels: words in English seed → EN,
        words in Urdu seed → UR, then train a trigram LR classifier.
        """
        print("Training CodeSwitchDetector...")
        try:
            from datasets import load_dataset
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import HashingVectorizer

            ds = load_dataset(
                "Khubaib01/RomanUrdu-NLP-Sentiment-Corpus",
                split="train",
                trust_remote_code=True
            )
            texts = [str(row["text"]).lower() for row in ds if row.get("text")]
        except Exception as e:
            print(f"Dataset load failed ({e}). Running in rule-only mode.")
            self.is_fitted = True
            return self

        # Build training data word-by-word
        X_words, y_labels = [], []
        for text in texts[:30000]:           # 30k sentences is enough
            for word in text.split():
                word = re.sub(r"[^\w]", "", word)
                if len(word) < 2:
                    continue
                label = self._seed_label(word)
                if label != "MIX":
                    X_words.append(word)
                    y_labels.append(label)

        if len(X_words) < 100:
            print("Not enough labeled words. Using rule-only mode.")
            self.is_fitted = True
            return self

        print(f"Training on {len(X_words):,} labeled tokens...")

        # Trigram character vectorizer — captures spelling patterns
        self._vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=2 ** 14,
            alternate_sign=False,
        )
        X = self._vectorizer.transform(X_words)

        self.clf = LogisticRegression(
            max_iter=200,
            C=1.0,
            solver="lbfgs",
            multi_class="auto",
        )
        self.clf.fit(X, y_labels)
        self.is_fitted = True
        print("CodeSwitchDetector trained.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list:
        """
        Label each token in the text.

        Parameters
        ----------
        text : str
            Raw input sentence (Roman Urdu / English mix).

        Returns
        -------
        list of (token, label) tuples
            label is "UR", "EN", or "MIX"

        Example
        -------
        >>> detector.detect("mujhe yeh project finish karna hai")
        [('mujhe','UR'),('yeh','UR'),('project','EN'),
         ('finish','EN'),('karna','UR'),('hai','UR')]
        """
        tokens = text.lower().split()
        results = []
        for tok in tokens:
            clean = re.sub(r"[^\w]", "", tok)
            if not clean:
                results.append((tok, "MIX"))
                continue
            label = self._classify_token(clean)
            results.append((tok, label))
        return results

    def spans(self, text: str) -> list:
        """
        Return contiguous spans of same-language tokens.

        Returns
        -------
        list of dicts:
            {"text": str, "lang": str, "start": int, "end": int}
            start/end are token indices (not character offsets).

        Example
        -------
        >>> detector.spans("mujhe yeh project finish karna hai")
        [{"text":"mujhe yeh","lang":"UR","start":0,"end":1},
         {"text":"project finish","lang":"EN","start":2,"end":3},
         {"text":"karna hai","lang":"UR","start":4,"end":5}]
        """
        labeled = self.detect(text)
        if not labeled:
            return []

        spans = []
        current_lang  = labeled[0][1]
        current_words = [labeled[0][0]]
        start_idx     = 0

        for i, (tok, lang) in enumerate(labeled[1:], start=1):
            if lang == current_lang:
                current_words.append(tok)
            else:
                spans.append({
                    "text":  " ".join(current_words),
                    "lang":  current_lang,
                    "start": start_idx,
                    "end":   i - 1,
                })
                current_lang  = lang
                current_words = [tok]
                start_idx     = i

        spans.append({
            "text":  " ".join(current_words),
            "lang":  current_lang,
            "start": start_idx,
            "end":   len(labeled) - 1,
        })
        return spans

    def switch_points(self, text: str) -> list:
        """
        Return the token indices where the language switches.

        Example
        -------
        >>> detector.switch_points("mujhe yeh project finish karna")
        [2, 4]   # switches at token 2 (project) and token 4 (karna)
        """
        labeled = self.detect(text)
        points  = []
        for i in range(1, len(labeled)):
            if labeled[i][1] != labeled[i - 1][1]:
                points.append(i)
        return points

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_label(self, word: str) -> str:
        """Quick label from seed sets."""
        if word in self._ur_vocab:
            return "UR"
        if word in self._en_vocab:
            return "EN"
        return "MIX"

    def _classify_token(self, word: str) -> str:
        """Classify a single clean token."""
        # 1. Direct seed lookup first (fast, high confidence)
        seed = self._seed_label(word)
        if seed != "MIX":
            return seed

        # 2. Classifier if trained
        if self.clf is not None:
            vec = self._vectorizer.transform([word])
            return self.clf.predict(vec)[0]

        # 3. Heuristic fallback
        return self._heuristic(word)

    def _heuristic(self, word: str) -> str:
        """
        Rule-based fallback when no classifier is available.
        - Words ending in common Urdu suffixes → UR
        - Words matching common English patterns → EN
        """
        ur_suffixes = ("na", "ni", "ne", "ta", "ti", "te", "ga", "gi",
                       "kar", "ke", "ko", "ka", "ki", "se", "mein")
        en_suffixes = ("tion", "ing", "ness", "ment", "ful", "less",
                       "ize", "ise", "ous", "ive", "ible", "able",
                       "ly", "ed", "er", "est")

        if word.endswith(ur_suffixes):
            return "UR"
        if word.endswith(en_suffixes):
            return "EN"
        # If mostly consonants with no typical Urdu combos → EN
        vowel_ratio = sum(1 for c in word if c in "aeiou") / max(len(word), 1)
        return "EN" if vowel_ratio < 0.2 else "MIX"

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"CodeSwitchDetector({status})"
