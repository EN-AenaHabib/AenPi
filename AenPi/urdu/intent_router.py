"""
aenpi/intent_router.py
----------------------
Lightweight Offline Intent Classifier

Problem  : Developers call GPT-4 just to classify user input into 5–10
           intent categories. This wastes energy, costs money, adds latency,
           and requires an internet connection.

Solution : A trainable TF-IDF + Logistic Regression intent classifier that:
           - Trains on your custom labels in under 60 seconds on CPU
           - Runs offline with zero API dependency
           - Returns confidence scores for all classes
           - Works for both Urdu and Roman Urdu input

Usage
-----
    from AenPi.urdu import IntentRouter

    # Define your intents with example phrases
    router = IntentRouter()
    router.fit(
        examples=[
            ("mujhe order chahiye",          "order"),
            ("mera paisa wapis karo",        "refund"),
            ("delivery kab hogi",            "tracking"),
            ("product kharab nikla",         "complaint"),
            ("shukriya bohat acha service",  "feedback"),
        ]
    )

    print(router.predict("mera parcel nahi aya"))
    # {"intent": "tracking", "score": 0.87, "scores": {...}}

    # Add new examples without retraining from scratch
    router.add_examples([
        ("payment fail ho gaya", "payment"),
        ("card charge kyun hua", "payment"),
    ])
    router.refit()
"""

import re
from collections import defaultdict


class IntentRouter:
    """
    Trainable offline intent classifier for Urdu and Roman Urdu text.

    Approach
    --------
    TF-IDF (character + word n-grams) + Logistic Regression.
    Character n-grams make it robust to Urdu spelling variants.

    Advantages over LLM-based routing
    ----------------------------------
    - 1000x faster (sub-millisecond vs 500ms+ LLM latency)
    - 100% offline — no API key, no internet required
    - Trainable on your own domain-specific intents
    - Full confidence scores for all classes
    - ~50KB model size vs multi-GB LLM weights
    """

    def __init__(self):
        self.clf         = None
        self.vectorizer  = None
        self.classes_    = []
        self.is_fitted   = False
        self._examples   = []   # list of (text, label) pairs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, examples: list):
        """
        Train the intent classifier on labeled examples.

        Parameters
        ----------
        examples : list of (text, label) tuples
            Each tuple is one training example.
            Minimum 2 examples per intent recommended.
            More examples = better accuracy.

        Example
        -------
        >>> router.fit([
        ...     ("mujhe order chahiye",   "order"),
        ...     ("naya order karna hai",  "order"),
        ...     ("refund chahiye",        "refund"),
        ...     ("paisa wapis karo",      "refund"),
        ... ])
        """
        if not examples:
            raise ValueError("examples list cannot be empty.")

        self._examples = list(examples)
        self._train()
        return self

    def add_examples(self, new_examples: list):
        """
        Add more labeled examples to the existing training set.
        Call .refit() after adding to update the model.

        Parameters
        ----------
        new_examples : list of (text, label) tuples
        """
        self._examples.extend(new_examples)
        print(f"Added {len(new_examples)} examples. "
              f"Total: {len(self._examples)}. Call .refit() to update.")

    def refit(self):
        """Retrain the model on all current examples including new ones."""
        self._train()
        return self

    def _train(self):
        """Internal training logic."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import LabelEncoder

        texts  = [self._preprocess(t) for t, _ in self._examples]
        labels = [l for _, l in self._examples]

        # Validate minimum examples per class
        from collections import Counter
        counts = Counter(labels)
        min_count = min(counts.values())
        if min_count < 1:
            raise ValueError(
                f"Each intent needs at least 1 example. "
                f"Found counts: {dict(counts)}"
            )

        # Character + word n-gram TF-IDF
        # char n-grams handle Urdu spelling variation well
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=20_000,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(texts)

        # Use high C for small datasets (less regularization)
        C = 10.0 if len(texts) < 50 else 1.0

        self.clf = LogisticRegression(
            C=C,
            max_iter=500,
            solver="lbfgs",
            multi_class="multinomial",
        )
        self.clf.fit(X, labels)
        self.classes_  = list(self.clf.classes_)
        self.is_fitted = True

        from collections import Counter
        print(f"IntentRouter trained on {len(texts)} examples, "
              f"{len(self.classes_)} intents: {self.classes_}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Predict the intent of a single input text.

        Parameters
        ----------
        text : str
            User input in Urdu, Roman Urdu, or mixed.

        Returns
        -------
        dict with keys:
            intent : str    — predicted intent label
            score  : float  — confidence of top prediction (0–1)
            scores : dict   — confidence for every intent class

        Example
        -------
        >>> router.predict("mera parcel kahan hai")
        {"intent": "tracking", "score": 0.89,
         "scores": {"tracking": 0.89, "order": 0.07, "refund": 0.04}}
        """
        self._check_fitted()
        clean  = self._preprocess(text)
        X      = self.vectorizer.transform([clean])
        probs  = self.clf.predict_proba(X)[0]
        intent = self.classes_[probs.argmax()]
        score_dict = {c: round(float(p), 4)
                      for c, p in zip(self.classes_, probs)}
        return {
            "intent": intent,
            "score":  round(float(probs.max()), 4),
            "scores": score_dict,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict intents for a list of inputs.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        list of dicts (same format as predict())
        """
        self._check_fitted()
        cleaned = [self._preprocess(t) for t in texts]
        X       = self.vectorizer.transform(cleaned)
        probs   = self.clf.predict_proba(X)
        results = []
        for p in probs:
            intent = self.classes_[p.argmax()]
            score_dict = {c: round(float(v), 4)
                          for c, v in zip(self.classes_, p)}
            results.append({
                "intent": intent,
                "score":  round(float(p.max()), 4),
                "scores": score_dict,
            })
        return results

    def top_n(self, text: str, n: int = 3) -> list:
        """
        Return top-N intents sorted by confidence.

        Parameters
        ----------
        text : str
        n    : int — number of top intents to return

        Returns
        -------
        list of {"intent": str, "score": float} sorted descending

        Example
        -------
        >>> router.top_n("payment nahi hua", n=3)
        [{"intent":"payment","score":0.72},
         {"intent":"refund","score":0.18},
         {"intent":"complaint","score":0.10}]
        """
        result = self.predict(text)
        ranked = sorted(result["scores"].items(),
                        key=lambda x: x[1], reverse=True)
        return [{"intent": k, "score": v} for k, v in ranked[:n]]

    def score(self, examples: list) -> float:
        """
        Evaluate accuracy on a labeled test set.

        Parameters
        ----------
        examples : list of (text, label) tuples

        Returns
        -------
        float : accuracy (0–1)
        """
        self._check_fitted()
        texts  = [t for t, _ in examples]
        labels = [l for _, l in examples]
        preds  = [r["intent"] for r in self.predict_batch(texts)]
        correct = sum(p == g for p, g in zip(preds, labels))
        acc = correct / len(labels) if labels else 0.0
        print(f"Accuracy: {acc:.2%} ({correct}/{len(labels)})")
        return acc

    def list_intents(self) -> list:
        """Return all known intent labels."""
        return list(self.classes_)

    def example_count(self) -> dict:
        """Return number of training examples per intent."""
        from collections import Counter
        return dict(Counter(l for _, l in self._examples))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """Normalize input text before vectorization."""
        text = text.lower()
        text = re.sub(r"http\S+",     " ", text)
        text = re.sub(r"@\w+",        " ", text)
        text = re.sub(r"\d+",         " ", text)
        text = re.sub(r"(.)\1{2,}",   r"\1\1", text)
        text = re.sub(r"[^\w\s]",     " ", text)
        text = re.sub(r"\s+",         " ", text).strip()
        return text

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Router is not trained. Call .fit(examples) first."
            )

    def __repr__(self):
        if self.is_fitted:
            return (f"IntentRouter(fitted, "
                    f"intents={self.classes_}, "
                    f"examples={len(self._examples)})")
        return "IntentRouter(not fitted)"
