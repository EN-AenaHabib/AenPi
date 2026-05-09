"""
AenPi.urdu.ngram
-----------------
N-gram language model for Urdu next-word prediction.
Trains on a list of sentences and predicts the most likely next word(s).
"""

from collections import defaultdict, Counter


class NGramPredictor:
    """
    N-gram based next-word predictor for Urdu text.

    Supports bigram (n=2) and trigram (n=3) models.
    No external libraries needed.

    Example:
        >>> sentences = ["میرا نام احمد ہے", "میرا گھر لاہور میں ہے"]
        >>> model = NGramPredictor(n=2)
        >>> model.train(sentences)
        >>> model.predict("میرا")
        [('نام', 1), ('گھر', 1)]
    """

    def __init__(self, n: int = 2):
        """
        Args:
            n (int): N-gram order. 2 = bigram, 3 = trigram.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self._model = defaultdict(Counter)
        self._trained = False

    def train(self, sentences: list):
        """
        Build n-gram model from a list of Urdu sentences.

        Args:
            sentences (list): List of whitespace-tokenized Urdu strings.

        Example:
            >>> model.train(["میرا نام علی ہے", "میرا گھر یہاں ہے"])
        """
        for sentence in sentences:
            tokens = sentence.split() if isinstance(sentence, str) else sentence
            if len(tokens) < self.n:
                continue
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i: i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self._model[context][next_word] += 1

        self._trained = True

    def predict(self, context_text: str, top_k: int = 3) -> list:
        """
        Predict the top-k most likely next words given context.

        Args:
            context_text (str): The preceding word(s) as a string.
                                 Uses the last (n-1) tokens as context.
            top_k (int): Number of predictions to return.

        Returns:
            list: List of (word, count) tuples sorted by frequency.
                  Returns empty list if context not seen during training.

        Example:
            >>> model.predict("میرا نام", top_k=2)
            [('احمد', 3), ('علی', 1)]
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        tokens = context_text.split()
        context = tuple(tokens[-(self.n - 1):])

        if context not in self._model:
            return []

        return self._model[context].most_common(top_k)

    def get_all_contexts(self) -> list:
        """Return all seen contexts (useful for debugging)."""
        return list(self._model.keys())

    def vocabulary(self) -> set:
        """Return all unique words seen during training."""
        words = set()
        for context, counter in self._model.items():
            words.update(context)
            words.update(counter.keys())
        return words

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return f"NGramPredictor(n={self.n}, status={status}, contexts={len(self._model)})"


# ─── Convenience function ─────────────────────────────────────────────────────

def ngram_predict(context_text: str, corpus: list, n: int = 2, top_k: int = 3) -> list:
    """
    One-shot n-gram prediction: train and predict in one call.

    Args:
        context_text (str): Context string for prediction.
        corpus (list): List of training sentences.
        n (int): N-gram order (default 2).
        top_k (int): Number of predictions to return.

    Returns:
        list: List of (word, count) tuples.

    Example:
        >>> corpus = ["میرا نام احمد ہے", "میرا نام علی ہے"]
        >>> ngram_predict("میرا", corpus, n=2)
        [('نام', 2)]
    """
    model = NGramPredictor(n=n)
    model.train(corpus)
    return model.predict(context_text, top_k=top_k)
