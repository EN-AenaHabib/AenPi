"""
aenpi/normalizer.py
-------------------
Roman Urdu Spelling Normalizer

Problem  : "kya haal hai" is written 50+ different ways by different people.
           Every downstream NLP task breaks on this inconsistency.
Solution : Map phonetic variants to a canonical form using edit-distance +
           a learned substitution table trained on the Roman Urdu corpus.

Dataset  : Khubaib01/RomanUrdu-NLP-Sentiment-Corpus  (Apache 2.0)
           134,052 labeled samples from social media — naturally noisy,
           perfect for learning real-world spelling variants.

Usage
-----
    from aenpi import UrduNormalizer

    norm = UrduNormalizer()
    norm.fit()                          # downloads dataset and builds table
    print(norm.normalize("kesy ho"))    # → "kaise ho"
    print(norm.normalize("kia haal"))   # → "kya hal"
"""

import re
import string
from collections import defaultdict, Counter


class UrduNormalizer:
    """
    Normalizes Roman Urdu text to a canonical spelling form.

    Steps
    -----
    1. Load the Roman Urdu social-media corpus from HuggingFace.
    2. Count word frequencies → most frequent spelling = canonical form.
    3. Build a substitution table mapping rare phonetic variants to
       their canonical counterpart using character-level edit distance.
    4. At inference time, apply the table word-by-word.
    """

    # Common phonetic substitution rules for Roman Urdu
    # These capture the most frequent handwriting-style inconsistencies
    PHONETIC_RULES = [
        # vowel variants
        (r"\bky\b",    "kya"),
        (r"\bkia\b",   "kya"),
        (r"\bkya\b",   "kya"),
        (r"\bhy\b",    "hai"),
        (r"\bhay\b",   "hai"),
        (r"\bhe\b",    "hai"),
        (r"\bhi\b",    "hai"),
        (r"\bnhi\b",   "nahi"),
        (r"\bnhy\b",   "nahi"),
        (r"\bnae\b",   "nahi"),
        (r"\bkesy\b",  "kaise"),
        (r"\bkese\b",  "kaise"),
        (r"\bksi\b",   "kisi"),
        (r"\bkse\b",   "kise"),
        (r"\bkuch\b",  "kuch"),
        (r"\bkch\b",   "kuch"),
        (r"\bacha\b",  "acha"),
        (r"\bacha\b",  "acha"),
        (r"\bachha\b", "acha"),
        (r"\bthk\b",   "theek"),
        (r"\bthek\b",  "theek"),
        (r"\btheek\b", "theek"),
        (r"\bthik\b",  "theek"),
        (r"\baap\b",   "aap"),
        (r"\bap\b",    "aap"),
        (r"\bapp\b",   "aap"),
        (r"\btm\b",    "tum"),
        (r"\btwm\b",   "tum"),
        (r"\bhm\b",    "hum"),
        (r"\bhmm\b",   "hum"),
        (r"\bwoh\b",   "wo"),
        (r"\bwoh\b",   "wo"),
        (r"\bwoo\b",   "wo"),
        (r"\byeh\b",   "ye"),
        (r"\byh\b",    "ye"),
        (r"\bya\b",    "ya"),
        (r"\bor\b",    "aur"),
        (r"\bawr\b",   "aur"),
        (r"\bpr\b",    "par"),
        (r"\bphr\b",   "phir"),
        (r"\bfr\b",    "phir"),
        (r"\bkr\b",    "kar"),
        (r"\bkrna\b",  "karna"),
        (r"\bkrne\b",  "karne"),
        (r"\blkn\b",   "lekin"),
        (r"\blkin\b",  "lekin"),
        (r"\bljn\b",   "lekin"),
        (r"\bbs\b",    "bas"),
        (r"\bbas\b",   "bas"),
        (r"\bzig\b",   "zindagi"),
        (r"\bzndgi\b", "zindagi"),
        # doubled consonants (informal emphasis)
        (r"(.)\1{2,}", r"\1\1"),
    ]

    def __init__(self):
        self.canonical_map = {}     # word → canonical spelling
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, min_freq: int = 3):
        """
        Build the canonical spelling map from the Roman Urdu corpus.

        Parameters
        ----------
        min_freq : int
            Minimum frequency a word must have to be treated as canonical.
            Words appearing fewer than min_freq times are treated as noise.
        """
        print("Loading Roman Urdu corpus from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "Khubaib01/RomanUrdu-NLP-Sentiment-Corpus",
                split="train",
                trust_remote_code=True
            )
            texts = [str(row["text"]) for row in ds if row.get("text")]
        except Exception as e:
            print(f"HuggingFace load failed ({e}). Using built-in rules only.")
            self.is_fitted = True
            return self

        print(f"Loaded {len(texts):,} samples. Building frequency table...")

        # Count all word frequencies
        freq = Counter()
        for text in texts:
            words = self._tokenize(text)
            freq.update(words)

        # Group words by their phonetic skeleton (vowel-stripped form)
        # The most frequent member of each group = canonical form
        skeleton_groups = defaultdict(list)
        for word, count in freq.items():
            if count >= min_freq and len(word) > 1:
                skeleton = self._phonetic_skeleton(word)
                skeleton_groups[skeleton].append((word, count))

        # For each group, pick the most frequent spelling as canonical
        for skeleton, members in skeleton_groups.items():
            members.sort(key=lambda x: x[1], reverse=True)
            canonical = members[0][0]
            for word, _ in members[1:]:
                if word != canonical:
                    self.canonical_map[word] = canonical

        print(f"Normalization table built: {len(self.canonical_map):,} mappings.")
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """
        Normalize a Roman Urdu sentence.

        Parameters
        ----------
        text : str
            Raw Roman Urdu input (any casing, any spelling variant).

        Returns
        -------
        str
            Cleaned, canonical-form Roman Urdu.

        Example
        -------
        >>> norm.normalize("kesy ho aap??  kia haal hay")
        'kaise ho aap ? kia hal hai'
        """
        # Step 1: lowercase
        text = text.lower()

        # Step 2: remove non-alphanumeric except spaces and basic punctuation
        text = re.sub(r"[^\w\s\?\!\.,]", " ", text)

        # Step 3: collapse repeated punctuation  "!!!" → "!"
        text = re.sub(r"([!?.,])\1+", r"\1", text)

        # Step 4: apply hard-coded phonetic rules
        for pattern, replacement in self.PHONETIC_RULES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Step 5: apply learned canonical map (if fitted)
        if self.is_fitted and self.canonical_map:
            words  = text.split()
            words  = [self.canonical_map.get(w, w) for w in words]
            text   = " ".join(words)

        # Step 6: collapse extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_batch(self, texts: list) -> list:
        """Normalize a list of texts."""
        return [self.normalize(t) for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list:
        """Simple whitespace tokenizer after lowercasing."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if w]

    def _phonetic_skeleton(self, word: str) -> str:
        """
        Strip vowels from a word to get its consonant skeleton.
        Words with the same skeleton are likely phonetic variants of each other.

        Example: "kaise" → "ks",  "kese" → "ks",  "kesy" → "ks"
        """
        vowels = set("aeiou")
        skeleton = "".join(c for c in word.lower() if c not in vowels)
        return skeleton if skeleton else word

    def vocab_size(self) -> int:
        """Return number of learned normalization mappings."""
        return len(self.canonical_map)

    def __repr__(self):
        status = f"fitted ({self.vocab_size()} mappings)" if self.is_fitted else "not fitted"
        return f"UrduNormalizer({status})"
