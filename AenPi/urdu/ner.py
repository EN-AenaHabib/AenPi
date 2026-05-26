"""
aenpi/ner.py
------------
Urdu Named Entity Recognition (NER)

Problem  : Urdu NER requires heavy multilingual models (mBERT, XLM-R)
           that use 270M+ parameters for a task that can be done with a
           tiny CRF or rule-enhanced model at a fraction of the cost.

Solution : Feature-rich CRF (Conditional Random Field) trained on the
           MK-PUCIT Urdu NER corpus via the mirfan899/Urdu GitHub collection.
           Falls back to a rule-based tagger when CRF training data is
           unavailable.

Dataset  : mirfan899/Urdu (GitHub collection, publicly available)
           Tags: PERSON, LOCATION, ORGANIZATION, DATE, NUMBER,
                 DESIGNATION, TIME, O (outside)

Usage
-----
    from aenpi import UrduNER

    ner = UrduNER()
    ner.fit()

    result = ner.tag("Ali Ahmed ne Lahore mein kaam kiya")
    # [("Ali","B-PERSON"),("Ahmed","I-PERSON"),("ne","O"),
    #  ("Lahore","B-LOCATION"),("mein","O"),
    #  ("kaam","O"),("kiya","O")]

    entities = ner.entities("Ali Ahmed ne Lahore mein kaam kiya")
    # [{"text":"Ali Ahmed","label":"PERSON"},
    #  {"text":"Lahore","label":"LOCATION"}]
"""

import re


# ---------------------------------------------------------------------------
# Built-in gazetteers (seed lists for rule-based fallback)
# These are well-known Urdu/Pakistani proper nouns
# ---------------------------------------------------------------------------

_PERSON_SEEDS = {
    "ali","ahmed","muhammad","usman","hassan","hussain","fatima","aisha",
    "zara","sara","imran","nawaz","asif","bilal","hamza","omar","tariq",
    "adnan","kamran","salman","amjad","khalid","nadeem","sajid","wasim",
    "arif","iqbal","akram","anwar","farooq","rashid","saeed","jameel",
    "rehman","qasim","babar","azam","shoaib","shahid","younis","waqar",
    "zahid","javaid","nasir","tahir","liaquat","zulfiqar","benazir","maryam",
    "shehbaz","asad","fawad","hina","shireen","pervaiz","chaudhry",
}

_LOCATION_SEEDS = {
    "lahore","karachi","islamabad","peshawar","quetta","multan","faisalabad",
    "rawalpindi","sialkot","gujranwala","hyderabad","sukkur","larkana",
    "abbottabad","mansehra","swat","chitral","gilgit","skardu","mirpur",
    "muzaffarabad","pakistan","india","china","america","london","dubai",
    "saudi","arabia","iran","afghanistan","bangladesh","sindh","punjab",
    "balochistan","kpk","kashmir","azad","frontier","tribal","fata",
}

_ORG_SEEDS = {
    "ptv","geo","ary","dawn","jang","express","nation","tribune",
    "pemra","nadra","fia","isi","army","navy","airforce","police",
    "judiciary","parliament","senate","assembly","government","ministry",
    "pta","ogra","nepra","secp","sbp","hec","pmdc","pec","bia",
    "pml","ppp","pti","mqm","anp","jui","jamaat","tehreek","league",
    "university","college","school","hospital","bank","court",
}

_DATE_PATTERNS = [
    r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",   # 12/03/2024
    r"\b\d{4}\b",                                 # 2024
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    r"\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(muharram|safar|rabi|jumada|rajab|shaban|ramadan|"
    r"shawwal|zilhaj|zilqad)\b",
    r"\b(somwar|mangal|budh|jumeraat|juma|hafta|itwar)\b",
]

_NUM_PATTERNS = [
    r"\b\d[\d,\.]*\b",
    r"\b(ek|do|teen|char|panch|che|saat|aath|nau|das|"
    r"bees|tees|chalis|pachas|saath|assi|nabbe|sau|hazar|lakh|crore)\b",
]


class UrduNER:
    """
    Lightweight Named Entity Recognizer for Urdu text (Nastaliq or Roman).

    Entity Types
    ------------
    PERSON       : People's names
    LOCATION     : Places, cities, countries
    ORGANIZATION : Companies, parties, institutions
    DATE         : Dates and time expressions
    NUMBER       : Numbers and quantities
    O            : Not an entity

    Approach
    --------
    1. Gazetteer lookup for known entities (fast, high precision).
    2. Pattern matching for dates and numbers.
    3. CRF model trained on MK-PUCIT corpus (when sklearn-crfsuite available).
    4. Capitalisation and context heuristics as fallback.
    """

    def __init__(self):
        self.crf        = None
        self.is_fitted  = False
        self._persons   = set(s.lower() for s in _PERSON_SEEDS)
        self._locations = set(s.lower() for s in _LOCATION_SEEDS)
        self._orgs      = set(s.lower() for s in _ORG_SEEDS)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self):
        """
        Train a CRF model on Urdu NER data.
        If sklearn-crfsuite is not installed, falls back to rule-based only.
        """
        try:
            import sklearn_crfsuite
            self._train_crf()
        except ImportError:
            print(
                "sklearn-crfsuite not found. Running in rule-only mode.\n"
                "For CRF mode: pip install sklearn-crfsuite"
            )
            self.is_fitted = True
        return self

    def _train_crf(self):
        """Download and train CRF on Urdu NER data."""
        import sklearn_crfsuite
        print("Loading Urdu NER dataset...")
        try:
            import requests, io
            # MK-PUCIT NER data via mirfan899's collection (public GitHub)
            url = ("https://raw.githubusercontent.com/mirfan899/Urdu/"
                   "master/ner/ner.txt")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
        except Exception as e:
            print(f"NER data download failed ({e}). Using rule-only mode.")
            self.is_fitted = True
            return

        # Parse CoNLL format:  word\tPOS\tNER_tag
        sentences, current = [], []
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag  = parts[-1]
                    current.append((word, tag))
        if current:
            sentences.append(current)

        if not sentences:
            print("Could not parse NER data. Using rule-only mode.")
            self.is_fitted = True
            return

        print(f"Training CRF on {len(sentences):,} sentences...")
        X = [self._sent2features(s) for s in sentences]
        y = [self._sent2labels(s)   for s in sentences]

        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        self.crf.fit(X, y)
        self.is_fitted = True
        print("Urdu CRF NER model trained.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def tag(self, text: str) -> list:
        """
        Tag each token with a BIO NER label.

        Parameters
        ----------
        text : str
            Input sentence in Roman Urdu or Nastaliq Urdu.

        Returns
        -------
        list of (token, tag) tuples
            tag format: "B-TYPE", "I-TYPE", or "O"

        Example
        -------
        >>> ner.tag("Ali Ahmed ne Lahore mein kaam kiya")
        [("Ali","B-PERSON"),("Ahmed","I-PERSON"),("ne","O"),
         ("Lahore","B-LOCATION"),("mein","O"),("kaam","O"),("kiya","O")]
        """
        self._check_fitted()
        tokens = text.split()
        if not tokens:
            return []

        if self.crf is not None:
            # Use trained CRF
            features = self._sent2features([(t, "O") for t in tokens])
            tags     = self.crf.predict([features])[0]
            return list(zip(tokens, tags))
        else:
            # Rule-based fallback
            return self._rule_tag(tokens)

    def entities(self, text: str) -> list:
        """
        Extract named entities as a clean list.

        Returns
        -------
        list of dicts:
            {"text": str, "label": str, "start": int, "end": int}
            start/end are token indices.

        Example
        -------
        >>> ner.entities("Ali Ahmed ne Lahore mein kaam kiya")
        [{"text":"Ali Ahmed","label":"PERSON","start":0,"end":1},
         {"text":"Lahore","label":"LOCATION","start":3,"end":3}]
        """
        tagged  = self.tag(text)
        result  = []
        i       = 0

        while i < len(tagged):
            tok, tag = tagged[i]
            if tag.startswith("B-"):
                entity_type  = tag[2:]
                entity_words = [tok]
                j = i + 1
                while j < len(tagged) and tagged[j][1] == f"I-{entity_type}":
                    entity_words.append(tagged[j][0])
                    j += 1
                result.append({
                    "text":  " ".join(entity_words),
                    "label": entity_type,
                    "start": i,
                    "end":   j - 1,
                })
                i = j
            else:
                i += 1
        return result

    def entity_types(self) -> list:
        """Return list of supported entity types."""
        return ["PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER"]

    # ------------------------------------------------------------------
    # Rule-based tagger (fallback)
    # ------------------------------------------------------------------

    def _rule_tag(self, tokens: list) -> list:
        """Simple gazetteer + pattern based NER."""
        tags   = ["O"] * len(tokens)
        lower  = [t.lower() for t in tokens]

        # Dates and numbers via regex
        full_text_lower = " ".join(lower)
        for pat in _DATE_PATTERNS:
            for m in re.finditer(pat, full_text_lower, re.IGNORECASE):
                word = m.group()
                for i, tok in enumerate(lower):
                    if tok == word and tags[i] == "O":
                        tags[i] = "B-DATE"

        for pat in _NUM_PATTERNS:
            for m in re.finditer(pat, full_text_lower, re.IGNORECASE):
                word = m.group()
                for i, tok in enumerate(lower):
                    if tok == word and tags[i] == "O":
                        tags[i] = "B-NUMBER"

        # Gazetteer lookup
        i = 0
        while i < len(lower):
            # Try bigram first (e.g. "Ali Ahmed")
            if i + 1 < len(lower):
                bigram = lower[i] + " " + lower[i + 1]
                if bigram.split()[0] in self._persons and bigram.split()[1] in self._persons:
                    if tags[i] == "O":
                        tags[i]     = "B-PERSON"
                        tags[i + 1] = "I-PERSON"
                        i += 2
                        continue
            # Unigram
            if lower[i] in self._persons and tags[i] == "O":
                tags[i] = "B-PERSON"
            elif lower[i] in self._locations and tags[i] == "O":
                tags[i] = "B-LOCATION"
            elif lower[i] in self._orgs and tags[i] == "O":
                tags[i] = "B-ORGANIZATION"
            i += 1

        return list(zip(tokens, tags))

    # ------------------------------------------------------------------
    # CRF feature extraction
    # ------------------------------------------------------------------

    def _word2features(self, sent, i):
        word = sent[i][0]
        low  = word.lower()
        feats = {
            "bias":          1.0,
            "word.lower":    low,
            "word[-3:]":     low[-3:],
            "word[-2:]":     low[-2:],
            "word[:3]":      low[:3],
            "word.len":      len(word),
            "is_upper":      word.isupper(),
            "is_title":      word.istitle(),
            "is_digit":      word.isdigit(),
            "in_person":     low in self._persons,
            "in_location":   low in self._locations,
            "in_org":        low in self._orgs,
        }
        if i > 0:
            pw = sent[i - 1][0].lower()
            feats["prev.word"]   = pw
            feats["prev[-2:]"]   = pw[-2:]
        else:
            feats["BOS"] = True

        if i < len(sent) - 1:
            nw = sent[i + 1][0].lower()
            feats["next.word"]   = nw
            feats["next[-2:]"]   = nw[-2:]
        else:
            feats["EOS"] = True

        return feats

    def _sent2features(self, sent):
        return [self._word2features(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        return [label for _, label in sent]

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

    def __repr__(self):
        mode = "CRF" if self.crf else "rule-based"
        status = "fitted" if self.is_fitted else "not fitted"
        return f"UrduNER(mode={mode}, {status})"
