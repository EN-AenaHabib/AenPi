"""
aenpi/pipeline.py
-----------------
The unified entry point for AenPi (proposal section 7).

One call chains any combination of the library's modules and returns a single
``Doc`` object carrying every annotation:

    from AenPi.urdu import Pipeline

    nlp = Pipeline(["normalize", "tokenize", "pos", "ner", "sentiment"])
    doc = nlp("Ali ne Lahore mein bohat acha kaam kiya")

    doc.tokens       # ['ali', 'ne', 'lahore', 'mein', 'bohat', 'acha', ...]
    doc.entities     # [{'text': 'Lahore', 'label': 'LOCATION'}, ...]
    doc.sentiment    # 'Positive'
    doc.language_spans
    doc.reduplications

Design goals
------------
* **Robust by default.** Building a stage that is missing (module not written
  yet, e.g. ``pos``) or whose dependency is unavailable (e.g. ``datasets`` for
  ``sentiment``) does NOT crash the pipeline — that stage is skipped with a
  warning and the rest still runs. This keeps the demo working on any machine.
* **Train once.** Trainable models are fitted when the pipeline is built, then
  reused for every call.
* **Order preserving.** Stages run in the exact order you list them.
"""

import warnings


# Every stage the pipeline knows how to build. Listing a stage that is not
# here raises immediately (a clear user typo); listing one that is here but
# whose module/deps are missing is skipped gracefully at build time.
AVAILABLE_STAGES = (
    "normalize",
    "tokenize",
    "stopwords",
    "stem",
    "pos",
    "ner",
    "sentiment",
    "codemix",
    "summarize",
    "reduplication",
)

# Friendly aliases.
_STAGE_ALIASES = {
    "redup":       "reduplication",
    "reduplicate": "reduplication",
    "code_switch": "codemix",
    "codeswitch":  "codemix",
    "summary":     "summarize",
    "token":       "tokenize",
    "norm":        "normalize",
}


class Doc:
    """
    Container for everything the pipeline annotates on one input string.

    Attributes are filled in only by the stages that actually ran; the rest
    keep their empty defaults, so it is always safe to read any attribute.

    Attributes
    ----------
    text             : str   — the original input
    normalized       : str   — text after the ``normalize`` stage
    tokens           : list  — word tokens (``tokenize`` / ``stopwords``)
    pos              : list  — [(token, POS_tag)] from the ``pos`` stage
    ner_tags         : list  — [(token, BIO_tag)] from the ``ner`` stage
    entities         : list  — [{'text','label','start','end'}] (``ner``)
    sentiment        : str   — 'Positive' | 'Negative' | 'Neutral'
    sentiment_scores : dict  — per-class probabilities
    language_tags    : list  — [(token, 'UR'|'EN'|'MIX')] (``codemix``)
    language_spans   : list  — contiguous language spans (``codemix``)
    summary          : str   — extractive summary (``summarize``)
    reduplications   : list  — echo/full word pairs (``reduplication``)
    stages           : list  — names of the stages that actually ran
    """

    __slots__ = (
        "text", "normalized", "tokens", "pos", "ner_tags", "entities",
        "sentiment", "sentiment_scores", "language_tags", "language_spans",
        "summary", "reduplications", "stages",
    )

    def __init__(self, text: str):
        self.text             = text
        self.normalized       = None
        self.tokens           = []
        self.pos              = []
        self.ner_tags         = []
        self.entities         = []
        self.sentiment        = None
        self.sentiment_scores = {}
        self.language_tags    = []
        self.language_spans   = []
        self.summary          = None
        self.reduplications   = []
        self.stages           = []

    @property
    def working_text(self) -> str:
        """Normalized text if normalization ran, otherwise the original."""
        return self.normalized if self.normalized is not None else self.text

    def to_dict(self) -> dict:
        """Return all populated annotations as a plain dictionary."""
        return {
            "text":             self.text,
            "normalized":       self.normalized,
            "tokens":           self.tokens,
            "pos":              self.pos,
            "ner_tags":         self.ner_tags,
            "entities":         self.entities,
            "sentiment":        self.sentiment,
            "sentiment_scores": self.sentiment_scores,
            "language_tags":    self.language_tags,
            "language_spans":   self.language_spans,
            "summary":          self.summary,
            "reduplications":   self.reduplications,
            "stages":           self.stages,
        }

    def __repr__(self):
        bits = [f"text={self.text!r}"]
        if self.sentiment:
            bits.append(f"sentiment={self.sentiment!r}")
        if self.entities:
            bits.append(f"entities={len(self.entities)}")
        if self.reduplications:
            bits.append(f"reduplications={len(self.reduplications)}")
        return f"Doc({', '.join(bits)})"


class Pipeline:
    """
    Chain AenPi modules into a single callable annotator.

    Parameters
    ----------
    stages : list of str
        The stages to run, in order. See ``AVAILABLE_STAGES``.
    verbose : bool
        Print build progress and skip warnings. Default True.

    Example
    -------
    >>> nlp = Pipeline(["normalize", "tokenize", "ner", "sentiment",
    ...                 "reduplication"])
    >>> doc = nlp("Ali Ahmed ne chai-shai pi")
    >>> doc.entities
    [{'text': 'Ali Ahmed', 'label': 'PERSON', 'start': 0, 'end': 1}]
    >>> doc.reduplications
    [{'text': 'chai-shai', 'base': 'chai', 'echo': 'shai', 'type': 'echo', ...}]
    """

    def __init__(self, stages, verbose: bool = True):
        if isinstance(stages, str):
            stages = [stages]
        self.verbose      = verbose
        self.requested    = []   # the (normalized) stage names asked for
        self._processors  = []   # list of (name, processor_callable)

        for raw in stages:
            name = _STAGE_ALIASES.get(raw, raw)
            if name not in AVAILABLE_STAGES:
                raise ValueError(
                    f"Unknown pipeline stage '{raw}'. "
                    f"Choose from: {', '.join(AVAILABLE_STAGES)}"
                )
            self.requested.append(name)

        self._build()

    # ------------------------------------------------------------------
    # Build — instantiate and fit each stage's backing model once.
    # A stage that cannot be built (missing module or dependency) is
    # skipped with a warning instead of crashing the whole pipeline.
    # ------------------------------------------------------------------

    def _build(self):
        for name in self.requested:
            try:
                processor = self._build_stage(name)
            except Exception as exc:           # noqa: BLE001 - degrade gracefully
                self._warn(f"stage '{name}' unavailable, skipping ({exc})")
                continue
            self._processors.append((name, processor))

        if self.verbose:
            active = [n for n, _ in self._processors]
            print(f"Pipeline ready. Active stages: {active}")

    def _build_stage(self, name: str):
        """Return a ``processor(doc)`` callable for ``name`` (lazy imports)."""
        if name == "normalize":
            from .normalizer import UrduNormalizer
            model = UrduNormalizer()          # rule-based normalize needs no fit
            def process(doc):
                doc.normalized = model.normalize(doc.text)
            return process

        if name == "tokenize":
            from .preprocessor import tokenize
            def process(doc):
                doc.tokens = tokenize(doc.working_text)
            return process

        if name == "stopwords":
            from .stopwords import remove_stopwords
            from .preprocessor import tokenize
            def process(doc):
                if not doc.tokens:
                    doc.tokens = tokenize(doc.working_text)
                filtered = remove_stopwords(" ".join(doc.tokens))
                doc.tokens = filtered.split()
            return process

        if name == "stem":
            from .stemmer import UrduStemmer       # not yet implemented
            model = UrduStemmer()
            from .preprocessor import tokenize
            def process(doc):
                if not doc.tokens:
                    doc.tokens = tokenize(doc.working_text)
                doc.tokens = model.stem_tokens(doc.tokens)
            return process

        if name == "pos":
            from .pos_tagger import UrduPOSTagger   # not yet implemented
            model = UrduPOSTagger()
            model.fit()
            def process(doc):
                doc.pos = model.tag(doc.working_text)
            return process

        if name == "ner":
            from .ner import UrduNER
            model = UrduNER()
            model.fit()                            # fast rule-based fallback
            def process(doc):
                doc.ner_tags = model.tag(doc.working_text)
                doc.entities = model.entities(doc.working_text)
            return process

        if name == "sentiment":
            from .sentiment import UrduSentiment
            model = UrduSentiment()
            model.fit()                            # needs sklearn + datasets
            def process(doc):
                result = model.predict(doc.working_text)
                doc.sentiment        = result["label"]
                doc.sentiment_scores = result["scores"]
            return process

        if name == "codemix":
            from .code_switch import CodeSwitchDetector
            model = CodeSwitchDetector()           # detect() works rule-based
            def process(doc):
                doc.language_tags  = model.detect(doc.working_text)
                doc.language_spans = model.spans(doc.working_text)
            return process

        if name == "summarize":
            from .summarizer import UrduSummarizer
            model = UrduSummarizer()               # summarize() works unfitted
            def process(doc):
                doc.summary = model.summarize_text(doc.working_text, n=2)
            return process

        if name == "reduplication":
            from .reduplication import ReduplicationDetector
            model = ReduplicationDetector()
            def process(doc):
                doc.reduplications = model.detect(doc.working_text)
            return process

        raise ValueError(f"No builder for stage '{name}'")  # pragma: no cover

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> Doc:
        """Run the pipeline over ``text`` and return an annotated ``Doc``."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        doc = Doc(text)
        for name, process in self._processors:
            try:
                process(doc)
                doc.stages.append(name)
            except Exception as exc:           # noqa: BLE001 - one bad stage shouldn't kill the run
                self._warn(f"stage '{name}' failed at runtime, skipping ({exc})")
        return doc

    def pipe(self, texts: list) -> list:
        """Run the pipeline over a list of texts; returns a list of ``Doc``."""
        return [self(t) for t in texts]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def stage_names(self) -> list:
        """Return the names of the stages that built successfully."""
        return [n for n, _ in self._processors]

    def _warn(self, message: str):
        if self.verbose:
            warnings.warn(f"[AenPi.Pipeline] {message}", stacklevel=2)

    def __repr__(self):
        return (f"Pipeline(active={self.stage_names()}, "
                f"requested={self.requested})")
