# AenPi

**AenPi** is a lightweight, modular Urdu NLP library built for simplicity and extensibility.
No heavy frameworks. No GPU required. Runs directly in Google Colab.

> Version 1.0.0 — `urdu` module  
> More language/domain modules coming in future versions.

---

## Installation

**From GitHub (recommended for Colab):**

```bash
!pip install git+https://github.com/your-username/AenPi.git
```

**For local development:**

```bash
git clone https://github.com/your-username/AenPi.git
cd AenPi
pip install -e .
```

**Dependencies:**

```bash
pip install psutil
```

---

## Quick Start

```python
from AenPi import urdu

# Preprocessing
urdu.preprocess("میرا نام احمد ہے!")

# Stopword removal
urdu.remove_stopwords("یہ ایک اچھا اور خوبصورت دن ہے")

# Normalization
urdu.normalize("ﻣﯿﺮﺍ ﻧﺎﻡ احمد هے")

# Spell correction
urdu.spell_correct("احمض", ["احمد", "محمد", "علی"])

# Next word prediction
corpus = ["میرا نام احمد ہے", "میرا نام علی ہے"]
urdu.ngram_predict("میرا", corpus, n=2)

# Green AI metrics
with urdu.GreenMetrics("My Task") as gm:
    urdu.preprocess("میرا نام احمد ہے")
gm.report()
```

---

## Project Structure

```
AenPi/
├── AenPi/
│   ├── __init__.py          # from AenPi import urdu
│   └── urdu/                # v1 — Urdu NLP module
│       ├── __init__.py
│       ├── preprocessor.py
│       ├── stopwords.py
│       ├── normalizer.py
│       ├── spell_corrector.py
│       ├── ngram.py
│       └── green_metrics.py
├── setup.py
├── requirements.txt
└── colab_demo.py
```

> Future versions will add modules like `AenPi.arabic`, `AenPi.hindi`, `AenPi.sentiment`, etc.
> The top-level `AenPi` package is designed to grow — each language or domain gets its own submodule.

---

## Module Reference — `AenPi.urdu`

### `preprocessor`

```python
from AenPi import urdu

urdu.preprocess(text, remove_diac=True, remove_non_urdu_chars=True)
# Full pipeline: removes diacritics, non-Urdu chars, punctuation, extra spaces

urdu.tokenize(text)
# Split text into word tokens → list

urdu.remove_punctuation(text)
# Strip Urdu and common punctuation
```

---

### `stopwords`

```python
urdu.remove_stopwords(text, custom_stopwords=None)
# Remove stopwords from text → str

urdu.get_stopwords()
# Returns the default stopword set → set

urdu.URDU_STOPWORDS
# Access the built-in stopword set directly
```

---

### `normalizer`

```python
urdu.normalize(text)
# Full normalization: Alef variants, Ya variants, Kaf, Ha, ligatures, punctuation spacing

# Individual normalizers also available:
from AenPi.urdu.normalizer import normalize_alef, normalize_ya, normalize_kaf, normalize_ha
```

---

### `spell_corrector`

Uses pure-Python **Levenshtein edit distance** — no external NLP libraries needed.

```python
urdu.edit_distance(word1, word2)
# Compute edit distance between two words → int

urdu.spell_correct(word, vocabulary, max_distance=2)
# Find closest word in vocabulary → str

urdu.spell_correct_text(text, vocabulary, max_distance=2)
# Correct every token in a text string → str
```

---

### `ngram`

```python
# One-shot convenience function
urdu.ngram_predict(context_text, corpus, n=2, top_k=3)
# Train and predict in one call → list of (word, count) tuples

# Full class for reusable models
model = urdu.NGramPredictor(n=2)
model.train(sentences)          # list of Urdu strings
model.predict("میرا", top_k=3)  # → [('نام', 2), ('گھر', 1)]
model.vocabulary()              # → set of all seen words
```

---

### `green_metrics`

Tracks **CPU usage**, **RAM consumption**, and **execution time** for any function or code block.
Useful for evaluating the environmental efficiency of NLP pipelines.

```python
# Context manager style
with urdu.GreenMetrics(label="Pipeline") as gm:
    urdu.preprocess("میرا نام احمد ہے")
gm.report()
# Prints: time, RAM before/after, delta, CPU%, green score

# One-liner wrapper
result, metrics = urdu.green_metrics(my_function, arg1, arg2, label="My Task")
print(metrics["green_score"])   # lower = more efficient
```

**Sample output:**

```
─────────────────────────────────────────────
  🌿 AenPi Green Metrics: Pipeline
─────────────────────────────────────────────
  ⏱  Execution time : 0.004 s
  🧠 RAM before     : 13.29 MB
  🧠 RAM after      : 13.29 MB
  📈 RAM delta      : 0.0 MB
  💻 CPU usage      : 12.4 %
  🌱 Green score    : 0.002 (lower = greener)
─────────────────────────────────────────────
```

---

## Colab Demo

A full runnable demo is included at `colab_demo.py`.  
Copy its contents into a Colab notebook and run cell by cell.

---

## Versioning

| Version | Modules |
|---------|---------|
| v1.0.0  | `AenPi.urdu` — preprocessing, stopwords, normalization, spell correction, n-gram prediction, green metrics |
| v1.x.x  | Planned: expanded stopword lists, trigram improvements, custom vocabulary support |
| v2.0.0+ | Planned: additional language/domain submodules |

---

## Contributing

Pull requests are welcome. To add a new submodule:

1. Create a new folder under `AenPi/` (e.g., `AenPi/arabic/`)
2. Add an `__init__.py` that exports your public functions
3. Import the submodule in `AenPi/__init__.py`
4. Keep dependencies lightweight — no transformers, TensorFlow, or PyTorch

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built with ❤️ for the Urdu NLP community.
