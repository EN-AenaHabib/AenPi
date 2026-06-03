<div align="center">

<img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" />
<img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square" />
<img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
<img src="https://img.shields.io/badge/platform-CPU%20only-lightgrey?style=flat-square" />
<img src="https://img.shields.io/badge/GPU-not%20required-success?style=flat-square" />
<img src="https://img.shields.io/badge/Green%20AI-aligned-brightgreen?style=flat-square" />

# AenPi

### A Lightweight Green-AI NLP Library for Urdu

*The first unified, pip-installable Urdu NLP library that runs entirely on CPU —  
no transformers, no GPU, no LLM API calls required.*

[Installation](#installation) · [Quick Start](#quick-start) · [Modules](#module-reference) · [Pipeline](#pipeline) · [Green AI](#green-ai) · [Benchmarks](#benchmarks)

</div>

---

## Why AenPi

Urdu is spoken by over **230 million people**. Yet every modern NLP tool either ignores it entirely or forces you into one of two bad choices:

- **Legacy tools** (UrduHack, iNLTK) — incomplete, undocumented, broken on real-world text
- **Transformer models** (mBERT, XLM-R, UrduBERT) — 270M+ parameters, GPU required, 500ms+ latency per call, and enormous energy cost

AenPi is built around a third option: **classical, interpretable, efficient methods** (CRFs, finite-state rules, TF-IDF, logistic regression) that run on any laptop, in any classroom, with zero API dependency.

```
Installed size   < 50 MB          vs   mBERT: 680 MB
Inference time   < 100 ms         vs   transformer: 300–800 ms
GPU required     No               vs   yes (for production use)
CO2 per 10K ops  ~0.00003 g       vs   GPT-4: ~10.5 g  (99.9% less)
```

---

## Installation

**From GitHub (recommended for Colab):**
```bash
pip install git+https://github.com/EN-AenaHabib/AenPi.git
```

**For local development:**
```bash
git clone https://github.com/EN-AenaHabib/AenPi.git
cd AenPi
pip install -e .
```

**Dependencies:**
```bash
pip install scikit-learn datasets requests numpy psutil
# Optional — for CRF-based POS and NER:
pip install sklearn-crfsuite
```

---
## Module Reference

### Preprocessor

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

### Normalizer

Handles Unicode normalization, Alef/Ya/Kaf/Ha character variants, ligatures, and punctuation spacing.

```python
norm = UrduNormalizer()
print(norm.normalize("ﻣﯿﺮﺍ ﻧﺎﻡ احمد هے"))
# Full normalization → str

from AenPi.urdu import UrduNormalizer
```

---

### Roman Urdu Normalizer

Maps the 50+ spelling variants of Roman Urdu to a canonical form. Trained on 134K real social-media samples.

```python
norm = UrduNormalizer()

norm.normalize("kesy ho aap??  kia haal hay")
# → "kaise ho aap ? kya hal hai"

norm.normalize_batch(["booohat acha", "nhi hoga", "kia kr rha"])
# → ["bohat acha", "nahi hoga", "kya kar raha"]
```

**Why it matters:** Every downstream model breaks on spelling inconsistency. This module solves the problem that has blocked Roman Urdu NLP for years.

---

### Stopwords Remover

```python
urdu.remove_stopwords(text, custom_stopwords=None)
# → cleaned str

urdu.get_stopwords()
# → set of all default stopwords

urdu.URDU_STOPWORDS
# Direct access to the built-in stopword set
```

---

### Spell Corrector

Pure-Python Levenshtein edit distance — no external NLP libraries needed.

```python
urdu.edit_distance(word1, word2)        # → int
urdu.spell_correct(word, vocabulary, max_distance=2)     # → str
urdu.spell_correct_text(text, vocabulary, max_distance=2) # → str
```

---


---

### Stemmer

Assas-Band-style affix stripping for Urdu. Strips common prefixes and suffixes to find the root form.

```python
from AenPi.urdu import UrduStemmer

stemmer = UrduStemmer()

# Single words
print(stemmer.stem("کھانے"))   # کھا
print(stemmer.stem("لکھتا"))   # لکھ
print(stemmer.stem("گئی"))     # جا

# Sentence
print(stemmer.stem_sentence("کھانے پینے جانا"))
# → ["کھا", "پی", "جا"]
```

**Use cases:** Search engines, text indexing, information retrieval, morphological analysis.

---

### POS Tagger

CRF based Part-of-Speech tagger trained on the UD Urdu Treebank and CLE Urdu POS corpus.

```python
from AenPi.urdu.pos_tagger import UrduPOSTagger
tagger = UrduPOSTagger()

tagger.tag_sentence("احمد نے کتاب پڑھی")
# → [("احمد","PROPN"), ("نے","ADP"), ("کتاب","NOUN"), ("پڑھی","VERB")]

```

**Target accuracy:** ≥ 90% on UD Urdu test split.

---

### Named Entity Recognizer

CRF-based NER trained on the MK-PUCIT Urdu NER dataset (250K tokens). Gazetteer-enhanced for Pakistani proper nouns.
unimelb-nlp/wikiann dataset  

consists of ~20,000 sentences containing roughly 163,000 individual tokens.

```python
from AenPi import UrduNER          
ner = UrduNER()

# BIO tags
ner.tag("Ali Ahmed ne Lahore mein kaam kiya")
# → [("Ali","B-PERSON"),("Ahmed","I-PERSON"),("ne","O"),
#    ("Lahore","B-LOCATION"),("mein","O"),...]

# Clean entity list
ner.entities("Ali Ahmed ne Lahore mein kaam kiya")
# → [{"text":"Ali Ahmed","label":"PERSON","start":0,"end":1},
#    {"text":"Lahore","label":"LOCATION","start":3,"end":3}]
```

**Entity types:** `PERSON` · `LOCATION` · `ORGANIZATION` · `DATE` · `NUMBER`  
**Target:** ≥ 75% macro F1 on MK-PUCIT test split.

---

### Sentiment Classifier

TF-IDF + Logistic Regression trained on 134K Roman Urdu social-media samples. Runs in < 1ms on CPU.
(https://huggingface.co/datasets/Khubaib01/RomanUrdu-NLP-Sentiment-Corpus) 

```python
from AenPi.urdu import sentiment, sentiment_batch

print(sentiment("yeh bohat acha tha"))
# {'label': 'Positive', 'score': 0.91, 'scores': {...}}

print("\n")
print(sentiment_batch(["bohat bura", "mazay ka din", "theek hai"]))
# → [{"label":"Negative",...}, {"label":"Positive",...}, {"label":"Neutral",...}]
```

**Labels:** `Positive` · `Negative` · `Neutral`  

---

### Code-Switch Detector

Token-level language identifier for mixed Urdu–English text. The **first pip-packaged Urdu code-switch detector**.

```python
detector = urdu.CodeSwitchDetector()

# Label each token
detector.detect("mujhe yeh project deadline tak finish karna hai")
# → [("mujhe","UR"),("yeh","UR"),("project","EN"),
#    ("deadline","EN"),("tak","UR"),("finish","EN"),
#    ("karna","UR"),("hai","UR")]

# Contiguous language spans
detector.spans("mujhe project finish karna hai")
# → [{"text":"mujhe","lang":"UR","start":0,"end":0},
#    {"text":"project finish","lang":"EN","start":1,"end":2},
#    {"text":"karna hai","lang":"UR","start":3,"end":4}]

```

**Target:** ≥ 80% token F1 on held-out social media sample.

---

### Transliterator

Bidirectional Roman Urdu ↔ Nastaliq Urdu conversion using character mapping tables and n-gram disambiguation.

```python
trans = urdu.UrduTransliterator()

trans.to_nastaliq("mera naam Ahmed hai")
# → "میرا نام احمد ہے"

trans.to_roman("میرا نام احمد ہے")
# → "mera naam Ahmed hai"

trans.transliterate("mujhe khana chahiye", target="nastaliq")
# → "مجھے کھانا چاہیے"
```

**Target:** ≥ 85% top-1 accuracy on parallel test set.  
**Use cases:** Keyboard apps, SMS normalization, content migration, cross-script search.

---

### Extractive Summarizer

Returns the most important sentences from Urdu text using TF-IDF density + position weight + entity density scoring. No generation, no hallucinations.

```python
summ = urdu.UrduSummarizer()
summ.fit()

results = summ.summarize(article_text, n=3)
for r in results:
    print(r["sentence"], "→ score:", r["score"])

# Quick string output
summary = summ.summarize_text(article_text, n=2)

# Keywords
keywords = summ.keyword_summary(article_text, top_k=10)
```

**Why extractive:** Abstractive summarization generates text, which hallucinates. Extractive picks real sentences — faster, more trustworthy, no model bias.

---

### Intent Router

Replace GPT-4 intent classification with a trainable offline classifier. Trains on your custom labels in under 60 seconds on CPU.

```python
from AenPi.urdu import intent, intent_batch, intent_top

print(intent("mera parcel kahan hai"))
# {'intent': 'tracking', 'score': 0.87, 'scores': {...}}

print("\n")
print(intent_batch(["naya order karna hai", "cancel kr do"]))
# → [{"intent":"order",...}, {"intent":"complaint,...}, {"intent":"feedback",...}]

print("\n")
print(intent_top("payment fail", n=3))
# → [{"intent":"refund","score":0.72}, ...]
```

**Use case:** Any chatbot, support ticket router, or form classifier that currently calls an LLM API just to pick from 5–10 categories.

---

### Text Statistics

NLTK-style utilities for corpus analysis and research.

```python
stats = urdu.TextStats(text)

stats.freq_dist(top_k=20)         # → [("word", count), ...]
stats.concordance("احمد", window=3) # → lines showing word in context
stats.collocations(n=2, top_k=10) # → most frequent bigrams
stats.ngram_counts(n=3)           # → trigram frequency table
stats.type_token_ratio()          # → lexical diversity score (0–1)
stats.avg_word_length()           # → float
stats.sentence_count()            # → int
stats.summary()                   # → dict of all stats at once
```

---

### Reduplication Detector

Detects Urdu / Roman Urdu **reduplicated word pairs** — echo and repeat forms that
carry meaning ("and such", "all kinds of", emphasis, plurality) yet break
tokenizers and search indexes because the echo word ("shai", "vitab") is not a
real dictionary word. Purely rule-based: no training data, no model download, no GPU.

It recognises three patterns:

- **full** — an exact repeat → `garm garm` ("piping hot"), `jaldi jaldi` ("hurriedly")
- **echo** — onset swapped for a fixed echo former (`v` / `w` / `sh`, or `و` / `ش`) → `chai-shai`, `kitab-vitab`
- **rhyming** — onset swapped for any other consonant (off by default to keep precision high)

Works on both Roman and Nastaliq Urdu, and on spaced (`kitab vitab`) and hyphenated (`kitab-vitab`) forms.

```python
from AenPi.urdu import ReduplicationDetector
redup = ReduplicationDetector()

# Full annotation
redup.detect("mujhe chai-shai pila do aur garm garm roti")
# → [{"text":"chai-shai","base":"chai","echo":"shai","type":"echo","start":2,"end":2},
#    {"text":"garm garm","base":"garm","echo":"garm","type":"full","start":5,"end":6}]

redup.has_reduplication("kitab vitab")        # → True
redup.reduplications("kitab-vitab le aao")    # → ["kitab-vitab"]
redup.count("garm garm chai-shai")            # → 2

# Optional rhyming pairs (e.g. "ulta-pulta")
ReduplicationDetector(include_rhyming=True).detect("ulta-pulta")

# One-shot helper — no detector to build
from AenPi.urdu import find_reduplications
find_reduplications("chai-shai aur garm garm")
```

**Returned fields:** `text` · `base` · `echo` · `type` (`full`/`echo`/`rhyming`) · `start` · `end`
**Constructor options:** `roman_formers`, `urdu_formers`, `include_rhyming`, `min_length`
**Use cases:** Tokenizer/stemmer pre-pass, search-index normalization, morphological analysis.

---

### Pipeline

Chain any combination of modules in a single call — the unified entry point for the
entire library. Trainable models are fitted once when the pipeline is built and reused
for every call. **Robust by default:** a stage whose module or dependency is missing is
skipped with a warning instead of crashing the run.

```python
from AenPi.urdu import Pipeline

nlp = Pipeline(["normalize", "tokenize", "ner", "sentiment", "reduplication"])
# Pipeline ready. Active stages: ['normalize', 'tokenize', 'ner', 'sentiment', 'reduplication']

doc = nlp("Ali Ahmed ne chai-shai pi aur bohat acha kaam kiya")

doc.tokens
# → ["ali", "ahmed", "ne", "chai-shai", "pi", "aur", "bohat", "acha", ...]

doc.entities
# → [{"text":"Ali Ahmed","label":"PERSON","start":0,"end":1}]

doc.sentiment
# → "Positive"

doc.reduplications
# → [{"text":"chai-shai","base":"chai","echo":"shai","type":"echo","start":3,"end":3}]

doc.stages              # → which stages actually ran
doc.to_dict()           # → all annotations as a plain dict

# Batch over many texts
docs = nlp.pipe(["pehla jumla", "doosra jumla"])
```

The `Doc` object always exposes every attribute (empty default if its stage didn't run):
`text`, `normalized`, `tokens`, `pos`, `ner_tags`, `entities`, `sentiment`,
`sentiment_scores`, `language_tags`, `language_spans`, `summary`, `reduplications`, `stages`.

Stage aliases are accepted, e.g. `"redup"` → `reduplication`, `"codeswitch"` → `codemix`,
`"norm"` → `normalize`.

**Available pipeline stages:**

| Stage | Module |
|---|---|
| `"normalize"` | UrduNormalizer |
| `"tokenize"` | Tokenizer |
| `"stopwords"` | Stopwords Remover |
| `"stem"` | UrduStemmer |
| `"pos"` | POS Tagger |
| `"ner"` | UrduNER |
| `"sentiment"` | UrduSentiment |
| `"codemix"` | CodeSwitchDetector |
| `"summarize"` | UrduSummarizer |
| `"reduplication"` | ReduplicationDetector |

---

### Green Metrics

Track CPU usage, RAM consumption, and execution time for any code block.

```python
with urdu.GreenMetrics("Pipeline") as gm:
    doc = nlp("میرا نام احمد ہے")
gm.report()
```

```
─────────────────────────────────────────────
  AenPi Green Metrics: Pipeline
─────────────────────────────────────────────
  Execution time : 0.004 s
  RAM before     : 13.29 MB
  RAM after      : 13.29 MB
  RAM delta      : 0.0 MB
  CPU usage      : 12.4 %
  Green score    : 0.002 (lower = greener)
─────────────────────────────────────────────
```

```python
# One-liner wrapper
result, metrics = urdu.green_metrics(my_function, arg1, arg2, label="Task")
print(metrics["green_score"])
```

---

### Carbon Estimator

Show exactly how much CO2 and energy you save by using AenPi instead of LLM API calls.

```python
carbon = urdu.CarbonEstimator()

report = carbon.compare(
    llm="gpt-4",
    module="intent_router",
    n_calls=10_000,
    avg_tokens=50
)
carbon.print_report(report)
```

```
═══════════════════════════════════════════════════════
  AenPi Carbon Cost Report
═══════════════════════════════════════════════════════
  Workload  : 10,000 calls, ~50 tokens each

  LLM       : GPT-4 (OpenAI)
    Energy  : 8.7500 mWh
    CO2     : 10.5000 g

  AenPi     : IntentRouter
    Energy  : 0.000021 mWh
    CO2     : 0.000025 g
    Runtime : 5.00 seconds total

  Savings
    CO2     : 10.4999 g saved  (99.9% reduction)
    ~ 0.05 km NOT driven by a car
    ~ 1312 smartphone charges avoided
═══════════════════════════════════════════════════════
```

---

## Green AI

AenPi is designed from the ground up around Green AI principles.

| Constraint | Target | Status |
|---|---|---|
| Installed package size | < 50 MB | Achieved |
| CPU-only inference (500 tokens) | < 100 ms | Achieved |
| Full pipeline training time | < 4 GPU-hours | Achieved |
| GPU required | No | No GPU needed |
| External API dependency | None | Zero |

Every model uses classical, interpretable methods: CRFs, logistic regression, TF-IDF, rule-based FSTs. No transformers. No PyTorch. No TensorFlow.

---

## Benchmarks

| Task | AenPi Target | UrduHack | mBERT |
|---|---|---|---|
| POS tagging accuracy | ≥ 90% | ~78% | ~93% |
| NER macro F1 | ≥ 75% | Not available | ~82% |
| Sentiment accuracy | ≥ 80% | Not available | ~84% |
| Code-mix token F1 | ≥ 80% | Not available | Not packaged |
| Inference latency | < 100 ms | ~150 ms | 400–800 ms |
| Installed size | < 50 MB | ~8 MB | 680 MB |
| GPU required | No | No | Yes (production) |

*Full benchmark report published in `/benchmarks/` directory.*

---

## Datasets

| Module | Dataset | License |
|---|---|---|
| Normalizer, Sentiment, CodeSwitch | Khubaib01/RomanUrdu-NLP-Sentiment-Corpus | Apache 2.0 |
| NER | MK-PUCIT Urdu NER Corpus | Public research |
| POS Tagger | UD Urdu Treebank + CLE POS Corpus | CC BY-SA |
| Summarizer | mirfan899/usummary | Public |
| Embeddings | Urdu Wikipedia dump | CC BY-SA |

All datasets are open, well-cited, and load automatically on first `.fit()` call via HuggingFace `datasets`.

---

## Project Structure

```
AenPi/
├── AenPi/
│   ├── __init__.py
│   └── urdu/
│       ├── __init__.py
│       ├── preprocessor.py       # tokenization, cleaning
│       ├── normalizer.py         # Nastaliq Unicode normalization
│       ├── roman_normalizer.py   # Roman Urdu spelling normalization
│       ├── stopwords.py          # stopword removal
│       ├── spell_corrector.py    # Levenshtein spell correction
│       ├── ngram.py              # n-gram language model
│       ├── stemmer.py            # Assas-Band-style stemmer
│       ├── pos_tagger.py         # CRF POS tagger
│       ├── ner.py                # CRF Named Entity Recognizer
│       ├── sentiment.py          # TF-IDF + LR sentiment classifier
│       ├── code_switch.py        # code-mix language detector
│       ├── transliterator.py     # Roman ↔ Nastaliq transliteration
│       ├── summarizer.py         # extractive summarizer
│       ├── reduplication.py      # rule-based reduplication detector
│       ├── intent_router.py      # offline intent classifier
│       ├── text_stats.py         # FreqDist, concordance, collocations
│       ├── pipeline.py           # unified Pipeline API
│       ├── green_metrics.py      # CPU/RAM/time tracker
│       └── carbon.py             # CO2 vs LLM estimator
├── tests/
│   └── test_all_modules.py       # 60+ unit tests
├── benchmarks/
│   └── benchmark_report.ipynb
├── setup.py
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Colab Demo

A full runnable demo notebook is included. Open directly in Colab:

```python
# Cell 1 — Install
!pip install git+https://github.com/EN-AenaHabib/AenPi.git

# Cell 2 — Run pipeline
from AenPi import urdu
nlp = urdu.Pipeline(["normalize", "tokenize", "ner", "sentiment"])
doc = nlp("Ali ne Lahore mein bohat acha kaam kiya")
print(doc.entities)
print(doc.sentiment)
```

---

## Versioning

| Version | Modules |
|---|---|
| v1.0.0 | Core: preprocessor, stopwords, normalizer, spell corrector, n-gram, green metrics |
| v1.1.0 | Added: sentiment, NER, code-switch detector, Roman normalizer, carbon estimator |
| v1.2.0 | Added: POS tagger, stemmer, transliterator, text stats, intent router, summarizer, Pipeline |
| v2.0.0 (planned) | Additional language submodules: Pashto, Sindhi |

---

## Contributing

Pull requests are welcome. To add a new submodule:

1. Create your module file under `AenPi/urdu/`
2. Export your public functions in `AenPi/urdu/__init__.py`
3. Add tests in `tests/test_all_modules.py`
4. Keep dependencies lightweight — no transformers, TensorFlow, or PyTorch

*Currently accepting contributions to the `urdu` module only.*

---

## Academic Citation

If you use AenPi in your research, please cite:

```bibtex
@software{aenpi2026,
  title  = {AenPi: A Lightweight Green-AI NLP Library for Urdu},
  author = {Aena Habib, Eman Asghar, Dua Kamal, Aleena
            Tahir, Saqlain Abbas},
  year   = {2026},
  url    = {https://github.com/EN-AenaHabib/AenPi},
  note   = {National University of Technology, Department of Artificial Intelligence}
}
```

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built at National University of Technology · Department of Artificial Intelligence · Spring 2026
</div>
