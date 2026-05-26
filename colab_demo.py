# ============================================================
# AenPi - Urdu NLP Library  |  Google Colab Demo
# Run each cell in order in Google Colab
# ============================================================

# ── Cell 1: Install & Setup ──────────────────────────────────
# Run this in Colab:
#   !git clone https://github.com/your-repo/AenPi.git   (or upload folder)
#   %cd AenPi
#   !pip install -e . -q

# For quick local testing without install:
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Cell 2: Import ───────────────────────────────────────────
from AenPi import urdu

print("✅ AenPi imported successfully!")

# ── Cell 3: Preprocessing ────────────────────────────────────
raw = "میرا نام احمد ہے! وہ لاہور میں رہتا ہے۔"

cleaned = urdu.preprocess(raw)
tokens  = urdu.tokenize(cleaned)

print("Raw     :", raw)
print("Cleaned :", cleaned)
print("Tokens  :", tokens)

# ── Cell 4: Stopword Removal ─────────────────────────────────
sentence = "یہ ایک اچھا اور خوبصورت دن ہے"
filtered = urdu.remove_stopwords(sentence)

print("Original :", sentence)
print("Filtered :", filtered)

# ── Cell 5: Normalization ────────────────────────────────────
messy = "ﻣﯿﺮﺍ ﻧﺎﻡ احمد هے"  # mixed ligatures + Arabic ha
normalized = urdu.normalize(messy)

print("Messy      :", messy)
print("Normalized :", normalized)

# ── Cell 6: Spell Correction ─────────────────────────────────
vocabulary = ["احمد", "محمد", "علی", "لاہور", "کراچی", "گھر", "ہے", "نام"]

wrong_word  = "احمض"
corrected   = urdu.spell_correct(wrong_word, vocabulary)
print(f"spell_correct('{wrong_word}') → '{corrected}'")

# Edit distance demo
dist = urdu.edit_distance("احمد", "احمض")
print(f"edit_distance('احمد', 'احمض') = {dist}")

# Full text correction
wrong_text    = "احمض گهر میں ہے"
corrected_text = urdu.spell_correct_text(wrong_text, vocabulary)   # if imported directly
# OR via module:
from AenPi.urdu.spell_corrector import spell_correct_text
print("Corrected text:", spell_correct_text(wrong_text, vocabulary))

# ── Cell 7: N-Gram Next Word Prediction ──────────────────────
corpus = [
    "میرا نام احمد ہے",
    "میرا نام علی ہے",
    "میرا گھر لاہور میں ہے",
    "وہ لاہور میں رہتا ہے",
    "وہ کراچی میں پڑھتا ہے",
]

# Bigram (n=2)
predictions = urdu.ngram_predict("میرا", corpus, n=2, top_k=3)
print("Bigram predictions after 'میرا'  :", predictions)

# Trigram (n=3)
predictions3 = urdu.ngram_predict("میرا نام", corpus, n=3, top_k=3)
print("Trigram predictions after 'میرا نام':", predictions3)

# Using the class directly
model = urdu.NGramPredictor(n=2)
model.train(corpus)
print(model)
print("Predict 'وہ':", model.predict("وہ", top_k=2))

# ── Cell 8: Green AI Metrics ─────────────────────────────────
# Method A: context manager
with urdu.GreenMetrics(label="Preprocessing Pipeline") as gm:
    for _ in range(500):
        urdu.preprocess("میرا نام احمد ہے اور میں لاہور میں رہتا ہوں")

gm.report()

# Method B: one-liner wrapper
def run_ngram():
    m = urdu.NGramPredictor(n=2)
    m.train(corpus * 100)
    return m.predict("میرا")

result, metrics = urdu.green_metrics(run_ngram, label="NGram Training x100")
print("Returned result:", result)
print("Green score    :", metrics["green_score"])

# ============================================================
# Aleena Tahir's modules — Reduplication Detector (T10) + Pipeline
# ============================================================

# ── Cell 9: Reduplication Detector (T10) ─────────────────────
# Detects echo-words (chai-shai, kitab-vitab) and full repeats
# (garm garm). Pure rule-based — no training, no download.
redup = urdu.ReduplicationDetector()

examples = [
    "mujhe chai-shai pila do aur garm garm roti khila do",   # echo + full (Roman)
    "Karachi se kitab vitab le aao",                          # spaced echo
    "jaldi jaldi karo",                                       # full reduplication
    "یہ کتاب وتاب لے آؤ",                                      # echo (Nastaliq)
    "گرم گرم چائے شائے",                                       # full + echo (Nastaliq)
]
for text in examples:
    print("Text :", text)
    for m in redup.detect(text):
        print(f"   {m['text']:<14} base={m['base']:<8} echo={m['echo']:<8} type={m['type']}")
    print()

print("has_reduplication('kitab vitab') :", redup.has_reduplication("kitab vitab"))
print("reduplications('chai-shai aao')  :", redup.reduplications("chai-shai aao"))

# ── Cell 10: Pipeline — the unified entry point ──────────────
# Chain any combination of modules in one call. Stages whose
# module or dependency is missing are skipped gracefully, so
# this runs on any machine. (proposal section 7)
nlp = urdu.Pipeline(["normalize", "tokenize", "codemix", "ner",
                     "sentiment", "reduplication"])

doc = nlp("Ali Ahmed ne Karachi mein bohat acha kaam kiya, chai-shai bhi pi")

print("Stages run     :", doc.stages)
print("Tokens         :", doc.tokens)
print("Entities       :", doc.entities)
print("Language spans  :", doc.language_spans)
print("Sentiment      :", doc.sentiment)        # None if 'datasets' not installed
print("Reduplications :", doc.reduplications)
print("Doc            :", doc)
