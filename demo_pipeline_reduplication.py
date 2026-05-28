"""
============================================================
AenPi — Demo for Aleena Tahir's modules
    1. Pipeline             (proposal section 7)
    2. ReduplicationDetector (proposal task T10)
============================================================

Self-contained and offline-friendly: stages whose model/dependency is missing
(e.g. 'pos' not yet implemented, or 'sentiment' without the `datasets` package)
are skipped gracefully, so this runs to completion on any machine.

Run:  python demo_pipeline_reduplication.py
"""

import sys

# Print Urdu (Nastaliq) safely on Windows consoles too.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from AenPi import urdu


def line(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ────────────────────────────────────────────────────────────
# 1. Reduplication Detector (T10)
# ────────────────────────────────────────────────────────────
line("Reduplication Detector (T10)")

redup = urdu.ReduplicationDetector()

examples = [
    "mujhe chai-shai pila do aur garm garm roti khila do",  # echo + full (Roman)
    "Karachi se kitab vitab le aao",                         # spaced echo
    "jaldi jaldi kaam khatam karo",                          # full reduplication
    "یہ کتاب وتاب لے آؤ",                                     # echo (Nastaliq)
    "گرم گرم چائے شائے",                                      # full + echo (Nastaliq)
    "this is a perfectly normal sentence",                   # nothing to detect
]

for text in examples:
    print(f"\nText : {text}")
    matches = redup.detect(text)
    if not matches:
        print("   (no reduplication)")
    for m in matches:
        print(f"   {m['text']:<16} base={m['base']:<8} "
              f"echo={m['echo']:<8} type={m['type']}")

print("\nConvenience helpers:")
print("  has_reduplication('kitab vitab') :", redup.has_reduplication("kitab vitab"))
print("  reduplications('chai-shai aao')  :", redup.reduplications("chai-shai aao"))
print("  count('chai-shai garm garm')     :", redup.count("chai-shai garm garm"))


# ────────────────────────────────────────────────────────────
# 2. Pipeline — the unified entry point
# ────────────────────────────────────────────────────────────
line("Pipeline — unified entry point (proposal section 7)")

# The exact call from the proposal. 'pos' is not implemented yet and
# 'sentiment' needs the `datasets` package — both are skipped gracefully.
nlp = urdu.Pipeline(["normalize", "tokenize", "pos", "ner", "sentiment",
                     "codemix", "reduplication"])

print("\n", nlp, "\n")

doc = nlp("Ali Ahmed ne Karachi mein bohat acha kaam kiya, chai-shai bhi pi")

print("Input          :", doc.text)
print("Stages run     :", doc.stages)
print("Tokens         :", doc.tokens)
print("Entities       :", doc.entities)
print("Language spans :", doc.language_spans)
print("Sentiment      :", doc.sentiment, "(None if `datasets` not installed)")
print("Reduplications :", doc.reduplications)
print("Doc repr       :", doc)

line("Demo complete.")
