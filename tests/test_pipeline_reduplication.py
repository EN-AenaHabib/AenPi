"""
Tests for Aleena Tahir's modules:
    * ReduplicationDetector  (proposal task T10)
    * Pipeline               (proposal section 7 — the unified entry point)

Run with:   pytest tests/test_pipeline_reduplication.py -v
"""

from AenPi.urdu import ReduplicationDetector, find_reduplications, Pipeline, Doc


# ───────────────────────────────────────────────────────────────────────────
# ReduplicationDetector
# ───────────────────────────────────────────────────────────────────────────

def test_echo_hyphenated_roman():
    redup = ReduplicationDetector()
    found = redup.detect("mujhe chai-shai pila do")
    assert len(found) == 1
    m = found[0]
    assert m["text"] == "chai-shai"
    assert m["base"] == "chai" and m["echo"] == "shai"
    assert m["type"] == "echo"


def test_echo_spaced_roman():
    redup = ReduplicationDetector()
    found = redup.detect("kitab vitab le aao")
    assert len(found) == 1
    assert found[0]["base"] == "kitab" and found[0]["echo"] == "vitab"
    assert found[0]["type"] == "echo"


def test_full_reduplication():
    redup = ReduplicationDetector()
    found = redup.detect("garm garm chai jaldi jaldi laao")
    types = [m["type"] for m in found]
    assert types == ["full", "full"]
    assert found[0]["text"] == "garm garm"


def test_echo_nastaliq():
    redup = ReduplicationDetector()
    found = redup.detect("یہ کتاب وتاب لے آؤ")
    assert len(found) == 1
    assert found[0]["type"] == "echo"
    assert found[0]["base"] == "کتاب" and found[0]["echo"] == "وتاب"


def test_full_nastaliq():
    redup = ReduplicationDetector()
    found = redup.detect("گرم گرم چائے")
    assert len(found) == 1
    assert found[0]["type"] == "full"


def test_no_false_positives():
    redup = ReduplicationDetector()
    # Coincidental rhyme + ordinary text must NOT be flagged by default.
    assert redup.detect("acha bacha") == []
    assert redup.detect("ek do teen char") == []
    assert redup.detect("this is a normal english sentence") == []


def test_rhyming_opt_in():
    loose = ReduplicationDetector(include_rhyming=True)
    found = loose.detect("ulta pulta cheezein")
    assert len(found) == 1
    assert found[0]["type"] == "rhyming"


def test_convenience_helpers():
    redup = ReduplicationDetector()
    assert redup.has_reduplication("kitab-vitab") is True
    assert redup.has_reduplication("normal text") is False
    assert redup.reduplications("chai-shai aur garm garm") == \
        ["chai-shai", "garm garm"]
    assert redup.count("chai-shai garm garm") == 2


def test_find_reduplications_function():
    found = find_reduplications("chai-shai")
    assert found and found[0]["type"] == "echo"


def test_fit_is_noop():
    # fit() exists only for Pipeline symmetry and returns self.
    redup = ReduplicationDetector()
    assert redup.fit() is redup


# ───────────────────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────────────────

def test_pipeline_runs_available_stages():
    nlp = Pipeline(["tokenize", "ner", "reduplication"], verbose=False)
    doc = nlp("Ali Ahmed ne chai-shai pi")
    assert isinstance(doc, Doc)
    assert doc.tokens                      # tokenized
    assert "reduplication" in doc.stages
    assert any(r["text"] == "chai-shai" for r in doc.reduplications)


def test_pipeline_section7_call_is_robust():
    # The exact proposal section-7 example. 'pos' is not implemented yet and
    # 'sentiment' may lack the 'datasets' dependency — the pipeline must still
    # build and run the stages that ARE available instead of crashing.
    nlp = Pipeline(["normalize", "tokenize", "pos", "ner", "sentiment"],
                   verbose=False)
    doc = nlp("Ali Ahmed ne Lahore mein kaam kiya")
    assert "tokenize" in doc.stages
    assert "ner" in doc.stages
    # entities found by the rule-based NER fallback
    labels = {e["label"] for e in doc.entities}
    assert "LOCATION" in labels


def test_pipeline_rejects_unknown_stage():
    import pytest
    with pytest.raises(ValueError):
        Pipeline(["tokenize", "definitely_not_a_stage"])


def test_pipeline_aliases_and_single_string():
    assert Pipeline("redup", verbose=False).stage_names() == ["reduplication"]
    assert Pipeline("codeswitch", verbose=False).stage_names() == ["codemix"]


def test_doc_to_dict_and_working_text():
    nlp = Pipeline(["tokenize"], verbose=False)
    doc = nlp("hello world")
    d = doc.to_dict()
    assert d["text"] == "hello world"
    assert d["tokens"] == ["hello", "world"]
    # No normalize stage -> working_text falls back to the original text.
    assert doc.working_text == "hello world"
