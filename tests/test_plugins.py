"""
Advanced verification plugins — soft-signal semantics (CPU, no GPU).

The plugins must NEVER hard-reject a draft: their output is diagnostic
evidence for verification_report.signals. Scene comprehension matching is
word-boundary + plural-aware; EasyOCR is report-only.
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.advanced_qa import (
    EasyOCRVerifier,
    SceneComprehensionVerifier,
    mentions,
    _name_variants,
)

_TINY = Image.new("RGB", (8, 8))


# --------------------------------------------------------------------------
# Name matching (word boundary + plural forms)
# --------------------------------------------------------------------------

def test_mentions_word_boundary_not_substring():
    assert mentions("a category of things", "cat") is False      # substring trap
    assert mentions("a cat on the mat", "cat") is True


def test_mentions_plural_forms():
    assert mentions("three cats sitting", "cat") is True
    assert mentions("a single cat", "cats") is True
    assert mentions("several cities at dusk", "city") is True
    assert mentions("boxes of fruit", "box") is True
    assert mentions("only dogs here", "cat") is False


def test_name_variants_symmetric():
    assert "cats" in _name_variants("cat")
    assert "cat" in _name_variants("cats")
    assert "cities" in _name_variants("city")
    assert "city" in _name_variants("cities")


# --------------------------------------------------------------------------
# SceneComprehensionVerifier — soft signal contract
# --------------------------------------------------------------------------

def _make_verifier_with_caption(monkeypatch, caption: str) -> SceneComprehensionVerifier:
    v = SceneComprehensionVerifier.__new__(SceneComprehensionVerifier)

    class _R:
        def generate_vision(self, messages, temperature=0.0):
            return caption

    v.router = _R()
    return v


def test_scene_verifier_reports_missing_entities_as_signal(monkeypatch):
    v = _make_verifier_with_caption(
        monkeypatch, "a red apple on a wooden table with a knife beside it")
    contract = {
        "parsed": {"primary_subject": "apple"},
        "entities": [{"class_name": "apple"}, {"class_name": "table"},
                     {"class_name": "helicopter"}],
    }
    signal = v.verify(image=_TINY, contract=contract)
    assert signal["described"] is True
    assert signal["missing"] == ["helicopter"]       # apple/table mentioned
    assert isinstance(signal, dict) and "hard_failures" not in signal


def test_scene_verifier_primary_subject_plural_match(monkeypatch):
    v = _make_verifier_with_caption(monkeypatch, "wildflowers blooming in a meadow")
    contract = {"parsed": {"primary_subject": "wildflower"}, "entities": []}
    signal = v.verify(image=_TINY, contract=contract)
    assert signal["missing"] == []                   # plural description matches


def test_scene_verifier_error_is_soft(monkeypatch):
    v = SceneComprehensionVerifier.__new__(SceneComprehensionVerifier)

    class _R:
        def generate_vision(self, messages, temperature=0.0):
            raise RuntimeError("provider down")

    v.router = _R()
    signal = v.verify(image=_TINY, contract={"parsed": {}, "entities": []})
    assert signal["described"] is False
    assert "provider down" in signal["error"]


# --------------------------------------------------------------------------
# EasyOCRVerifier — report-only semantics
# --------------------------------------------------------------------------

def test_ocr_prompt_text_detection_keyword_scoping():
    assert EasyOCRVerifier._prompt_requests_text("a shop sign that says OPEN") is True
    assert EasyOCRVerifier._prompt_requests_text("three red apples") is False


def test_ocr_verifier_incidental_text_is_reported_not_failed(monkeypatch):
    v = EasyOCRVerifier.__new__(EasyOCRVerifier)
    monkeypatch.setattr(v, "_load", lambda: None)
    monkeypatch.setattr(v, "_unload", lambda: None)

    class _FakeReader:
        def readtext(self, img):
            return [([[0, 0]], "OPEN", 0.9)]

    v.reader = _FakeReader()
    signal = v.verify(image=_TINY, contract={"prompt": "three red apples"})
    assert signal["texts_found"] == ["open"]
    assert signal["prompt_requests_text"] is False
    assert "not failed" in signal["note"]
    assert "error" not in signal


def test_ocr_verifier_requested_text_missing_gets_note(monkeypatch):
    v = EasyOCRVerifier.__new__(EasyOCRVerifier)
    monkeypatch.setattr(v, "_load", lambda: None)
    monkeypatch.setattr(v, "_unload", lambda: None)

    class _FakeReader:
        def readtext(self, img):
            return []

    v.reader = _FakeReader()
    signal = v.verify(image=_TINY, contract={"prompt": "a sign that says hello"})
    assert signal["prompt_requests_text"] is True
    assert "found none" in signal["note"]
