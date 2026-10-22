"""
Optional verification plugins — DIAGNOSTIC SOFT SIGNALS by design.

Both verifiers answer "what does an independent model see?" and report
structured evidence into `verification_report.signals`. They NEVER hard-reject
a draft:

  - A vision LLM's phrasing must not burn repair loops (it may paraphrase:
    "a feline on the sofa" IS a cat).
  - Photorealistic scenes legitimately contain incidental background text
    (signs, labels) that the prompt never asked for.

Latency contract: both are gated to run ONCE per run, on the FINAL attempt
only (wired in verify_stage.py). Each costs one vision-LLM roundtrip
(scene comprehension) or one local OCR pass (EasyOCR, ~10-20s incl. model
load on first use).

History: `florence2` was the original design (docs 06/09); the Florence-2
HF checkpoint is incompatible with transformers 5.x (missing image_token,
forced_bos_token_id), so scene comprehension rides the LLMRouter vision
path instead — frontier-grade models, zero VRAM, no local lifecycle.
"""
import re
from typing import Dict, Any, List
from src.utils import free_memory


def _name_variants(class_name: str) -> List[str]:
    """Singular/plural surface forms for word-boundary matching.
    'cat' -> [cat, cats]; 'cats' -> [cats, cat]; 'city' -> [city, cities]."""
    name = (class_name or "").lower().strip()
    if not name:
        return []
    forms = {name}
    forms.add(name + "s")
    forms.add(name + "es")
    if name.endswith("y"):
        forms.add(name[:-1] + "ies")
    if name.endswith("s"):
        forms.add(name[:-1])
    if name.endswith("ies"):
        forms.add(name[:-3] + "y")
    return sorted(forms)


def mentions(caption: str, class_name: str) -> bool:
    """Word-boundary, plural-aware containment test."""
    low = (caption or "").lower()
    return any(re.search(rf"\b{re.escape(f)}\b", low) for f in _name_variants(class_name))


class SceneComprehensionVerifier:
    """Independent 'what do you see?' pass via the LLM vision router.

    Returns a SOFT signal dict (never raises, never hard-rejects):
      {"described": bool, "missing": [...], "caption_excerpt": str, "error": str?}
    `missing` lists contract entities the vision model's description never
    mentioned — evidence for reflect/UI, weighted by nothing.
    """

    def __init__(self):
        from src.core.llm_router import LLMRouter
        self.router = LLMRouter()

    def verify(self, image, contract: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import io

        signal = {"described": False, "missing": [], "caption_excerpt": ""}
        try:
            buffered = io.BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this image in detail. List every object, "
                        "person, animal, and notable element you can see. "
                        "Include colors, positions, and counts. "
                        "Be thorough and factual."
                    )},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                ],
            }]
            caption = self.router.generate_vision(messages=messages, temperature=0.0)
            low = (caption or "").lower()
            signal["described"] = bool(low.strip())
            signal["caption_excerpt"] = low[:200]

            parsed = contract.get("parsed", {}) or {}
            checks = []
            primary = (parsed.get("primary_subject") or "").strip()
            if primary:
                checks.append(primary)
            seen = set()
            for entity in contract.get("entities", []):
                cn = (entity.get("class_name") or "").strip().lower()
                if cn and cn not in seen:
                    seen.add(cn)
                    checks.append(cn)

            for name in checks:
                if not mentions(low, name):
                    signal["missing"].append(name)
        except Exception as e:
            signal["error"] = str(e)[:200]
            print(f"[SceneComprehensionVerifier] soft-failed: {e}")
        return signal


class EasyOCRVerifier:
    """Local OCR text-hallucination diagnostic (EasyOCR, load->infer->unload).

    Returns a SOFT signal dict:
      {"texts_found": [...], "prompt_requests_text": bool, "note": str, "error": str?}
    Policy: incidental background text in photorealistic scenes is normal and
    is REPORTED, never failed. When the prompt explicitly requests text and
    OCR finds none, the note says so — evidence for the text_rendering
    category, still not a hard rejection (benchmarks decide later).
    """

    _TEXT_KEYWORDS = ("text", "sign", "says", "written", "letters", "words",
                      "label", "caption", "banner", "title", "number")

    def __init__(self):
        self.reader = None

    def _load(self):
        from src.core.device import require_device_lease
        require_device_lease()
        if self.reader is None:
            import easyocr  # type: ignore
            import torch
            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def _unload(self):
        self.reader = None
        free_memory()

    @classmethod
    def _prompt_requests_text(cls, prompt: str) -> bool:
        low = (prompt or "").lower()
        return any(kw in low for kw in cls._TEXT_KEYWORDS)

    def verify(self, image, contract: Dict[str, Any]) -> Dict[str, Any]:
        signal: Dict[str, Any] = {"texts_found": [],
                                  "prompt_requests_text": False, "note": ""}
        try:
            self._load()
            import numpy as np

            img_np = np.array(image.convert("RGB"))
            results = self.reader.readtext(img_np)
            texts = [t.lower() for (_bbox, t, prob) in results if prob > 0.5]
            self._unload()

            signal["texts_found"] = texts[:10]
            requests = self._prompt_requests_text(contract.get("prompt", ""))
            signal["prompt_requests_text"] = requests
            if texts and not requests:
                signal["note"] = ("incidental background text (normal for "
                                  "photorealistic scenes) — reported, not failed")
            elif not texts and requests:
                signal["note"] = "prompt requests text but OCR found none"
            else:
                signal["note"] = "ok"
        except Exception as e:
            self._unload()
            signal["error"] = str(e)[:200]
            print(f"[EasyOCRVerifier] soft-failed: {e}")
        return signal
