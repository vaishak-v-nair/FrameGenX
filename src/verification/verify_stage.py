"""
VerifyStage — composes three verifiers per attempt:

  1. QAVerifier (GroundingDINO): object-count hard failures
  2. CLIPScoreVerifier: image vs ORIGINAL user prompt similarity
     (never final_prompt — +RAG/reflect configs must not rewrite the
     ruler they're measured with; eng review D9)
  3. VQAVerifier (external vision provider): existence + color questions

Scoring verifiers ALWAYS run in every ablation config and their runtime is
always inside the latency column; `--disable reflect` only turns off reward
GATING and corrective edits (eng review scoring-vs-gating rule).

All verifier models: fp16, GPU for inference, CPU-resident between uses.
"""
import torch
import re
from PIL import Image
from typing import Dict, Any, List, Tuple

from src.core.stage import Stage, register_stage
from src.core.context import RunContext
from src.verification.qa import QAVerifier
from src.verification.contract_evidence import build_contract_evidence
from src.utils import free_memory

REWARD_WEIGHTS = (0.5, 0.5)  # (CLIP_norm, VQA)


def clip_norm_from_cos(cos: float) -> float:
    """clamp((cos - 0.15) / 0.25, 0, 1) — calibrated for
    openai/clip-vit-base-patch32 cosine range (~0.15-0.40 for SDXL-era pairs)."""
    return max(0.0, min(1.0, (cos - 0.15) / 0.25))


def compute_reward(clip_norm: float, vqa_score: float) -> float:
    w_clip, w_vqa = REWARD_WEIGHTS
    return w_clip * clip_norm + w_vqa * vqa_score


COLOR_SYNONYMS = {
    "red": {"red", "crimson", "scarlet", "maroon", "ruby"},
    "blue": {"blue", "navy", "azure", "cobalt", "teal"},
    "green": {"green", "emerald", "lime", "olive"},
    "yellow": {"yellow", "gold", "golden", "amber"},
    "orange": {"orange", "tangerine"},
    "purple": {"purple", "violet", "lavender", "magenta"},
    "pink": {"pink", "rose", "salmon"},
    "brown": {"brown", "tan", "beige", "chocolate"},
    "black": {"black", "dark"},
    "white": {"white", "ivory", "cream"},
    "gray": {"gray", "grey", "silver"},
}


def color_matches(expected: str, answer: str) -> bool:
    answer = answer.lower().strip().strip(" .,!?")
    group = COLOR_SYNONYMS.get(expected.lower(), {expected.lower()})
    return answer in group


def _parse_vqa_answers(response_text: str, questions: List[Dict[str, str]]) -> List[str]:
    """Preserve question identity; never turn malformed model prose into a 'no'."""
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    numbered = {}
    for line in lines:
        match = re.fullmatch(r"(\d+)[.)]\s*(.*?)\s*", line)
        if match:
            numbered[int(match.group(1))] = match.group(2)
    if numbered:
        raw_answers = [numbered.get(i + 1, "") for i in range(len(questions))]
    elif len(lines) == len(questions):
        raw_answers = lines
    else:
        raw_answers = [""] * len(questions)

    known_colors = set().union(*COLOR_SYNONYMS.values())
    parsed = []
    for question, raw in zip(questions, raw_answers):
        answer = raw.lower().strip().strip(" .,!?")
        if question["type"] == "existence":
            parsed.append(answer if answer in {"yes", "no"} else "unknown")
        else:
            allowed = known_colors | {question["expected"].lower()}
            parsed.append(answer if answer in allowed else "unknown")
    return parsed


def build_vqa_questions(contract: Dict[str, Any]) -> List[Dict[str, str]]:
    """Deduped on (class_name, attribute-set): 'three red apples' yields ONE
    existence question, not three (counting belongs to DINO, not VQA).
    class_name is the extractor's surface form — plural stays plural."""
    entities = contract.get("entities", [])
    seen: set = set()
    counts: Dict[Tuple, int] = {}
    order: List[Tuple] = []
    for e in entities:
        key = (e.get("class_name", "").lower(), tuple(sorted(e.get("attributes", []))))
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.add(key)
            order.append(key)

    questions = []
    for key in order:
        class_name, attrs = key
        if not class_name:
            continue
        attr_str = " ".join(attrs)
        label = f"{attr_str} {class_name}".strip()
        if counts[key] > 1:
            q = f"Are there {label} in the image?"
        else:
            article = "an" if label[0] in "aeiou" else "a"
            q = f"Is there {article} {label} in the image?"
        questions.append({"type": "existence", "question": q, "expected": "yes",
                          "class_name": class_name})
        for attr in attrs:
            if attr in COLOR_SYNONYMS:
                questions.append({
                    "type": "color",
                    "question": f"What color is the {class_name}?",
                    "expected": attr,
                    "class_name": class_name,
                })
    return questions


class CLIPScoreVerifier:
    def __init__(self, config):
        self.model_id = config.get("models", {}).get("clip", "openai/clip-vit-base-patch32")
        self.device = "cpu"
        self.model = None
        self.processor = None

    def _load(self):
        from src.core.device import require_device_lease
        require_device_lease()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            # Shared process-lifetime singleton (CPU fp32) — retrieval
            # already paid the disk load; a second per-run load cost
            # 60-90s through the WSL bind mount. fp32 on GPU transiently
            # (~1.2GB VRAM) is safe: the SDXL pipe is CPU-resident during
            # verify under model offload.
            from src.utils import get_clip
            self.model, self.processor = get_clip(self.model_id)
        self.model.to(self.device)

    def _to_cpu(self):
        if self.model is not None:
            self.model.to("cpu")
        free_memory()

    def score(self, image: Image.Image, original_prompt: str) -> Dict[str, float]:
        self._load()
        inputs = self.processor(text=[original_prompt], images=image,
                                return_tensors="pt", padding=True, truncation=True)
        from src.utils import clip_image_embed, clip_text_embed, cast_inputs
        inputs = cast_inputs(dict(inputs), self.device, self.model.dtype)
        with torch.no_grad():
            img_f = clip_image_embed(self.model, inputs["pixel_values"])
            txt_f = clip_text_embed(self.model, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        cos = float((img_f * txt_f).sum().item())
        self._to_cpu()
        return {"clip_cos": round(cos, 4), "clip_norm": round(clip_norm_from_cos(cos), 4)}


class VQAVerifier:
    def __init__(self, config):
        from src.core.llm_router import LLMRouter
        self.router = LLMRouter()

    def score(self, image: Image.Image, contract: Dict[str, Any]) -> Dict[str, Any]:
        questions = build_vqa_questions(contract)
        if not questions:
            # No visual questions means no VQA measurement. A synthetic 1.0
            # would inflate benchmark averages for empty-entity contracts.
            return {"vqa_score": None, "vqa_details": [], "vqa_vacuous": True,
                    "vqa_status": "not_applicable"}

        details = []
        correct = 0
        
        # Batch questioning
        q_texts = []
        for i, q in enumerate(questions):
            q_texts.append(f"{i+1}. {q['question']}")
            
        prompt = "Answer the following questions about the image very concisely. Use ONLY 'yes', 'no', or a single color name. Format your response as a numbered list:\n" + "\n".join(q_texts)
        
        # Convert image to base64
        import base64
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response_text = self.router.generate_vision(messages=messages, temperature=0.0)
            answers = _parse_vqa_answers(response_text, questions)
            for i, q in enumerate(questions):
                answer = answers[i]
                ok = None if answer == "unknown" else (
                    answer == "yes" if q["type"] == "existence" else
                    color_matches(q["expected"], answer)
                )
                if ok is not None:
                    correct += int(ok)
                details.append({**q, "answer": answer, "correct": ok})
                
        except Exception as e:
            print(f"[VQAVerifier] Vision provider unavailable: {e}")
            for q in questions:
                details.append({**q, "answer": "unavailable", "correct": None})
            return {"vqa_score": None, "vqa_details": details,
                    "vqa_vacuous": False, "vqa_status": "unavailable"}

        all_observed = all(d["correct"] is not None for d in details)
        any_observed = any(d["correct"] is not None for d in details)
        return {
            "vqa_score": round(correct / len(questions), 4) if all_observed else None,
            "vqa_details": details,
            "vqa_vacuous": False,
            "vqa_status": "observed" if all_observed else
                          ("partial" if any_observed else "unavailable"),
        }


@register_stage
class VerifyStage(Stage):
    name = "verify"

    def __init__(self, config):
        super().__init__(config)
        self.qa = QAVerifier()
        self.clip = CLIPScoreVerifier(config)
        verify_config = config.get("verification", {}) or {}
        self.vqa = VQAVerifier(config) if verify_config.get("external_vqa", True) else None
        self.local_instances = None
        if verify_config.get("local_instances", False):
            from src.verification.instance_color import InstanceColorVerifier
            self.local_instances = InstanceColorVerifier()
        # Spatial verifier: pure CPU geometry, 0 VRAM
        from src.verification.spatial_verifier import SpatialRelationVerifier
        self.spatial = SpatialRelationVerifier(config)

        # Optional diagnostic plugins (SOFT signals — never hard-reject).
        # Both run ONCE per run, on the FINAL attempt, to bound latency.
        plugins_config = self.config.get("verification", {}).get("plugins", {})
        self.use_scene_comprehension = bool(
            plugins_config.get("scene_comprehension",
                               plugins_config.get("florence2", False)))
        self.use_easyocr = bool(plugins_config.get("easyocr", False))

        if self.use_scene_comprehension:
            from src.verification.advanced_qa import SceneComprehensionVerifier
            self.scene_verifier = SceneComprehensionVerifier()

        if self.use_easyocr:
            from src.verification.advanced_qa import EasyOCRVerifier
            self.easyocr_verifier = EasyOCRVerifier()

    def run(self, ctx: RunContext) -> None:
        print(f"[verify] attempt {ctx.attempt}")
        image_path = ctx.run_dir / f"generated_{ctx.attempt}.png"

        count_decision = self.qa.verify(image_path, ctx.contract)

        image = Image.open(image_path).convert("RGB")
        local_evidence = None
        if getattr(self, "local_instances", None) is not None:
            from src.verification.instance_color import refine_count_decision
            try:
                local_evidence = self.local_instances.verify(image, ctx.contract,
                    count_decision.get("detected_boxes", {}), ctx.run_dir / f"instance_masks_{ctx.attempt}")
                count_decision = refine_count_decision(count_decision, local_evidence)
                for name, info in local_evidence["classes"].items():
                    if info["status"] == "contradicted_diagnostic":
                        count_decision["hard_failures"].append({"type": "attribute_mismatch", "class_name": name,
                            "expected": info["expected_colors"], "detected": info["observed_colors"]})
                    elif self.config.get("verification", {}).get("require_chromatic_evidence", False):
                        from src.verification.instance_color import CHROMATIC
                        if (info["status"] == "unknown" and info["expected_colors"]
                                and all(c in CHROMATIC for c in info["expected_colors"])):
                            count_decision["hard_failures"].append({"type": "attribute_unverified", "class_name": name,
                                "expected": info["expected_colors"], "detected": info.get("reason", "unknown color")})
            except Exception as error:
                local_evidence = {"status": "unavailable", "error_type": type(error).__name__,
                                  "scope": "local_instance_verification_unavailable"}
                count_decision.setdefault("hard_failures", []).append({"type": "verification_unavailable",
                    "class_name": "objects", "expected": "local instance evidence", "detected": type(error).__name__})
            count_decision.setdefault("signals", {})["local_instances"] = local_evidence
        clip_scores = self.clip.score(image, ctx.contract.get("prompt", ctx.prompt))
        vqa_scores = self.vqa.score(image, ctx.contract) if self.vqa is not None else {
            "vqa_score": None, "vqa_details": [], "vqa_vacuous": False, "vqa_status": "disabled"}

        # --- Spatial Relation Verification (CPU-only, uses DINO boxes) ---
        spatial_result = {"passed": True, "total": 0, "verified": 0, "failures": []}
        spatial_rels = (ctx.contract or {}).get("spatial_relationships", [])
        if spatial_rels:
            # Collect detected boxes from QAVerifier's DINO output
            detected_boxes = count_decision.get("detected_boxes", {})
            spatial_result = self.spatial.verify(
                relationships=spatial_rels,
                detected_boxes=detected_boxes,
                image_width=image.width,
                image_height=image.height,
            )
            if spatial_result["failures"]:
                print(f"[verify] spatial: {spatial_result['verified']}/{spatial_result['total']} passed")

        # Merge hard failures: count + spatial
        hard_failures = list(count_decision.get("hard_failures", []))
        for sf in spatial_result.get("failures", []):
            hard_failures.append({
                "type": "spatial_violation",
                "class_name": f"{sf['subject']}->{sf['object']}",
                "expected": sf["predicate"],
                "detected": sf["reason"],
            })

        contract_evidence = build_contract_evidence(
            ctx.contract or {}, count_decision, spatial_result, vqa_scores, local_evidence,
        )

        # --- Optional diagnostic plugins (SOFT signals, final attempt only) ---
        # They never extend hard_failures: evidence lands in
        # verification_report.signals for the UI and post-hoc analysis.
        max_loops = int(self.config.get("pipeline", {}).get("max_repair_loops", 3))
        is_final_attempt = ctx.attempt >= max_loops
        plugins_active = (getattr(self, "use_easyocr", False)
                          or getattr(self, "use_scene_comprehension", False))
        if plugins_active and not is_final_attempt:
            print("[verify] plugins deferred to final attempt (latency)")
        if getattr(self, "use_easyocr", False) and is_final_attempt:
            print("[verify] EasyOCR plugin (diagnostic)...")
            ocr_signal = self.easyocr_verifier.verify(image, ctx.contract)
            count_decision.setdefault("signals", {})["ocr"] = ocr_signal
            if ocr_signal.get("texts_found"):
                print(f"[verify] OCR found {len(ocr_signal['texts_found'])} text element(s): "
                      f"{ocr_signal['note']}")

        if getattr(self, "use_scene_comprehension", False) and is_final_attempt:
            print("[verify] Scene comprehension plugin (diagnostic)...")
            scene_signal = self.scene_verifier.verify(image, ctx.contract)
            count_decision.setdefault("signals", {})["scene_comprehension"] = scene_signal
            if scene_signal.get("missing"):
                print(f"[verify] Scene description missing: {scene_signal['missing']} "
                      f"(soft signal — not a rejection)")

        reward = (compute_reward(clip_scores["clip_norm"], vqa_scores["vqa_score"])
                  if vqa_scores["vqa_score"] is not None else None)
        ctx.reward_history.append({
            "attempt": ctx.attempt,
            "clip_cos": clip_scores["clip_cos"],
            "clip_norm": clip_scores["clip_norm"],
            "vqa_score": vqa_scores["vqa_score"],
            "reward": round(reward, 4) if reward is not None else None,
        })

        ctx.verification_report = {
            "accepted": False,  # Pipeline/reflect decide
            "hard_failures": hard_failures,
            "signals": count_decision.get("signals", {}),
            **clip_scores,
            "vqa_score": vqa_scores["vqa_score"],
            "vqa_vacuous": vqa_scores.get("vqa_vacuous", False),
            "vqa_status": vqa_scores["vqa_status"],
            "vqa_details": vqa_scores.get("vqa_details", []),
            "reward": round(reward, 4) if reward is not None else None,
            "spatial_result": spatial_result,
            "contract_evidence": contract_evidence,
            "detected_boxes": count_decision.get("detected_boxes", {}),
            "count_method": count_decision.get("count_method", "grounding_dino"),
        }
        ctx.save_artifact(f"verification_{ctx.attempt}.json", ctx.verification_report)
        spatial_label = f" spatial={spatial_result['verified']}/{spatial_result['total']}" if spatial_result['total'] > 0 else ""
        print(f"[verify] hard={len(hard_failures)} "
              f"clip_norm={clip_scores['clip_norm']} vqa={vqa_scores['vqa_score']} "
              f"R={reward if reward is not None else 'unavailable'}{spatial_label}")
        # RAM budget: CLIP scored its last point this attempt — evict the
        # singleton so it never co-resides with the SDXL pipe during the
        # next generate/refine (VM headroom keeps the host off the cliff).
        from src.utils import release_clip
        release_clip()
