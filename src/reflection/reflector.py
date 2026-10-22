"""
ReflectStage — SRRL-style reward gating + corrective prompt edits.

Reflective loop, not RL policy training. The Pipeline owns the loop; this
stage only computes {accept, prompt_edits} from the verify report.

Acceptance rule: accept = (no hard failures) AND (R >= tau).
Hard failures always block regardless of reward.

Corrective edit table (design doc — edits recomputed FRESH each attempt,
never accumulated; prepending "exactly {n}..." twice would be garbage):

  count_mismatch (detected < expected) -> prepend "exactly {n} {class}, ..."
  count_mismatch (detected > expected) -> prepend "only {n} {class}, ..."
  CLIP_norm < 0.5                      -> append subject emphasis
  VQA existence "no"                   -> move entity to front, "prominent"
  VQA color mismatch                   -> "({color} colored {class})"

contract["repair_policy"] (emitted by the extractor, paper-contract parity)
refines the count edits with PROMPT-SAFE suffix fragments (e.g.
object_missing -> "prominently featured"). Raw meta-action names are
forbidden in policy actions by construction — they would leak into the
drawing prompt.

class_name is the extractor's surface form — never append "s".
"""
from typing import Dict, Any, List

from src.core.stage import Stage, register_stage
from src.core.context import RunContext


from src.core.llm_router import LLMRouter

def build_corrective_edits(report: Dict[str, Any], contract: Dict[str, Any],
                           vqa_details: List[Dict[str, Any]] = None, use_llm: bool = True) -> List[str]:
    edits: List[str] = []

    repair_policy = contract.get("repair_policy", {}).get("mappings", [])
    policy_map = {m.get("failure"): m.get("action") for m in repair_policy}

    # Single router instance for all LLM fallbacks in this call
    _router = None

    for failure in report.get("hard_failures", []):
        f_type = failure.get("type")
        cls = failure.get("class_name", "object")
        n = failure.get("expected", 1)

        mapped_action = None
        if f_type == "attribute_mismatch":
            expected = failure.get("expected") or []
            if isinstance(expected, list) and expected and len(set(expected)) == 1:
                edits.append(f"{expected[0]} colored {cls}, correct color on every instance")
                continue
        if f_type == "count_mismatch":
            # Pass all count mismatches directly to the LLM.
            # Mechanical string injections like "exactly 1 person, each clearly separated"
            # destroy the aesthetics of the prompt and result in generic generations.
            mapped_action = None

        if mapped_action:
            edits.append(mapped_action)
        elif not use_llm:
            if f_type == "count_mismatch":
                edits.append(f"exactly {n} {cls}, individually visible")
        else:
            # LLM Fallback for unmapped hard failures
            if _router is None:
                _router = LLMRouter()
            instruction = (
                f"The generated image failed a verification check.\n"
                f"Failure type: {f_type}\nClass: {cls}\nExpected: {n}\n"
                f"Repair Policy: {policy_map}\n"
                f"Generate a short, corrective prompt prefix (1-5 words) to fix this failure.\n"
                f"Output ONLY the prefix."
            )
            try:
                llm_edit = _router.generate([{"role": "user", "content": instruction}], temperature=0.3)
                edits.append(llm_edit.strip())
            except Exception as e:
                print(f"[reflect] LLM repair failed: {e}")

    if report.get("clip_norm", 1.0) < 0.5:
        if "low_alignment" in policy_map:
            edits.append(policy_map["low_alignment"])
        # Keep corrective text tied to the user's subject; retrieved
        # captions are not generation instructions.
        subject = (contract.get("parsed") or {}).get("primary_subject") or contract.get("prompt", "")
        if subject:
            edits.append(f"{subject}, prominent, clearly visible")

    for d in (vqa_details or []):
        if d.get("correct") is None:
            continue
        if d.get("correct"):
            continue
        if d.get("type") == "existence":
            edits.append(f"{d.get('class_name', 'subject')}, prominent, clearly visible")
        elif d.get("type") == "color":
            edits.append(f"({d.get('expected')} colored {d.get('class_name')})")

    # Dedupe preserving order
    seen = set()
    return [e for e in edits if not (e in seen or seen.add(e))]


@register_stage
class ReflectStage(Stage):
    name = "reflect"

    def __init__(self, config):
        super().__init__(config)
        r_cfg = config.get("reflection", {}) or {}
        self.mode = r_cfg.get("mode", "hard_only")
        self.tau = float(r_cfg.get("tau", 0.575))
        self.tau_source = r_cfg.get("tau_source", "paper")

    def run(self, ctx: RunContext) -> None:
        report = ctx.verification_report or {}
        reward = report.get("reward", 0.0)
        failures = report.get("hard_failures")
        hard_ok = isinstance(failures, list) and not failures

        if self.mode == "hard_only":
            accept = hard_ok
        else:
            # Legacy reward-gated mode
            accept = hard_ok and reward is not None and reward >= self.tau

        edits: List[str] = []
        if not accept:
            # Pull vqa_details from the saved verification artifact fields
            import json
            vqa_details = []
            vpath = ctx.run_dir / f"verification_{ctx.attempt}.json"
            if vpath.exists():
                with open(vpath, "r", encoding="utf-8") as f:
                    vqa_details = (json.load(f)).get("vqa_details", [])
            if self.config.get("reflection", {}).get("deterministic_only", False):
                edits = build_corrective_edits(report, ctx.contract or {}, vqa_details, use_llm=False)
            else:
                edits = build_corrective_edits(report, ctx.contract or {}, vqa_details)

        ctx.decision = {
            "accept": accept,
            "reward": reward,
            "mode": self.mode,
            "tau": self.tau if self.mode == "reward_gate" else None,
            "tau_source": self.tau_source,
            "hard_ok": hard_ok,
            "prompt_edits": edits,
        }
        mode_label = f"mode={self.mode}" if self.mode == "hard_only" else f"tau={self.tau}"
        print(f"[reflect] R={reward if reward is not None else 'unavailable'} {mode_label} hard_ok={hard_ok} "
              f"-> {'ACCEPT' if accept else 'RETRY'} edits={len(edits)}")
