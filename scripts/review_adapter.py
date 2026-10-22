"""Blind VLM diagnostics for paired pilot images; never a qualified CSR score."""
import argparse
import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import atomic_json, file_sha256, project_path


def parse_answers(reply: str, total: int) -> list[dict]:
    if total < 1:
        raise ValueError("No requirements to review")
    text = reply.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Incomplete JSON fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"answers"} or not isinstance(value["answers"], list):
        raise ValueError("Invalid vision answer schema")
    answers = value["answers"]
    if len(answers) != total:
        raise ValueError("Vision response omitted requirements")
    ids = set()
    for answer in answers:
        if (not isinstance(answer, dict) or set(answer) != {"index", "answer", "observation"}
                or type(answer["index"]) is not int or not 0 <= answer["index"] < total
                or answer["index"] in ids or not isinstance(answer["answer"], str)
                or answer["answer"] not in {"yes", "no", "unknown"}
                or not isinstance(answer["observation"], str)):
            raise ValueError("Invalid or duplicate vision requirement answer")
        ids.add(answer["index"])
    return sorted(answers, key=lambda a: a["index"])


def summarize(records: list[dict]) -> dict:
    arms = {}
    paired = {}
    for arm in ("baseline", "candidate"):
        selected = [r for r in records if r["arm"] == arm]
        total = sum(len(r["answers"]) for r in selected)
        yes = sum(a["answer"] == "yes" for r in selected for a in r["answers"])
        no = sum(a["answer"] == "no" for r in selected for a in r["answers"])
        unknown = total - yes - no
        passes = sum(bool(r["answers"]) and all(a["answer"] == "yes" for a in r["answers"]) for r in selected)
        arms[arm] = {"images": len(selected), "requested_requirements": total,
                     "supported": yes, "contradicted": no, "unknown": unknown,
                     "answered_requirements": yes + no,
                     "uncalibrated_all_yes_images": passes if yes + no else None,
                     "uncalibrated_supported_requirement_rate": yes / total if yes + no else None}
        for record in selected:
            paired.setdefault((record["case_id"], record["seed"]), {})[arm] = {
                "all_yes": bool(record["answers"]) and all(a["answer"] == "yes" for a in record["answers"]),
                "known": bool(record["answers"]) and not any(a["answer"] == "unknown" for a in record["answers"])}
    comparable = [p for p in paired.values() if set(p) == {"baseline", "candidate"}]
    answered_pairs = [p for p in comparable if p["baseline"]["known"] and p["candidate"]["known"]]
    return {"arms": arms, "paired_cases": len(comparable),
            "paired_fully_answered_cases": len(answered_pairs),
            "uncalibrated_candidate_only_passes": sum(p["candidate"]["all_yes"] and not p["baseline"]["all_yes"]
                                                       for p in answered_pairs) if answered_pairs else None,
            "uncalibrated_baseline_only_passes": sum(p["baseline"]["all_yes"] and not p["candidate"]["all_yes"]
                                                      for p in answered_pairs) if answered_pairs else None,
            "full_contract_CSR": None, "quality_status": "unverified", "promotion_status": "unqualified",
            "scope": "uncalibrated_vision_diagnostics_not_independent_human_validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    root = project_path(args.evaluation)
    generated = json.loads((root / "result.json").read_text())
    if generated["status"] != "generated_pending_independent_review":
        raise ValueError("Generation did not complete; refuse to score partial or failed output")
    images = generated["images"][:]
    if not images or len(images) > args.max_images:
        raise ValueError("Review exceeds its explicit request budget or has no images")
    for item in images:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != item["image_sha256"]:
            raise ValueError("Review image changed or is outside its run directory")
    from dotenv import load_dotenv
    from src.core.llm_router import LLMRouter
    load_dotenv(PROJECT_ROOT / ".env")
    router = LLMRouter()
    random.Random(731).shuffle(images)
    records = []
    started = time.monotonic()
    for index, item in enumerate(images):
        image_data = base64.b64encode((root / item["file"]).read_bytes()).decode()
        instructions = (
            "Inspect this image independently. For each indexed statement below, answer yes only if visibly supported, "
            "no if visibly contradicted, and unknown if it cannot be determined. Do not assume the statements are true. "
            "Count every visible object; consider overlapping objects separately. Ignore aesthetic preference. "
            "Return only JSON with key answers, an array of objects with exactly index (integer), answer "
            "(yes/no/unknown), and observation (short visible evidence). Statements: " +
            json.dumps([{"index": i, "statement": statement} for i, statement in enumerate(item["requirements"])])
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_data}}]}]
        # Provider exception text can include credential fragments. Capture it
        # in memory and retain only safe routing labels and exception types.
        captured = io.StringIO()
        record = {k: item[k] for k in ("case_id", "seed", "arm", "file", "image_sha256", "requirements")}
        try:
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("Review request budget expired")
            with redirect_stdout(captured):
                reply = router.generate_vision(messages, temperature=0.0)
            record["answers"] = parse_answers(reply, len(item["requirements"]))
            record["status"] = "answered"
        except Exception as exc:
            record["status"] = "unavailable_or_invalid"
            record["error_type"] = type(exc).__name__
            record["provider_failures"] = router.last_failures[:]
            record["answers"] = [{"index": i, "answer": "unknown", "observation": "No valid independent response"}
                                 for i in range(len(item["requirements"]))]
        routes = re.findall(r"Routing vision request to ([A-Za-z0-9 _-]+) \(([^)\r\n]+)\)", captured.getvalue())
        record["requested_provider_and_model"] = list(routes[-1]) if routes else None
        records.append(record)
        status = "reviewed_diagnostics_only" if all(r["status"] == "answered" for r in records) else "incomplete_diagnostics"
        atomic_json(root / "vision_review.json", {"status": status, "records": records, "summary": summarize(records),
            "blind_arm_order_seed": 731, "reviewer_code_sha256": file_sha256(Path(__file__))})
        print(f"Reviewed blind image {index+1}/{len(images)}: {record['status']}", flush=True)
    print(json.dumps(summarize(records), indent=2), flush=True)
    return 0 if all(r["status"] == "answered" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
