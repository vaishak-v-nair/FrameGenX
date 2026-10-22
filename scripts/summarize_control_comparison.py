"""Summarize saved, hash-checked regional-controller diagnostics and images.

This report preserves missing evidence and never turns operational checks into
full-contract CSR or a claim about trained adapter weights.
"""
import argparse
from html import escape
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.common import atomic_json, file_sha256, project_path
from src.training.evaluation import expected_images, validate_progress


def load_comparison(root: Path, verification_file: Path) -> dict:
    generation_file = root / "result.json"
    generated = json.loads(generation_file.read_text(encoding="utf-8"))
    if generated["status"] != "generated_pending_independent_review":
        raise ValueError("Generation is incomplete")
    images = validate_progress(root, generated, generated["generation_signature"],
                               expected_images(generated["protocol"]), 512)
    verified = json.loads(verification_file.read_text(encoding="utf-8"))
    if verified["signature"]["generation_result_sha256"] != file_sha256(generation_file):
        raise ValueError("Verification describes a different generation journal")
    records = {}
    for record in verified["records"]:
        name = record["file"]
        if name in records or name not in images or record["image_sha256"] != images[name]["image_sha256"]:
            raise ValueError("Duplicate, unknown, or changed diagnostic image")
        if any(record.get(key) != images[name][key] for key in ("case_id", "seed", "arm")):
            raise ValueError("Diagnostic case/seed/arm identity differs")
        if record["status"] == "observed_diagnostics":
            for instance in record["color"]["instances"]:
                mask = root / record["mask_directory"] / instance["mask_file"]
                if file_sha256(mask) != instance["mask_sha256"]:
                    raise ValueError("Saved diagnostic mask changed")
        records[name] = record
    pairs = []
    for case in generated["protocol"]["cases"]:
        for seed in generated["protocol"]["seeds"]:
            pair = {"case_id": case["id"], "seed": seed, "prompt": case["prompt"],
                    "requirements": case["requirements"]}
            for arm in ("baseline", "candidate"):
                name = f"{case['id']}_{seed}_{arm}.png"
                record = records.get(name, {})
                observed = record.get("status") == "observed_diagnostics"
                chromatic = record.get("chromatic_binding_supported") if observed else None
                pair[arm] = {"file": name, "image_sha256": images[name]["image_sha256"],
                    "observed": observed,
                    "counts": record["count"]["signals"]["detected_counts"] if observed else None,
                    "count_check_passed": record.get("count_check_passed") if observed else None,
                    "geometry_passed": record.get("geometry_passed") if observed else None,
                    "chromatic_binding_supported": chromatic,
                    "joint_operational_checks_passed": bool(record["count_check_passed"] and
                        record["geometry_passed"] and chromatic is not False) if observed else None,
                    "color_classes": record["color"]["classes"] if observed else None}
            pairs.append(pair)
    summary = {}
    for arm in ("baseline", "candidate"):
        known = [p[arm] for p in pairs if p[arm]["observed"]]
        summary[arm] = {"total_images": len(pairs), "observed_images": len(known),
            "unknown_images": len(pairs)-len(known),
            "count_check_passes": sum(p["count_check_passed"] for p in known) if known else None,
            "count_and_geometry_passes": sum(p["count_check_passed"] and p["geometry_passed"] for p in known) if known else None,
            "chromatic_applicable_images": sum(p["chromatic_binding_supported"] is not None for p in known),
            "chromatic_supported_images": sum(p["chromatic_binding_supported"] is True for p in known) if known else None,
            "joint_operational_check_passes": sum(p["joint_operational_checks_passed"] for p in known) if known else None}
    matched = [p for p in pairs if p["baseline"]["observed"] and p["candidate"]["observed"]]
    changes = {"matched_pairs": len(matched), "improved_pairs": 0, "regressed_pairs": 0, "unchanged_pairs": 0}
    for pair in matched:
        delta = int(pair["candidate"]["joint_operational_checks_passed"]) - int(pair["baseline"]["joint_operational_checks_passed"])
        changes["improved_pairs" if delta > 0 else "regressed_pairs" if delta < 0 else "unchanged_pairs"] += 1
    return {"schema": "framegenx-controlled-comparison-v1", "scope": "uncalibrated_local_operational_diagnostics",
        "full_contract_CSR": None, "trained_adapter_improvement": None, "adapters_enabled": generated["adapters_enabled"],
        "promotion_status": "unqualified", "generation_directory": str(root),
        "generation_result_sha256": file_sha256(generation_file),
        "verification_file": str(verification_file), "verification_sha256": file_sha256(verification_file),
        "native_pipeline_parity": generated.get("native_pipeline_parity"),
        "generation_seconds": generated["elapsed_seconds"], "generation_peak_reserved_gb": generated["peak_reserved_gb"],
        "verification_seconds": verified["elapsed_seconds"], "verification_peak_reserved_gb": verified["peak_reserved_gb"],
        "summary": summary, "paired_changes": changes, "pairs": pairs}


def render_gallery(comparison: dict, destination: Path) -> str:
    root = Path(comparison["generation_directory"])
    def link(path):
        return escape(os.path.relpath(path, destination.parent).replace("\\", "/"), quote=True)
    cards = []
    for pair in comparison["pairs"]:
        arms = []
        for arm in ("baseline", "candidate"):
            item = pair[arm]
            verdict = ("passed" if item["joint_operational_checks_passed"] else "failed") if item["observed"] else "unavailable"
            details = f"Counts: {item['counts']}; geometry: {item['geometry_passed']}; chromatic binding: {item['chromatic_binding_supported']}"
            arms.append(f'<figure><figcaption>{escape(arm)} — operational checks {verdict}</figcaption>'
                f'<a href="{link(root/item["file"])}"><img loading="lazy" src="{link(root/item["file"])}" alt="{escape(pair["case_id"])} {arm}, seed {pair["seed"]}"></a>'
                f'<p>{escape(details)}</p><small>SHA256 {item["image_sha256"]}</small></figure>')
        requirements = "".join(f"<li>{escape(r)}</li>" for r in pair["requirements"])
        cards.append(f'<section><h2>{escape(pair["case_id"])} · seed {pair["seed"]}</h2><p>{escape(pair["prompt"])}</p>'
                     f'<ul>{requirements}</ul><div class="pair">{"".join(arms)}</div></section>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + \
        '<title>FrameGenX constraint-control comparison</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#17202b;background:#f4f6f8}h1{line-height:1.15}section{background:white;border:1px solid #cbd2d9;padding:20px;margin:28px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:0;min-width:0}figcaption{font-weight:650;margin:10px 0}img{width:100%;height:auto}small{display:block;overflow-wrap:anywhere}pre{white-space:pre-wrap}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>' + \
        '<main><h1>Constraint-control comparison</h1><p>Native saved images, unchanged pixels. Same SDXL NF4 backbone, Euler steps, CFG and paired seeds. Trained adapters disabled.</p>' + \
        '<p>Labels describe local detector, segmentation, hue and box-geometry diagnostics. They are uncalibrated and do not establish full-contract CSR, visual quality, or trained-adapter improvement.</p>' + \
        '<pre>' + escape(json.dumps(comparison["summary"], indent=2)) + '</pre>' + "".join(cards) + '</main></html>'


def track_comparison(comparison: dict, json_file: Path) -> None:
    from src.training.tracking import ExperimentTracker
    tracker = ExperimentTracker("runs/mlflow.db", "framegenx-constraint-control", json_file.stem)
    try:
        tracker.client.set_tag(tracker.run_id, "metric_scope", comparison["scope"])
        tracker.client.set_tag(tracker.run_id, "promotion_status", "unqualified")
        tracker.client.set_tag(tracker.run_id, "full_contract_CSR", "unmeasured")
        tracker.params({"generation_result_sha256": comparison["generation_result_sha256"],
            "verification_sha256": comparison["verification_sha256"],
            "adapters_enabled": comparison["adapters_enabled"],
            "case_groups": len({p["case_id"] for p in comparison["pairs"]}),
            "seeds": sorted({p["seed"] for p in comparison["pairs"]})})
        metrics = {f"{arm}_{name}": value for arm, info in comparison["summary"].items()
                   for name, value in info.items() if value is not None}
        metrics.update({k: comparison[k] for k in ("generation_seconds", "generation_peak_reserved_gb",
            "verification_seconds", "verification_peak_reserved_gb")})
        tracker.metrics(metrics)
        comparison["mlflow_run_id"] = tracker.run_id
        atomic_json(json_file, comparison)
        tracker.artifact(json_file)
        tracker.artifact(Path(comparison["verification_file"]))
        tracker.artifact(Path(comparison["generation_directory"]) / "result.json")
    except BaseException:
        tracker.finish("FAILED")
        raise
    else:
        tracker.finish("FINISHED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output-prefix", default="reports/constraint_comparison")
    parser.add_argument("--track", action="store_true", help="Record a new local MLflow assessment run")
    args = parser.parse_args()
    result = load_comparison(project_path(args.evaluation), project_path(args.verification))
    prefix = project_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(prefix.with_suffix(".json"), result)
    prefix.with_suffix(".html").write_text(render_gallery(result, prefix.with_suffix(".html")), encoding="utf-8")
    if args.track:
        track_comparison(result, prefix.with_suffix(".json"))
    print(json.dumps({"summary": result["summary"], "paired_changes": result["paired_changes"]}, indent=2))


if __name__ == "__main__":
    main()
