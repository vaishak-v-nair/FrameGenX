"""Run local count, geometry and segmented-color diagnostics with saved masks.

This measures operational verifier outcomes. It never qualifies an adapter or
replaces independent full-contract labels with a detector/color proxy.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch

from src.training.common import atomic_json, file_sha256, project_path
from src.training.evaluation import expected_images, validate_progress
from src.verification.qa import QAVerifier
from src.verification.instance_color import InstanceColorVerifier, SAM_WEIGHTS_SHA256, refine_count_decision
from src.verification.spatial_verifier import SpatialRelationVerifier


def summarize(records: list[dict]) -> dict:
    arms = {}
    for arm in ("baseline", "candidate"):
        selected = [r for r in records if r["arm"] == arm]
        known = [r for r in selected if r["status"] == "observed_diagnostics"]
        applicable = [r for r in known if r.get("chromatic_binding_supported") is not None]
        arms[arm] = {"images": len(selected), "observed_images": len(known),
            "unknown_images": len(selected)-len(known),
            "count_check_passes": sum(r["count_check_passed"] for r in known) if known else None,
            "count_and_geometry_passes": sum(r["count_check_passed"] and r["geometry_passed"] for r in known) if known else None,
            "chromatic_binding_applicable_images": len(applicable),
            "chromatic_binding_supported_images": sum(r["chromatic_binding_supported"] for r in applicable) if applicable else None}
    return {"arms": arms, "full_contract_CSR": None,
            "scope": "uncalibrated_local_verifier_diagnostics_only", "promotion_status": "unqualified"}


from src.core.device import exclusive_device


@exclusive_device
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = project_path(args.evaluation)
    generated = json.loads((root / "result.json").read_text())
    if generated["status"] != "generated_pending_independent_review":
        raise ValueError("Generation incomplete; refuse partial-image scoring")
    images = validate_progress(root, generated, generated["generation_signature"],
                              expected_images(generated["protocol"]), 512)
    if not images or len(images) > args.max_images:
        raise ValueError("Local review exceeds its explicit image budget")
    cases_file = PROJECT_ROOT / "configs/constraint_cases.json"
    cases = json.loads(cases_file.read_text())["cases"]
    signature = {"generation_result_sha256": file_sha256(root / "result.json"),
        "case_contract_sha256": file_sha256(cases_file), "segmentation_weights_sha256": SAM_WEIGHTS_SHA256,
        "code_sha256": {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in
            (Path(__file__).resolve(), PROJECT_ROOT / "src/verification/qa.py",
             PROJECT_ROOT / "src/verification/instance_color.py", PROJECT_ROOT / "src/verification/spatial_verifier.py")}}
    from huggingface_hub import hf_hub_download
    detector_root = Path(hf_hub_download("IDEA-Research/grounding-dino-tiny", "config.json", local_files_only=True)).parent
    signature["detector_snapshot"] = detector_root.name
    signature["detector_weights_sha256"] = file_sha256(detector_root / "model.safetensors")
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    result_file = root / f"local_verification_{fingerprint}.json"
    report = {"status": "running", "signature": signature, "records": [], "full_contract_CSR": None,
              "scope": "local_count_geometry_and_segmented_chromatic_diagnostics"}
    if result_file.exists():
        if not args.resume:
            raise FileExistsError("Use --resume for existing verification evidence")
        report = json.loads(result_file.read_text())
        if report["signature"] != signature:
            raise ValueError("Verifier provenance changed")
    existing = {r["file"]: r for r in report["records"] if r["status"] == "observed_diagnostics"}
    for name, record in existing.items():
        if name not in images or record["image_sha256"] != images[name]["image_sha256"]:
            raise ValueError("Verification resume image changed")
        for instance in record["color"]["instances"]:
            path = root / record["mask_directory"] / instance["mask_file"]
            if file_sha256(path) != instance["mask_sha256"]:
                raise ValueError("Verification resume mask changed")
    report["records"] = list(existing.values())
    print(f"Local verification report: {result_file}", flush=True)
    detector, color, geometry = QAVerifier(), InstanceColorVerifier(), SpatialRelationVerifier()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for name, item in images.items():
        if name in existing:
            continue
        record = {k: item[k] for k in ("case_id", "seed", "arm", "file", "image_sha256")}
        try:
            if time.monotonic()-started > 900:
                raise TimeoutError("Local diagnostic wall-clock budget exceeded")
            case = next(c for c in generated["protocol"]["cases"] if c["id"] == item["case_id"])
            contract = deepcopy(case.get("contract") or cases[item["case_id"]])
            contract["prompt"] = case["prompt"]
            counted = detector.verify(root / name, contract)
            with Image.open(root / name) as image:
                image = image.convert("RGB")
                mask_directory = f"local_masks_{fingerprint}/{Path(name).stem}"
                colors = color.verify(image, contract, counted["detected_boxes"], root / mask_directory)
                raw_counted = counted
                counted = refine_count_decision(counted, colors)
                spatial = geometry.verify(contract.get("spatial_relationships") or [],
                    counted["detected_boxes"], image.width, image.height)
            from src.verification.instance_color import CHROMATIC
            applicable = [info for info in colors["classes"].values()
                          if info["expected_colors"] and all(c in CHROMATIC for c in info["expected_colors"])]
            record.update(status="observed_diagnostics", count=counted, raw_count=raw_counted, spatial=spatial, color=colors,
                count_check_passed=counted["accepted"], geometry_passed=spatial["passed"],
                mask_directory=mask_directory,
                chromatic_binding_supported=all(c["status"] == "supported_diagnostic" for c in applicable) if applicable else None)
            if torch.cuda.max_memory_reserved()/1e9 > 5.5:
                raise RuntimeError("Local verifier exceeded the 5.5 GB VRAM budget")
        except Exception as error:
            record.update(status="unavailable_diagnostics", error_type=type(error).__name__)
            traceback.print_exc()
        report["records"].append(record)
        report.update(summary=summarize(report["records"]),
            peak_reserved_gb=torch.cuda.max_memory_reserved()/1e9, elapsed_seconds=time.monotonic()-started)
        atomic_json(result_file, report)
        print(f"Verified {name}: {record['status']}; counts={record.get('count',{}).get('signals',{}).get('detected_counts')}", flush=True)
    report["status"] = ("completed_diagnostics" if all(r["status"] == "observed_diagnostics" for r in report["records"])
                         else "incomplete_diagnostics")
    atomic_json(result_file, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["status"] == "completed_diagnostics" else 1


if __name__ == "__main__":
    raise SystemExit(main())
