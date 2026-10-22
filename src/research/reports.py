"""Bounded, allowlisted research views; no checkpoint, provider or GPU loading."""
import heapq
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from src.research.artifacts import ROOT

RESEARCH_ROOT = ROOT / "runs/research_cards"


def read_json(path, byte_limit=1_000_000, *, expected_sha256=None):
    if path.is_symlink() or path.stat().st_size > byte_limit:
        raise ValueError("Research document exceeds safe read boundaries")
    with path.open("rb") as stream:
        raw = stream.read(byte_limit+1)
    if len(raw) > byte_limit:
        raise ValueError("Research document exceeds safe read boundaries")
    # Hash the same bounded snapshot that is parsed; no unbounded second read.
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Research report checksum mismatch")
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("Research document must be an object")
    return value


def number(value, *, rate=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (rate and not 0 <= value <= 1):
        raise ValueError("Invalid research metric")
    return value


def count(value):
    if type(value) is not int or value < 0 or value > 1_000_000:
        raise ValueError("Invalid research denominator")
    return value


def view_completed(folder):
    report_file = folder / "report.json"
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema") != "framegenx-research-manifest-v1" or not isinstance(manifest.get("report_sha256"), str):
        raise ValueError("Unsupported research manifest")
    report = read_json(report_file, expected_sha256=manifest["report_sha256"])
    if (report.get("schema") != "framegenx-research-report-v1" or report.get("id") != folder.name
            or report.get("execution_status") != "completed" or report.get("quality_status") != "unverified"
            or report.get("study_scope") != "cards_cpu_feasibility_only"):
        raise ValueError("Unsupported research report identity or scope")
    evaluation = report["evaluation"]
    if (evaluation.get("full_contract_CSR") is not None or evaluation.get("reduction_claim") is not False
            or evaluation.get("scope") != "color_aware_cards_template_validity_only"):
        raise ValueError("Research evidence cannot qualify general image generation")
    results = {}
    for arm in ("baseline", "candidate"):
        data = evaluation["arms"][arm]
        denominator = count(data["n"])
        counts = {status: count(data["counts"][status]) for status in ("valid", "invalid", "unknown", "error")}
        rate = number(data["valid_rate"], rate=True)
        if denominator < 1 or sum(counts.values()) != denominator or abs(rate-counts["valid"]/denominator) > 1e-9:
            raise ValueError("Research metric disagrees with denominator")
        trained = report["deep_learning"][arm]
        results[arm] = {"n": denominator, "counts": counts, "valid_rate": rate,
                       "steps": count(trained["steps"]), "rho": number(trained["rho"])}
    test = report["classical_ml"]["metrics"]["test"]
    if report["sdxl_integration"].get("automatic_loading") is not False:
        raise ValueError("Experimental Cards checkpoints cannot be activated here")
    # Only these scalar/enum fields leave local research storage. No source paths,
    # SDK bodies, free text, prompts, commands or arbitrary metadata are exported.
    return {"id": folder.name, "execution_status": "completed", "quality_status": "unverified",
        "study_scope": "cards_cpu_feasibility_only", "dataset": "VSM Cards",
        "records": count(report["dataset"]["records"]),
        "classical_ml": {"test_n": count(test["n"]),
            "balanced_accuracy": number(test["balanced_accuracy"], rate=True),
            "roc_auc": number(test["roc_auc"], rate=True), "scope": "synthetic_corruption_diagnostic"},
        "deep_learning": results, "full_contract_CSR": None, "reduction_claim": False,
        "ihalla_status": "held_out_evaluation_unavailable", "sdxl_activation": "blocked",
        "report_integrity": "checksum_checked_not_semantic_qualification"}


def list_experiments(root=None, limit=10):
    root = Path(root) if root is not None else RESEARCH_ROOT
    if type(limit) is not int or not 1 <= limit <= 25:
        raise ValueError("Research list limit must be between 1 and 25")
    if root.is_symlink():
        raise ValueError("Research root cannot be a symlink")
    if not root.exists():
        return {"schema": "framegenx-research-list-v1", "experiments": [], "scan_truncated": False}
    candidates = []
    truncated = False
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= 1000:
                truncated = True
                break
            try:
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False) or str(uuid.UUID(entry.name)) != entry.name:
                    continue
            except ValueError:
                continue
            candidates.append(Path(entry.path))
    folders = heapq.nlargest(limit, candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    views = []
    for folder in folders:
        try:
            # The report is committed before its manifest. During that atomic
            # publication window, the running journal is still authoritative.
            if (folder / "report.json").is_file() and (folder / "manifest.json").is_file():
                views.append(view_completed(folder))
                continue
            progress = read_json(folder / "progress.json")
            status = progress["execution_status"]
            if (progress.get("id") != folder.name or status not in {"running", "failed", "interrupted"}
                    or progress.get("quality_status") != "unverified"):
                raise ValueError("Invalid research progress")
            views.append({"id": folder.name, "execution_status": status, "quality_status": "unverified",
                          "study_scope": "cards_cpu_feasibility_only", "dataset": "VSM Cards"})
        except (OSError, ValueError, KeyError, TypeError):
            views.append({"id": folder.name, "execution_status": "evidence_error", "quality_status": "unverified",
                          "study_scope": "unknown", "error_code": "FGX_RESEARCH_EVIDENCE_CORRUPT"})
    return {"schema": "framegenx-research-list-v1", "experiments": views, "scan_truncated": truncated}
