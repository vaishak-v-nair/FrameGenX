"""Validate local training provenance before a candidate can be activated."""
import json
import math
from pathlib import Path

from src.training.common import file_sha256, project_path


def adapter_manifest(source: str, expected_model: str, allow_unqualified: bool = False,
                     require_manifest: bool = False) -> dict | None:
    path = project_path(source)
    manifest_file = path.with_suffix(".manifest.json")
    if not manifest_file.is_file():
        if require_manifest or path.is_relative_to(project_path("runs/training")):
            raise ValueError("Trained adapter has no provenance manifest")
        return None
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "framegenx-adapter-v1":
        raise ValueError("Unknown trained adapter manifest schema")
    if not path.is_file() or file_sha256(path) != manifest.get("adapter_sha256"):
        raise ValueError("Trained adapter weights are missing or changed")
    if manifest.get("base_model") != expected_model:
        raise ValueError("Trained adapter base checkpoint differs from the configured generator")
    # The training CLI never writes a qualified manifest. A later qualification
    # must carry a hashed, independent full-contract evaluation with no gaps.
    qualified = manifest.get("promotion_status") == "qualified"
    if qualified:
        evidence = manifest.get("qualification", {})
        report_file = (manifest_file.parent / evidence.get("report", "")).resolve()
        if (not report_file.is_relative_to(manifest_file.parent.resolve()) or not report_file.is_file()
                or file_sha256(report_file) != evidence.get("report_sha256")):
            raise ValueError("Qualified adapter is missing intact evaluation evidence")
        report = json.loads(report_file.read_text(encoding="utf-8"))
        bounds = report.get("paired_CSR_delta_95_ci")
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                not all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1 for x in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError("Qualification confidence interval is invalid or nonfinite")
        if (report.get("metric") != "full_contract_CSR" or report.get("unverified_requirements") != 0
                or report.get("independent_image_groups", 0) < 50 or report.get("seeds", 0) < 3
                or bounds[0] <= 0
                or report.get("adapter_sha256") != manifest["adapter_sha256"]
                or report.get("base_model") != expected_model):
            raise ValueError("Evaluation does not establish independent full-contract improvement")
    if not qualified and not allow_unqualified:
        raise ValueError("Trained candidate is unqualified; explicit experimental opt-in is required")
    return {"quality_status": "qualified" if qualified else "unverified",
            "experimental": not qualified, "training_run_id": manifest.get("run_id"),
            "adapter_sha256": manifest["adapter_sha256"], "base_model": manifest["base_model"]}
