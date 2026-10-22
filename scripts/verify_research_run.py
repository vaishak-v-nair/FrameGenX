"""Verify real saved artifacts and the mounted local API, without generation."""
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from src.research.artifacts import atomic_json, sha256
from src.research.reports import read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run_id, evidence_root):
    require(str(uuid.UUID(run_id)) == run_id, "Use a canonical research UUID")
    folder = ROOT / "runs/research_cards" / run_id
    report = read_json(folder / "report.json")
    inventory = read_json(folder / "manifest.json")
    require(inventory["report_sha256"] == sha256(folder / "report.json"), "Report hash mismatch")
    checked = 0
    for item in inventory["artifacts"]:
        path = (folder / item["file"]).resolve()
        require(path.is_relative_to(folder.resolve()), "Artifact escaped its run")
        require(sha256(path) == item["sha256"], "Artifact hash mismatch")
        checked += 1
    from src.research.experiment import source_identity, runtime_identity
    import torch
    from safetensors import safe_open
    protocol = read_json(folder / "protocol.json")
    require(protocol["source"] == source_identity(), "Current research code differs from the executed source")
    current_runtime = runtime_identity()
    require(all(protocol["runtime"][key] == current_runtime[key] for key in ("python", "system", "machine", "packages")),
            "Current research packages/runtime differ from the executed environment")
    require(protocol["runtime"]["torch_threads"] == protocol["config"]["threads"]
            and protocol["runtime"]["strict_determinism"] is True, "Executed runtime did not honor the frozen CPU policy")
    for name, expected in protocol["source"].items():
        require(sha256(folder / "source" / name) == expected, "Saved source snapshot differs from its frozen identity")
    require(report["execution_status"] == "completed" and report["quality_status"] == "unverified", "Wrong lifecycle/qualification")
    require(report["evaluation"]["full_contract_CSR"] is None and report["evaluation"]["reduction_claim"] is False, "Unsupported quality claim")
    require(report["sdxl_integration"]["automatic_loading"] is False, "Experimental weights activated")
    require(report["deep_learning"]["baseline"]["initial_sha256"] == report["deep_learning"]["candidate"]["initial_sha256"], "Paired initializations differ")
    shapes = {}
    for arm in ("baseline", "candidate"):
        trained = report["deep_learning"][arm]
        require(trained["steps"] == protocol["config"]["steps"], "Training budget differs")
        with safe_open(str(folder / arm / "checkpoint.safetensors"), framework="pt", device="cpu") as weights:
            require(weights.metadata()["schema"] == "framegenx-cards-denoiser-v1", "Wrong checkpoint type")
            shapes[arm] = weights.get_slice("output.2.weight").get_shape()
            require(shapes[arm][0] == 6, "Cards model must have separate RGB epsilon/variance outputs")
        samples = read_json(folder / arm / "sampling.json")
        require(samples["status"] == "completed", "Sampling is incomplete")
        require([sample["seed"] for sample in samples["samples"]] == protocol["config"]["sample_seeds"], "Missing/reordered planned samples")
        for sample in samples["samples"]:
            require(sample["sha256"] == sha256(folder / arm / "samples" / f"seed_{sample['seed']}.png"), "Sample hash mismatch")
    ids = [report["classical_ml"]["mlflow_run_id"], *[report["deep_learning"][arm]["mlflow_run_id"] for arm in ("baseline", "candidate")]]
    database = (ROOT / "runs/mlflow_research.db").as_uri() + "?mode=ro"
    with sqlite3.connect(database, uri=True) as connection:
        statuses = {value: connection.execute("SELECT status FROM runs WHERE run_uuid=?", (value,)).fetchone()[0] for value in ids}
    require(all(value == "FINISHED" for value in statuses.values()), "An actual MLflow lifecycle did not finish")

    denied = []
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_dns = socket.getaddrinfo

    def deny(*args, **kwargs):
        denied.append(True)
        raise AssertionError("Artifact/API verification must not call providers or services")

    from scripts.research_test_guard import _allowed

    def connect(instance, address):
        if not _allowed(address):
            deny()
        return original_connect(instance, address)

    def connect_ex(instance, address):
        if not _allowed(address):
            deny()
        return original_connect_ex(instance, address)

    def dns(host, port, *args, **kwargs):
        if host not in (None, "localhost", "127.0.0.1", "::1"):
            deny()
        return original_dns(host, port, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, dns
    try:
        from fastapi.testclient import TestClient
        from src.api import app
        with TestClient(app) as client:
            response = client.get("/api/research/experiments?limit=25")
            require(response.status_code == 200, "Mounted application research read failed")
            body = response.json()
            actual = next(run for run in body["experiments"] if run["id"] == run_id)
            require(actual["execution_status"] == "completed" and actual["sdxl_activation"] == "blocked", "UI/API state disagrees with actual research evidence")
            require(client.get("/api/research/experiments?limit=0").status_code == 422, "Unbounded list query accepted")
            require(client.post("/api/research/experiments", json={}).status_code == 405, "Read-only research endpoint accepted mutation")
            page = client.get("/app/")
            require(page.status_code == 200 and 'type="module"' in page.text, "Built frontend is not served")
            import re
            asset = re.search(r'src="([^"]+\.js)"', page.text)
            require(asset is not None, "Built entrypoint is missing")
            require(client.get(asset[1]).status_code == 200, "Built frontend entrypoint cannot be read")
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = original_connect, original_connect_ex, original_dns
    require(not denied and not torch.cuda.is_initialized(), "Verification initialized a forbidden service or CUDA")
    evidence_root = Path(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    payload_path = evidence_root / "mounted_api_payload.json"
    atomic_json(payload_path, body)
    node_code = "import fs from 'node:fs'; const {parseResearchList}=await import(process.argv[1]); parseResearchList(JSON.parse(fs.readFileSync(process.argv[2],'utf8'))); console.log('frontend_contract_verified');"
    node = subprocess.run(["node", "--input-type=module", "-e", node_code,
                           (ROOT / "frontend/src/researchState.js").as_uri(), str(payload_path.resolve())],
                          capture_output=True, text=True, check=True, timeout=15)
    result = {"run_id": run_id, "status": "passed_local_artifact_and_mounted_api_checks",
              "checked_artifacts": checked, "checkpoint_output_shapes": shapes, "actual_mlflow_statuses": statuses,
              "mounted_api": "read, limits and mutation rejection passed", "built_frontend": "HTML/entrypoint served; actual API payload accepted by frontend parser",
              "frontend_parser": node.stdout.strip(), "external_calls": len(denied), "cuda_initialized": torch.cuda.is_initialized(),
              "limits": ["No rendered browser, accessibility, GPU generation, hosted or human calibration proof", "Image quality remains unverified"]}
    atomic_json(evidence_root / "integration_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-root", default="reports/research_execution_2026-10-16")
    arguments = parser.parse_args()
    verify(arguments.run_id, arguments.evidence_root)
