"""API regressions that run in process without starting a GPU generation."""
import base64
import json
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import api
from src.core.context import RunContext
from src.core.db import RunTrackerDB


def test_history_pagination_rejects_unbounded_and_negative_limits():
    client = TestClient(api.app)
    for path in (
        "/api/runs?page=0",
        "/api/runs?per_page=0",
        "/api/runs?per_page=-1",
        "/api/runs?per_page=100000",
    ):
        assert client.get(path).status_code == 422


def test_only_configured_local_frontend_origin_can_use_cors():
    client = TestClient(api.app)
    headers = {"Access-Control-Request-Method": "POST"}
    allowed = client.options("/jobs", headers={**headers, "Origin": "http://127.0.0.1:5173"})
    denied = client.options("/jobs", headers={**headers, "Origin": "https://example.invalid"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_artifacts_require_an_existing_uuid_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "runs_dir", tmp_path)
    client = TestClient(api.app)
    assert client.get("/artifacts/not-a-run").status_code == 404
    assert client.get(f"/artifacts/{uuid.uuid4()}").status_code == 404

    run_id = str(uuid.uuid4())
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "contract_v2.json").write_text(json.dumps({"prompt": "a cat"}))
    (run_dir / "decision_0.json").write_text(json.dumps({"accept": False}))
    response = client.get(f"/artifacts/{run_id}")
    assert response.status_code == 200
    assert response.json()["attempts"][0]["decision"]["accept"] is False


def test_runs_route_serves_previews_but_not_database_or_json(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "runs_dir", tmp_path)
    run_id = str(uuid.uuid4())
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (tmp_path / "framegenx.db").write_bytes(b"private-db")
    (run_dir / "contract_v2.json").write_text('{"prompt":"a cat"}')
    (run_dir / "generated_0.png").write_bytes(b"preview")
    client = TestClient(api.app)
    assert client.get(f"/runs/{run_id}/generated_0.png").content == b"preview"
    assert client.get("/runs/framegenx.db").status_code == 404
    assert client.get(f"/runs/{run_id}/contract_v2.json").status_code == 404
    assert client.get(f"/runs/{run_id}/../framegenx.db").status_code == 404


def test_job_input_and_busy_gpu_are_reported_without_starting_generation():
    client = TestClient(api.app)
    assert client.post("/jobs", json={"prompt": "   "}).status_code == 400
    assert client.post("/jobs", json={"prompt": "x" * 2001}).status_code == 422
    assert api.GPU_BUSY.acquire(blocking=False)
    try:
        response = client.post("/jobs", json={"prompt": f"qa-{uuid.uuid4()}"})
        assert response.status_code == 409
    finally:
        api.GPU_BUSY.release()


def test_contract_submission_is_not_silently_ignored():
    client = TestClient(api.app)
    response = client.post("/generate", json={
        "contract": {"created_at": "2026-01-01T00:00:00Z", "prompt": "two cats"}
    })
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_worker_failure_does_not_expose_provider_request_credentials(monkeypatch, caplog):
    sentinel = "synthetic-credential-must-stay-private"
    def failure(*args, **kwargs):
        raise RuntimeError(f"provider request failed https://provider.invalid/?key={sentinel}")
    monkeypatch.setattr(api.pipeline, "run", failure)
    job = {"prompt": "a cat", "events": [], "status": "running"}
    assert api.GPU_BUSY.acquire(blocking=False)
    api._run_job(job)
    assert job["status"] == "error" and job["error_code"] == "generation_failed"
    assert sentinel not in job["error"] and sentinel not in caplog.text
    assert not api.GPU_BUSY.locked()


def test_cached_sync_response_uses_generated_image_not_layout_mask(tmp_path, monkeypatch):
    run_id = str(uuid.uuid4())
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "global_mask_0.png").write_bytes(b"mask-bytes")
    (run_dir / "generated_0.png").write_bytes(b"image-bytes")
    monkeypatch.setattr(api, "runs_dir", tmp_path)
    monkeypatch.setattr(api.pipeline.db, "get_run_by_prompt", lambda prompt, cache_key=None: {
        "run_id": run_id, "repair_loops": 0, "is_successful": True,
        "verification_report_json": "{}",
    })

    response = TestClient(api.app).post("/generate", json={"prompt": "a cat"})
    assert response.status_code == 200
    assert base64.b64decode(response.json()["image_base64"]) == b"image-bytes"


def test_sync_response_image_matches_its_final_verification_report(tmp_path, monkeypatch):
    (tmp_path / "generated_0.png").write_bytes(b"higher-reward-first-draft")
    (tmp_path / "generated_1.png").write_bytes(b"last-verified-draft")
    ctx = SimpleNamespace(
        run_id=str(uuid.uuid4()), run_dir=tmp_path, attempt=1, is_successful=False,
        reward_history=[{"attempt": 0, "reward": .9}, {"attempt": 1, "reward": .5}],
        verification_report={"accepted": False, "signals": {"detected_counts": {"cat": 2}}},
    )
    monkeypatch.setattr(api.pipeline.db, "get_run_by_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(api.pipeline, "run", lambda prompt: ctx)
    response = TestClient(api.app).post("/generate", json={"prompt": "a cat"})
    assert response.status_code == 200
    assert base64.b64decode(response.json()["image_base64"]) == b"last-verified-draft"
    assert response.json()["verification_report"] == ctx.verification_report


def test_run_history_distinguishes_execution_error_from_visual_rejection(tmp_path, monkeypatch):
    db = RunTrackerDB(str(tmp_path / "runs.db"))
    try:
        ctx = RunContext(prompt="a cat")
        ctx.verification_report = {"accepted": False, "error": "provider timeout"}
        db.log_run(ctx)
        monkeypatch.setattr(api.pipeline, "db", db)
        response = TestClient(api.app).get("/api/runs")
        assert response.status_code == 200
        assert response.json()["runs"][0]["status"] == "error"
        assert response.json()["runs"][0]["accepted"] is False
    finally:
        db.close()


def test_stats_names_the_actual_hard_check_metric(monkeypatch):
    monkeypatch.setattr(api.pipeline.db, "all_benchmark_rows", lambda **kwargs: [
        ("baseline", "a red cat", True, 0, json.dumps({"attempts": 1})),
    ])
    data = TestClient(api.app).get("/stats").json()
    assert data["metric_scope"] == "count_and_spatial_hard_checks"
    assert data["configs"]["baseline"]["hard_pass_rate"] == 1.0
    assert "csr" not in data["configs"]["baseline"]
