"""Read-only report/API behavior, including hostile and partial evidence."""
import uuid
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.research.artifacts import atomic_json, sha256
from src.research.api import router
from src.research import reports


def fixture_report(root):
    folder = root / str(uuid.uuid4())
    folder.mkdir(parents=True)
    arm = {"n": 32, "counts": {"valid": 0, "invalid": 31, "unknown": 1, "error": 0}, "valid_rate": 0}
    document = {"schema": "framegenx-research-report-v1", "id": folder.name,
        "execution_status": "completed", "quality_status": "unverified", "study_scope": "cards_cpu_feasibility_only",
        "dataset": {"records": 4096}, "evaluation": {"scope": "color_aware_cards_template_validity_only",
            "full_contract_CSR": None, "reduction_claim": False, "arms": {"baseline": arm, "candidate": arm}},
        "deep_learning": {"baseline": {"steps": 500, "rho": 0}, "candidate": {"steps": 500, "rho": .1}},
        "classical_ml": {"metrics": {"test": {"n": 100, "balanced_accuracy": .9, "roc_auc": .95}}},
        "sdxl_integration": {"automatic_loading": False},
        "raw_exception": "SYNTHETIC_SECRET_MUST_NOT_LEAVE_LOCAL_STORAGE",
        "api_key": "SYNTHETIC_SECRET_MUST_NOT_LEAVE_LOCAL_STORAGE"}
    atomic_json(folder / "report.json", document)
    refresh_manifest(folder)
    return folder, document


def refresh_manifest(folder):
    atomic_json(folder / "manifest.json", {"schema": "framegenx-research-manifest-v1",
                                           "report_sha256": sha256(folder / "report.json")})


def client_for(root, monkeypatch):
    monkeypatch.setattr(reports, "RESEARCH_ROOT", root)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_real_api_shape_is_bounded_and_allowlisted(tmp_path, monkeypatch):
    fixture_report(tmp_path)
    client = client_for(tmp_path, monkeypatch)
    result = client.get("/api/research/experiments?limit=1")
    assert result.status_code == 200
    body = result.json()
    assert body["experiments"][0]["quality_status"] == "unverified"
    assert body["experiments"][0]["full_contract_CSR"] is None
    assert "SYNTHETIC_SECRET" not in result.text
    for limit in [0, -1, 26, 100000]:
        assert client.get(f"/api/research/experiments?limit={limit}").status_code == 422
    assert client.post("/api/research/experiments", json={}).status_code == 405


def test_missing_evidence_is_empty_not_a_fake_completed_run(tmp_path):
    assert reports.list_experiments(tmp_path / "absent")["experiments"] == []


def test_report_publication_waits_for_manifest_without_false_corruption(tmp_path):
    folder, _ = fixture_report(tmp_path)
    (folder / "manifest.json").unlink()
    atomic_json(folder / "progress.json", {"id": folder.name, "execution_status": "running",
                                           "quality_status": "unverified"})
    view = reports.list_experiments(tmp_path)["experiments"][0]
    assert view["execution_status"] == "running" and "classical_ml" not in view
    refresh_manifest(folder)
    assert reports.list_experiments(tmp_path)["experiments"][0]["execution_status"] == "completed"
    (folder / "manifest.json").unlink()
    atomic_json(folder / "progress.json", {"id": folder.name, "execution_status": "completed",
                                           "quality_status": "unverified"})
    assert reports.list_experiments(tmp_path)["experiments"][0]["execution_status"] == "evidence_error"


def test_failed_and_interrupted_runs_keep_lifecycle_without_metrics(tmp_path):
    for status in ("failed", "interrupted", "running"):
        folder = tmp_path / str(uuid.uuid4())
        folder.mkdir()
        atomic_json(folder / "progress.json", {"id": folder.name, "execution_status": status,
                                               "quality_status": "unverified"})
    result = reports.list_experiments(tmp_path)
    assert {run["execution_status"] for run in result["experiments"]} == {"failed", "interrupted", "running"}
    assert all("classical_ml" not in run for run in result["experiments"])


@pytest.mark.parametrize("change", [{"quality_status": "qualified"}, {"evaluation": {}},
                                  {"id": "different"}, {"schema": "unknown"}])
def test_semantically_wrong_report_stays_corrupt_even_with_a_matching_checksum(tmp_path, change):
    folder, document = fixture_report(tmp_path)
    document.update(change)
    atomic_json(folder / "report.json", document)
    refresh_manifest(folder)
    assert reports.list_experiments(tmp_path)["experiments"][0]["execution_status"] == "evidence_error"


def test_tampered_nonfinite_or_oversized_report_never_becomes_a_success(tmp_path):
    folder, _ = fixture_report(tmp_path)
    (folder / "report.json").write_text('{"metric":NaN}', encoding="utf-8")
    assert reports.list_experiments(tmp_path)["experiments"][0]["execution_status"] == "evidence_error"
    (folder / "report.json").write_text('x'*1_000_001, encoding="utf-8")
    refresh_manifest(folder)
    assert reports.list_experiments(tmp_path)["experiments"][0]["execution_status"] == "evidence_error"


def test_research_paths_cannot_be_selected_by_client(tmp_path, monkeypatch):
    client = client_for(tmp_path, monkeypatch)
    response = client.get("/api/research/experiments?root=../../&file=.env")
    assert response.status_code == 200 and response.json()["experiments"] == []


def test_unknown_folders_are_ignored_and_corrupt_known_run_is_visible(tmp_path):
    (tmp_path / "private").mkdir()
    folder = tmp_path / str(uuid.uuid4())
    folder.mkdir()
    (folder / "progress.json").write_text("incomplete", encoding="utf-8")
    result = reports.list_experiments(tmp_path)
    assert len(result["experiments"]) == 1
    assert result["experiments"][0]["execution_status"] == "evidence_error"
