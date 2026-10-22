"""Fault injection at real orchestration boundaries, without fitting or services."""
import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.research import experiment
from src.research.config import load_config
from src.research.reports import list_experiments


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("Lifecycle tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    torch.set_num_threads(2)


@pytest.mark.parametrize("fault,status,mlflow", [
    (MemoryError, "failed", "FAILED"),
    (TimeoutError, "failed", "FAILED"),
    (KeyboardInterrupt, "interrupted", "KILLED"),
])
def test_training_guard_failure_is_never_a_finished_model(tmp_path, monkeypatch, fault, status, mlflow):
    transitions = []

    class Tracker:
        run_id = "synthetic-local-run"

        def __init__(self, *args):
            pass

        def params(self, values):
            pass

        def finish(self, value):
            transitions.append(value)

    class FailingBudget:
        def check(self):
            raise fault("injected resource/lifecycle fault")

    monkeypatch.setattr(experiment, "ExperimentTracker", Tracker)
    monkeypatch.setattr(experiment, "pixels_for", lambda *args: np.zeros((128, 128, 3), dtype=np.uint8))
    config = load_config().model_copy(update={"steps": 1})
    with pytest.raises(fault):
        experiment.train_arm(config, [{"split": "train", "source_status": "valid"}],
                             tmp_path, "candidate", .1, FailingBudget())
    progress = json.loads((tmp_path / "candidate/progress.json").read_text())
    failure = json.loads((tmp_path / "candidate/failure.json").read_text())
    assert transitions == [mlflow]
    assert progress["status"] == failure["status"] == status
    assert failure["error_type"] == fault.__name__ and failure["last_logged_step"] == 0
    assert failure["resources"]["device"] == "cpu"
    assert progress["quality_status"] == "unverified"
    assert not (tmp_path / "candidate/training.json").exists()
    assert not (tmp_path / "candidate/checkpoint.safetensors").exists()


@pytest.mark.parametrize("fault,status", [
    (MemoryError, "failed"), (TimeoutError, "failed"), (KeyboardInterrupt, "interrupted"),
])
def test_experiment_fault_preserves_journal_and_read_api_state(tmp_path, monkeypatch, fault, status):
    manifest = tmp_path / "data.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "runs"
    monkeypatch.setattr(experiment, "workspace_path", lambda value: output)
    monkeypatch.setattr(experiment, "prepare_data", lambda config: manifest)
    monkeypatch.setattr(experiment, "load_data", lambda *args: ({"records": []}, None))
    monkeypatch.setattr(experiment, "source_identity", lambda: {"synthetic_fixture": "lifecycle-only"})
    monkeypatch.setattr(experiment, "runtime_identity", lambda: {"scope": "fixture_only"})
    monkeypatch.setattr(experiment, "snapshot_source", lambda *args: None)

    def fail(*args):
        raise fault("injected orchestration fault")

    monkeypatch.setattr(experiment, "train_classical", fail)
    with pytest.raises(fault):
        experiment.run_experiment(reuse_completed=False)
    folders = [path for path in output.iterdir() if path.is_dir()]
    assert len(folders) == 1
    journal = json.loads((folders[0] / "progress.json").read_text())
    assert journal["execution_status"] == status and journal["error_type"] == fault.__name__
    assert journal["resources"]["device"] == "cpu"
    assert not (folders[0] / "report.json").exists()
    assert list_experiments(output)["experiments"][0]["execution_status"] == status
    # Exception handling must release the real process exclusion lock.
    with experiment.experiment_lock(output):
        pass


@pytest.mark.parametrize("rss,available", [(8_000_000_001, 2_000_000_000), (100_000_000, 749_999_999)])
def test_actual_memory_guard_records_limits_and_observations(monkeypatch, rss, available):
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss), children=lambda: [])
    monkeypatch.setattr(experiment.psutil, "Process", lambda: process)
    monkeypatch.setattr(experiment.psutil, "virtual_memory", lambda: SimpleNamespace(used=12_000_000_000, available=available))
    budget = experiment.Budget(60)
    with pytest.raises(MemoryError):
        budget.check()
    result = budget.summary()
    assert result["observed_peak_process_tree_rss_bytes"] == rss
    assert result["observed_minimum_available_bytes"] == available
    assert result["limits"]["minimum_available_bytes"] == 750_000_000


def test_source_snapshot_rejects_changes_before_any_fit(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "implementation.py"
    source.write_text("value = 1\n", encoding="utf-8")
    sources = {"implementation.py": experiment.sha256(source)}
    monkeypatch.setattr(experiment, "ROOT", checkout)
    experiment.snapshot_source(tmp_path / "run", sources)
    assert (tmp_path / "run/source/implementation.py").read_bytes() == source.read_bytes()
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed before fitting"):
        experiment.snapshot_source(tmp_path / "another-run", sources)
    assert not (tmp_path / "another-run/source/implementation.py").exists()


def test_runtime_identity_includes_ml_decoder_tracking_and_python_versions():
    try:
        identity = experiment.runtime_identity()
    except PackageNotFoundError:
        pytest.skip("Install requirements-research.txt for research runtime-version checks")
    assert identity["python"] and identity["system"]
    assert all(identity["packages"][name] for name in (
        "torch", "numpy", "scipy", "Pillow", "scikit-learn", "safetensors", "mlflow-skinny", "SQLAlchemy"))
