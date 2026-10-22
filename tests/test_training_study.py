"""Study recovery tests use explicit synthetic artifacts, never real model work."""
from pathlib import Path
import uuid

from PIL import Image
import pytest

from src.training import study
from src.training.common import TrainingConfig, atomic_json, file_sha256
from src.training.train import save_source_snapshot


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    import json
    dataset = tmp_path / "records.jsonl"
    rows = []
    for i, split in enumerate(["train", "train", "validation", "test"]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(path)
        rows.append({"id": str(i), "source": "fixture", "caption": "one object", "image": path.name,
            "image_sha256": file_sha256(path), "positive": True, "split": split,
            "license": {"images": "fixture-only"}, "provenance": {"scope": "unit-test"},
            "constraints": [{"type": "object_count"}]})
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    backbone = tmp_path / "synthetic-backbone"
    (backbone / "unet").mkdir(parents=True)
    (backbone / "unet/diffusion_pytorch_model.fp16.safetensors").write_bytes(b"fixture-only-not-a-model")
    monkeypatch.setattr(study, "training_model_source", lambda model: str(backbone))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic unit-test protocol")
    config = TrainingConfig(dataset=str(dataset), output_root=str(tmp_path / "training"), max_steps=1)
    output = tmp_path / "study"
    calls = []
    def fake_run(config):
        calls.append(1)
        frozen = study.freeze_study(config, protocol)
        root = Path(config.output_root) / str(uuid.uuid4())
        checkpoint = root / "checkpoint-000001"
        checkpoint.mkdir(parents=True)
        adapter, state = checkpoint / "adapter.safetensors", checkpoint / "training_state.pt"
        adapter.write_bytes(b"synthetic-adapter")
        state.write_bytes(b"synthetic-state")
        signature = {key: frozen[key] for key in ("training_code_sha256", "dataset_sha256", "model_sha256")}
        manifest = {"schema": "framegenx-adapter-v1", "signature": signature, "step": 1,
            "quality_status": "unverified", "promotion_status": "unqualified",
            "adapter_sha256": file_sha256(adapter), "state_sha256": file_sha256(state)}
        atomic_json(checkpoint / "adapter.manifest.json", manifest)
        save_source_snapshot(root, dataset, frozen["training_code_sha256"])
        result = {"run_id": root.name, "status": "trained_pending_evaluation",
            "quality_status": "unverified", "promotion_status": "unqualified", "steps_completed": 1,
            "train_images": 2, "distinct_train_images_seen": 2, "latest_checkpoint": str(checkpoint),
            "config": config.model_dump(), "packages": frozen["packages"],
            "model_snapshot": frozen["model_snapshot"], "signature": signature}
        atomic_json(root / "result.json", result)
        return result
    monkeypatch.setattr(study, "run_training", fake_run)
    return config, protocol, output, calls


def test_completed_study_reuses_intact_artifacts_without_fitting_twice(inputs):
    config, protocol, output, calls = inputs
    first, reused = study.run_study(config, protocol, output)
    assert not reused
    second, reused = study.run_study(config, protocol, output)
    assert reused and first == second and len(calls) == 1


def test_changed_training_state_is_not_hidden_by_a_completed_result(inputs):
    config, protocol, output, calls = inputs
    result, _ = study.run_study(config, protocol, output)
    (Path(result["latest_checkpoint"]) / "training_state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="training state changed"):
        study.run_study(config, protocol, output)
    assert len(calls) == 1


def test_changed_protocol_does_not_implicitly_refit(inputs):
    config, protocol, output, calls = inputs
    study.run_study(config, protocol, output)
    protocol.write_text("Changed protocol")
    with pytest.raises(ValueError, match="inputs changed"):
        study.run_study(config, protocol, output)
    assert len(calls) == 1


def test_interrupted_study_keeps_failed_journal_and_requires_explicit_recovery(inputs, monkeypatch):
    config, protocol, output, calls = inputs
    def interrupted(config):
        calls.append(1)
        raise KeyboardInterrupt()
    monkeypatch.setattr(study, "run_training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.run_study(config, protocol, output)
    assert '"interrupted"' in (output / "execution.json").read_text()
    with pytest.raises(ValueError, match="previous study attempt"):
        study.run_study(config, protocol, output)
    assert len(calls) == 1
