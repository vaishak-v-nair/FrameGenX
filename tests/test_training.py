"""Dataset leakage, objective semantics and experimental adapter safety gates."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from src.training.cache import checked_tokens, full_image_tensor, load_cached_item
from src.training.common import TrainingConfig, atomic_json, file_sha256
from src.training.data import assign_splits, validate_records
from src.training.objective import denoising_loss, make_target
from src.training.registry import adapter_manifest
from src.training.train import ImageSampler, choose_validation
from src.generation.lora import LoraManager


@pytest.fixture
def dataset(tmp_path):
    records = []
    for source_index, source in enumerate(("clevr", "vsr", "coco")):
        for index in range(10):
            image = tmp_path / f"{source}_{index}.png"
            Image.new("RGB", (48, 32), (index * 20, source_index * 70, 50)).save(image)
            records.append({"id": f"{source}:{index}", "source": source, "image": image.name,
                "image_sha256": file_sha256(image), "caption": f"{index + 1} objects.", "positive": True,
                "license": {"images": "fixture-only"}, "provenance": {"scope": "synthetic_unit_fixture"},
                "constraints": [{"type": "object_count", "count": index + 1}]})
    records = assign_splits(records)
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    return path, records


def write_records(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_grouped_splits_are_reproducible_and_audited(dataset):
    path, records = dataset
    assert assign_splits(records) == records
    validated, audit = validate_records(path)
    assert len(validated) == 30
    assert audit["image_groups"] == 30
    assert audit["positive_by_source_and_split"]["vsr"] == {"train": 8, "validation": 1, "test": 1}


def test_same_image_cannot_leak_to_heldout_split(dataset):
    path, records = dataset
    copied = {**records[0], "id": "leaked", "caption": "Another description.",
              "split": "test" if records[0]["split"] != "test" else "train"}
    write_records(path, [*records, copied])
    with pytest.raises(ValueError, match="leakage"):
        validate_records(path)


def test_tampered_image_fails_validation(dataset):
    path, records = dataset
    Image.new("RGB", (48, 32), "red").save(path.parent / records[0]["image"])
    with pytest.raises(ValueError, match="hash changed"):
        validate_records(path)


def test_reencoded_identical_pixels_cannot_leak_to_a_different_split(dataset):
    from PIL.PngImagePlugin import PngInfo
    path, records = dataset
    info = PngInfo()
    info.add_text("source", "different-container-metadata")
    copied_image = path.parent / "reencoded.png"
    with Image.open(path.parent / records[0]["image"]) as image:
        image.save(copied_image, pnginfo=info)
    assert file_sha256(copied_image) != records[0]["image_sha256"]
    duplicate = {**records[0], "id": "reencoded", "image": copied_image.name,
                 "image_sha256": file_sha256(copied_image), "caption": "Another caption",
                 "split": "test" if records[0]["split"] != "test" else "train"}
    write_records(path, [*records, duplicate])
    with pytest.raises(ValueError, match="Pixel-identical image leakage"):
        validate_records(path)


@pytest.mark.parametrize("field,value", [("license", {}), ("provenance", {}), ("constraints", []),
                                          ("image_sha256", "not-a-hash"), ("positive", 1)])
def test_invalid_provenance_and_labels_are_rejected(dataset, field, value):
    path, records = dataset
    records[0][field] = value
    write_records(path, records)
    with pytest.raises(ValueError):
        validate_records(path)


def test_image_outside_dataset_is_rejected(dataset):
    path, records = dataset
    records[0]["image"] = "../outside.png"
    write_records(path, records)
    with pytest.raises(ValueError, match="outside"):
        validate_records(path)


def test_duplicate_caption_and_missing_split_are_rejected(dataset):
    path, records = dataset
    write_records(path, [*records, {**records[0], "id": "duplicate"}])
    with pytest.raises(ValueError, match="Duplicate image/caption"):
        validate_records(path)
    write_records(path, [r for r in records if r["source"] != "vsr" or r["split"] != "test"])
    with pytest.raises(ValueError, match="lacks"):
        validate_records(path)


def test_negative_caption_never_counts_as_training_supervision(dataset):
    path, records = dataset
    negative = {**records[0], "id": "negative", "positive": False, "caption": "False spatial statement."}
    write_records(path, [*records, negative])
    validated, audit = validate_records(path)
    assert audit["negative_examples"] == 1
    assert sum(r["positive"] for r in validated) == 30


def test_full_image_letterbox_keeps_edge_objects(tmp_path):
    image = Image.new("RGB", (200, 100), "white")
    image.paste("red", (0, 0, 20, 100))
    image.paste("blue", (180, 0, 200, 100))
    path = tmp_path / "edges.png"
    image.save(path)
    tensor, original = full_image_tensor(path, 256)
    assert original == (100, 200)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor[0, 0, 128, 0] > .9  # left red edge survives
    assert tensor[0, 2, 128, -1] > .9  # right blue edge survives


def test_long_caption_is_not_silently_truncated():
    tokenizer = SimpleNamespace(model_max_length=77)
    class Tokenizer:
        model_max_length = 77
        def __call__(self, caption, **kwargs):
            return {"input_ids": list(range(80))}
    with pytest.raises(ValueError, match="truncation"):
        checked_tokens(Tokenizer(), "long prompt", "test")


def test_changed_cached_tensor_is_rejected(tmp_path):
    path = tmp_path / "cache.safetensors"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        load_cached_item(tmp_path, {"file": path.name, "sha256": "0" * 64})


@pytest.mark.parametrize("prediction_type, expected", [("epsilon", .5), ("v_prediction", .25)])
def test_min_snr_weights_follow_prediction_type(prediction_type, expected):
    scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type=prediction_type),
                                alphas_cumprod=torch.tensor([.5]))
    loss, raw = denoising_loss(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2),
                               torch.tensor([0]), scheduler, .5)
    assert raw.item() == 1
    assert loss.item() == expected


def test_unknown_diffusion_target_is_rejected():
    scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type="guess"))
    with pytest.raises(ValueError, match="Unsupported"):
        make_target(scheduler, None, None, None)


def test_sampler_resume_matches_uninterrupted_sample_order():
    sampler = ImageSampler(7, 42)
    first_epoch = [sampler.next() for _ in range(7)]
    assert sorted(first_epoch) == list(range(7))
    state = sampler.state_dict()
    resumed = ImageSampler(7, 0)
    resumed.load_state_dict(state)
    assert [sampler.next() for _ in range(20)] == [resumed.next() for _ in range(20)]


def test_validation_includes_every_source():
    items = [{"id": str(i), "source": source} for source in ("clevr", "coco", "vsr") for i in range(3)]
    assert {i["source"] for i in choose_validation(items, 3)} == {"clevr", "coco", "vsr"}


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(b"unit-fixture-only-not-real-weights")
    manifest = {"schema": "framegenx-adapter-v1", "base_model": "sdxl-base", "run_id": "fixture",
                "adapter_sha256": file_sha256(path), "promotion_status": "unqualified"}
    atomic_json(path.with_suffix(".manifest.json"), manifest)
    return path, manifest


def test_unqualified_candidate_requires_explicit_opt_in(adapter):
    path, _ = adapter
    with pytest.raises(ValueError, match="unqualified"):
        adapter_manifest(str(path), "sdxl-base")
    evidence = adapter_manifest(str(path), "sdxl-base", allow_unqualified=True)
    assert evidence["quality_status"] == "unverified"
    assert evidence["experimental"] is True


def test_different_base_and_tampered_adapter_are_rejected(adapter):
    path, _ = adapter
    with pytest.raises(ValueError, match="differs"):
        adapter_manifest(str(path), "sdxl-turbo", allow_unqualified=True)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        adapter_manifest(str(path), "sdxl-base", allow_unqualified=True)


def test_fake_qualification_cannot_bypass_evaluation(adapter):
    path, manifest = adapter
    manifest["promotion_status"] = "qualified"
    atomic_json(path.with_suffix(".manifest.json"), manifest)
    with pytest.raises(ValueError, match="evaluation evidence"):
        adapter_manifest(str(path), "sdxl-base")


def test_lora_hook_never_loads_unqualified_adapter_by_default(adapter):
    path, _ = adapter
    class Pipe:
        def load_lora_weights(self, *args, **kwargs):
            pytest.fail("Unqualified weights must be blocked before loading")
    manager = LoraManager({"models": {"generator": "sdxl-base"}, "lora": {
        "adapters": [{"name": "candidate", "source": str(path), "kind": "trained"}]}})
    manager.hook(Pipe())
    assert manager.load_report[0]["loaded"] is False
    assert "unqualified" in manager.load_report[0]["error"]


def test_training_profile_rejects_unbounded_configuration():
    with pytest.raises(ValueError):
        TrainingConfig(max_steps=0)
    with pytest.raises(ValueError):
        TrainingConfig(snr_gamma=float("inf"))
    with pytest.raises(ValueError):
        TrainingConfig(target_modules=[])


def test_failed_training_never_reports_finished(dataset, tmp_path, monkeypatch):
    from src.training import train as training
    statuses = []
    class Tracker:
        run_id = "synthetic-tracker-fixture"
        def __init__(self, *args):
            pass
        def params(self, values):
            pass
        def finish(self, status, quality_status="unverified"):
            statuses.append(status)
        def artifact(self, path):
            pass
    def failed_cache(*args):
        raise RuntimeError("synthetic frozen-model failure")
    monkeypatch.setattr(training, "ExperimentTracker", Tracker)
    monkeypatch.setattr(training, "prepare_cache", failed_cache)
    config = TrainingConfig(dataset=str(dataset[0]), output_root=str(tmp_path / "runs"))
    result = training.run_training(config)
    assert result["status"] == "failed"
    assert result["quality_status"] == "unverified"
    assert result["steps_completed"] == 0
    assert statuses == ["FAILED"]
    assert json.loads((tmp_path / "runs" / result["run_id"] / "result.json").read_text())["status"] == "failed"


def test_local_mlflow_refuses_nonfinite_metrics_and_records_failure(tmp_path):
    pytest.importorskip("mlflow", reason="Local tracking integration requires requirements-ml.txt")
    from src.training.tracking import ExperimentTracker
    tracker = ExperimentTracker(str(tmp_path / "tracking.db"), "fixture-only", "synthetic-failure-test")
    with pytest.raises(ValueError, match="Nonfinite"):
        tracker.metrics({"invalid_probe": float("nan")})
    tracker.finish("FAILED")
    saved = tracker.client.get_run(tracker.run_id)
    assert saved.info.status == "FAILED"
    assert saved.data.tags["quality_status"] == "unverified"


def test_diffusers_export_roundtrip_retains_tensors_and_alpha(tmp_path):
    from diffusers import UNet2DConditionModel, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from src.training.export import complete_lora_metadata
    def tiny_unet():
        return UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32,), down_block_types=("CrossAttnDownBlock2D",),
            up_block_types=("CrossAttnUpBlock2D",), cross_attention_dim=32, attention_head_dim=8,
            norm_num_groups=8)
    model = tiny_unet()
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    model.add_adapter(config)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(.01)
    expected = get_peft_model_state_dict(model)
    StableDiffusionXLPipeline.save_lora_weights(str(tmp_path),
        unet_lora_layers=convert_state_dict_to_diffusers(expected),
        unet_lora_adapter_metadata=complete_lora_metadata(config), weight_name="adapter.safetensors")
    loaded = tiny_unet()
    loaded.load_lora_adapter(str(tmp_path), prefix="unet", weight_name="adapter.safetensors", adapter_name="candidate")
    actual = get_peft_model_state_dict(loaded, adapter_name="candidate")
    assert actual.keys() == expected.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert loaded.peft_config["candidate"].lora_alpha == 4
    assert loaded.peft_config["candidate"].r == 2


def test_six_gb_profile_blocks_unvalidated_plain_lora_before_weight_loading(monkeypatch):
    from src.training.model import load_trainable_unet
    from diffusers import UNet2DConditionModel
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (6_000_000_000, 6_442_450_944))
    def forbidden_load(*args, **kwargs):
        pytest.fail("Unvalidated full SDXL weights must not be allocated on the 6 GB profile")
    monkeypatch.setattr(UNet2DConditionModel, "from_pretrained", forbidden_load)
    from src.core.device import device_lease
    with device_lease(), pytest.raises(RuntimeError, match="requires QLoRA"):
        load_trainable_unet(TrainingConfig(method="lora"))


@pytest.mark.parametrize("bounds", [[float("nan"), .2], [.2, .1], [float("inf"), 2], [.1]])
def test_invalid_confidence_intervals_cannot_qualify_a_model(adapter, bounds):
    path, manifest = adapter
    report = {"metric": "full_contract_CSR", "unverified_requirements": 0,
              "independent_image_groups": 50, "seeds": 3, "paired_CSR_delta_95_ci": bounds,
              "adapter_sha256": manifest["adapter_sha256"], "base_model": manifest["base_model"]}
    report_file = path.parent / "qualification.json"
    # json.loads permits NaN, so exercise the gate with a real malformed report.
    report_file.write_text(json.dumps(report))
    manifest.update(promotion_status="qualified", qualification={"report": report_file.name,
                                                               "report_sha256": file_sha256(report_file)})
    atomic_json(path.with_suffix(".manifest.json"), manifest)
    with pytest.raises(ValueError, match="confidence interval"):
        adapter_manifest(str(path), "sdxl-base")


def test_training_snapshot_keeps_exact_source_and_dataset_provenance(dataset, tmp_path):
    from src.training.train import code_fingerprint, save_source_snapshot
    root = tmp_path / "run"
    save_source_snapshot(root, dataset[0], code_fingerprint())
    assert file_sha256(root / "dataset_records.jsonl") == file_sha256(dataset[0])
    assert not (root / "code_snapshot/.env").exists()
    assert (root / "code_snapshot/train.py").read_bytes() == Path("src/training/train.py").read_bytes()


def test_local_experimental_adapter_uses_explicit_offline_filename(adapter):
    path, _ = adapter
    calls = []
    class Pipe:
        def load_lora_weights(self, source, adapter_name, **kwargs):
            calls.append((source, adapter_name, kwargs))
        def set_adapters(self, names, adapter_weights):
            assert names == ["candidate"]
    manager = LoraManager({"models": {"generator": "sdxl-base"}, "lora": {"adapters": [
        {"name": "candidate", "source": str(path), "kind": "trained", "allow_unqualified": True}]}})
    manager.hook(Pipe())
    assert calls == [(str(path.parent), "candidate", {"weight_name": path.name, "local_files_only": True})]
    assert manager.load_report[0]["activated"] is True
    assert manager.load_report[0]["training_provenance"]["quality_status"] == "unverified"
