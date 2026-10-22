from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from src.core.context import RunContext
from src.generation.controlled import validate_control_profile
from src.generation.engine import GenerationEngine
from src.scene.planner import ScenePlanner
from src.verification.verify_stage import VerifyStage


def test_invalid_control_sampler_rejected_before_model_loading():
    with pytest.raises(ValueError, match="SDXL base"):
        validate_control_profile("stabilityai/sdxl-turbo", {"steps": 4, "cfg_scale": 0})
    with pytest.raises(ValueError, match="sampler"):
        validate_control_profile("stabilityai/stable-diffusion-xl-base-1.0", {"steps": 4})


@pytest.mark.parametrize("settings", [{"steps": 25.5}, {"bootstrap_steps": True},
    {"seed": -1}, {"seed": 2**63}, {"seed": True}, {"cfg_scale": "7"}, {"cfg_scale": float("nan")}])
def test_invalid_control_values_fail_before_expensive_gpu_loading(settings):
    with pytest.raises(ValueError):
        validate_control_profile("stabilityai/stable-diffusion-xl-base-1.0", settings)


def test_scene_plan_and_engine_dispatch_share_the_same_immutable_contract(monkeypatch, tmp_path):
    contract = {"prompt": "a red car left of a blue bus", "entities": [
        {"id": "car", "class_name": "car", "attributes": ["red"]},
        {"id": "bus", "class_name": "bus", "attributes": ["blue"]}]}
    _, hints, config, prompt, masks = ScenePlanner().plan(contract, gen_defaults={
        "region_controller": "fusion", "steps": 25, "cfg_scale": 7, "seed": 43})
    assert hints["instance_regions"]["enabled"]
    assert all("bounding_box" not in e for e in contract["entities"])
    calls = []
    def generate(model, text, settings, plan, path, hook):
        calls.append((model, text, plan))
        return {"generation_backend": "synthetic-dispatch-fixture"}
    monkeypatch.setattr("src.generation.controlled.generate_controlled", generate)
    engine = GenerationEngine({"models": {"generator": "stabilityai/stable-diffusion-xl-base-1.0"}})
    monkeypatch.setattr(engine, "ensure_loaded", lambda **kwargs: pytest.fail("Dense pipeline must not load for NF4 fusion"))
    result = engine.generate(prompt, config, hints, masks, tmp_path / "generated_0.png")
    assert calls[0][2] == hints["instance_regions"]
    assert result["generation_backend"] == "synthetic-dispatch-fixture"


def verifier_context(tmp_path, monkeypatch, local):
    monkeypatch.setattr("src.utils.release_clip", lambda: None)
    context = RunContext(prompt="one red car", contract={"prompt": "one red car",
        "entities": [{"class_name": "car", "attributes": ["red"]}]},
        config={"paths": {"output_dir": str(tmp_path)}})
    Image.new("RGB", (32, 32), "blue").save(context.run_dir / "generated_0.png")
    stage = VerifyStage.__new__(VerifyStage)
    stage.config = {}
    stage.qa = SimpleNamespace(verify=lambda *args: {"accepted": True, "hard_failures": [],
        "signals": {"expected_counts": {"car": 1}, "detected_counts": {"car": 1}},
        "detected_boxes": {"car": [{}]}})
    stage.local_instances = local
    stage.clip = SimpleNamespace(score=lambda *args: {"clip_norm": .8, "clip_cos": .35})
    stage.vqa = None
    return stage, context


def test_local_color_contradiction_blocks_acceptance_without_an_external_judge(tmp_path, monkeypatch):
    local = SimpleNamespace(verify=lambda *args: {"refined_boxes": {"car": [{}]},
        "mask_count_status": "observed_diagnostic", "classes": {"car": {
            "status": "contradicted_diagnostic", "expected_colors": ["red"], "observed_colors": ["blue"]}}})
    stage, context = verifier_context(tmp_path, monkeypatch, local)
    stage.run(context)
    assert context.verification_report["hard_failures"][0]["type"] == "attribute_mismatch"
    assert context.verification_report["vqa_status"] == "disabled"
    attributes = [r for r in context.verification_report["contract_evidence"]["requirements"] if r["kind"] == "attribute"]
    assert attributes[0]["status"] == "failed"


def test_missing_local_weights_cannot_silently_pass(tmp_path, monkeypatch):
    def missing(*args): raise FileNotFoundError("Synthetic missing segmentation fixture")
    stage, context = verifier_context(tmp_path, monkeypatch, SimpleNamespace(verify=missing))
    stage.run(context)
    assert context.verification_report["hard_failures"][0]["type"] == "verification_unavailable"
    assert context.verification_report["signals"]["local_instances"]["status"] == "unavailable"


def test_color_repair_is_deterministic_without_a_provider(monkeypatch):
    from src.reflection import reflector
    monkeypatch.setattr(reflector, "LLMRouter", lambda: pytest.fail("Known local color failures need no LLM"))
    edits = reflector.build_corrective_edits({"hard_failures": [{"type": "attribute_mismatch",
        "class_name": "car", "expected": ["red"], "detected": ["blue"]}]}, {})
    assert edits == ["red colored car, correct color on every instance"]


def test_unknown_required_chromatic_color_blocks_the_explicit_profile(tmp_path, monkeypatch):
    local = SimpleNamespace(verify=lambda *args: {"refined_boxes": {"car": [{}]},
        "mask_count_status": "observed_diagnostic", "classes": {"car": {
            "status": "unknown", "expected_colors": ["red"], "reason": "insufficient chromatic pixels"}}})
    stage, context = verifier_context(tmp_path, monkeypatch, local)
    stage.config = {"verification": {"require_chromatic_evidence": True}}
    stage.run(context)
    assert context.verification_report["hard_failures"][0]["type"] == "attribute_unverified"


def test_regional_repair_uses_recorded_different_seeds():
    contract = {"prompt": "a cat and a dog", "entities": [{"class_name": "cat"}, {"class_name": "dog"}]}
    defaults = {"region_controller": "fusion", "seed": 42, "repair_seed_stride": 1}
    planner = ScenePlanner()
    first = planner.plan(contract, repair_attempt=0, gen_defaults=defaults)[2]
    second = planner.plan(contract, repair_attempt=1, gen_defaults=defaults)[2]
    assert first["seed"] == 42 and second["seed"] == 43


def test_explicit_profile_can_repair_without_loading_optional_failure_memory(monkeypatch, tmp_path):
    from src.core.stage import Pipeline
    monkeypatch.setattr("src.retrieval.memory.FailureMemoryDB",
        lambda: pytest.fail("Disabled failure memory must not load Milvus or an embedding model"))
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = {"paths": {"output_dir": str(tmp_path)},
        "retrieval": {"failure_memory_enabled": False}, "pipeline": {"max_repair_loops": 1}}
    pipeline.cache_key = "synthetic-test"
    pipeline.db = None
    def verify(ctx):
        ctx.verification_report = {"hard_failures": [{"type": "count_mismatch"}] if ctx.attempt == 0 else []}
    def reflect(ctx):
        ctx.decision = {"accept": ctx.attempt == 1, "prompt_edits": ["individually visible objects"]}
    pipeline.stages = {"generate": SimpleNamespace(run=lambda ctx: None),
        "verify": SimpleNamespace(run=verify), "reflect": SimpleNamespace(run=reflect)}
    assert pipeline.run("synthetic offline repair").is_successful


def test_provided_contract_bypasses_intent_and_cannot_enter_normal_prompt_cache(tmp_path):
    from src.core.stage import Pipeline
    def forbidden(context): pytest.fail("Provided source contract must not be re-extracted")
    def verify(context): context.verification_report = {"hard_failures": []}
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = {"paths": {"output_dir": str(tmp_path)}, "pipeline": {"max_repair_loops": 0}}
    pipeline.stages = {"intent": SimpleNamespace(run=forbidden),
                       "generate": SimpleNamespace(run=lambda c: None), "verify": SimpleNamespace(run=verify)}
    pipeline.db = None; pipeline.cache_key = "fixture-only-normal-cache"
    given = {"prompt": "a cat", "entities": [{"id": "cat", "class_name": "cat"}]}
    context = pipeline.run("a cat", provided_contract=given)
    assert context.is_successful and context.contract["entities"][0]["id"] == "cat"
    assert "created_at" not in given
    assert context.cache_key != pipeline.cache_key
    with pytest.raises(ValueError, match="differs"):
        pipeline.run("a dog", provided_contract=given)
    with pytest.raises(ValueError, match="benchmark manifest"):
        pipeline.run("a cat", provided_contract=given, is_benchmark=True)
