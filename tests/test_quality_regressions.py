"""Acceptance and provenance regressions that do not require model downloads."""
import json
from types import SimpleNamespace

import pytest

from src.core.context import RunContext
from src.core.db import RunTrackerDB
from src.core import stage as stage_module
from src.generation.lora import AdaptStage
from src.generation.engine import GenerateStage, GenerationEngine
from src.generation.model_source import resolve_model_source
from src.reporting.analyzer import Reporter
from src.scene.planner import ScenePlanner
from src.verification.qa import QAVerifier
from src.verification.spatial_verifier import SpatialRelationVerifier


def test_pipeline_cannot_accept_without_verifier(monkeypatch):
    monkeypatch.setattr(stage_module, "build_stages", lambda config: {})
    pipeline = stage_module.Pipeline({"pipeline": {"stages": [], "max_repair_loops": 0}})
    with pytest.raises(ValueError, match="verify"):
        pipeline.run("a red cat")


def test_previous_attempt_cannot_certify_unverified_retry(monkeypatch, mock_config):
    class Generate(stage_module.Stage):
        name = "generate"

        def run(self, ctx):
            pass

    class Verify(stage_module.Stage):
        name = "verify"

        def run(self, ctx):
            if ctx.attempt == 0:
                ctx.verification_report = {"hard_failures": [{"type": "count_mismatch"}]}

    monkeypatch.setattr(stage_module, "build_stages", lambda config: {
        "generate": Generate(config), "verify": Verify(config),
    })
    config = {**mock_config, "pipeline": {"max_repair_loops": 1}}
    with pytest.raises(RuntimeError, match="attempt 1"):
        stage_module.Pipeline(config).run("one red cat")


@pytest.mark.parametrize("predicate", ["teleporting_to", "between"])
def test_unverifiable_relation_is_a_failure(predicate):
    boxes = {
        "cat": [{"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
        "chair": [{"xmin": 5, "ymin": 45, "xmax": 60, "ymax": 90}],
    }
    result = SpatialRelationVerifier().verify(
        [{"subject": "cat", "predicate": predicate, "object": "chair"}], boxes,
    )
    assert result["passed"] is False
    assert result["verified"] == 0
    assert len(result["failures"]) == 1


def test_on_relation_is_checked_as_support():
    boxes = {
        "cat": [{"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
        "chair": [{"xmin": 5, "ymin": 45, "xmax": 60, "ymax": 90}],
    }
    result = SpatialRelationVerifier().verify(
        [{"subject": "cat", "predicate": "on", "object": "chair"}], boxes,
    )
    assert result["passed"] is True
    assert "unknown" not in result["results"][0]["reason"]


def test_broad_boxes_cannot_certify_support_relation():
    # A previous accepted run had nearly full-image cat/chair boxes and
    # treated "on" as a skipped predicate.
    boxes = {
        "cat": [{"xmin": 73, "ymin": 42, "xmax": 388, "ymax": 512}],
        "chair": [{"xmin": 10, "ymin": 6, "xmax": 508, "ymax": 512}],
    }
    result = SpatialRelationVerifier().verify(
        [{"subject": "cat", "predicate": "on", "object": "chair"}], boxes,
    )
    assert result["passed"] is False
    assert "support_contact=fail" in result["results"][0]["reason"]


def test_detector_class_match_does_not_count_substrings():
    assert QAVerifier._label_matches_class("a cat", "cat") is True
    assert QAVerifier._label_matches_class("caterpillar", "cat") is False


def test_spatial_class_match_does_not_use_unrelated_box():
    result = SpatialRelationVerifier().verify(
        [{"subject": "cat", "predicate": "left_of", "object": "chair"}],
        {
            "caterpillar": [{"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20}],
            "chair": [{"xmin": 100, "ymin": 10, "xmax": 150, "ymax": 100}],
        },
    )
    assert result["passed"] is False
    assert "not detected" in result["failures"][0]["reason"]


def test_control_image_mode_is_explicit_and_observed():
    contract = {"prompt": "a cat beside a chair", "entities": [
        {"class_name": "cat", "attributes": []},
        {"class_name": "chair", "attributes": []},
    ]}
    planner = ScenePlanner(resolution=512)
    _, hints, _, _, masks = planner.plan(
        contract, gen_defaults={"control_image_mode": "canny"},
    )
    assert "global_mask" in masks
    assert hints["control_image_mode"] == "canny"
    assert hints["use_canny"] is True
    with pytest.raises(ValueError, match="control_image_mode"):
        planner.plan(contract, gen_defaults={"control_image_mode": "unknown"})


def test_scene_plan_does_not_mutate_extracted_contract():
    contract = {"prompt": "a red cat on a blue chair", "entities": [
        {"class_name": "cat", "attributes": ["red"], "bounding_box": None},
        {"class_name": "chair", "attributes": ["blue"], "bounding_box": None},
    ], "spatial_relationships": [
        {"subject": "cat", "predicate": "on", "object": "chair"},
    ]}
    scene_plan, _, _, _, _ = ScenePlanner().plan(contract)
    assert all(entity["bounding_box"] is None for entity in contract["entities"])
    assert all(entity["bounding_box"] for entity in scene_plan["entities"])


def test_preview_uses_the_final_decode_and_supported_default_settings(
        monkeypatch, tmp_path):
    from PIL import Image, ImageChops

    generated = Image.new("RGB", (8, 8), (220, 70, 30))

    class Pipe:
        _execution_device = "cpu"
        unet = SimpleNamespace(set_attn_processor=lambda processor: None)

        def encode_prompt(self, **kwargs):
            return None, None, None, None

        def __call__(self, **kwargs):
            assert "callback_on_step_end" not in kwargs  # no unsafe second VAE decode
            assert "image" not in kwargs  # ControlNet is opt-in
            return SimpleNamespace(images=[generated])

    engine = GenerationEngine({})
    assert engine.base_model == "stabilityai/sdxl-turbo"
    assert engine._offload_mode == "sequential"
    engine.pipe = Pipe()
    monkeypatch.setattr(engine, "ensure_loaded", lambda **kwargs: None)
    monkeypatch.setattr("src.generation.engine.resolve_model_source", lambda source: source)
    monkeypatch.setattr("src.generation.engine.free_memory", lambda: None)
    meta = engine.generate("a cat", {"steps": 4}, {}, {}, tmp_path / "generated_0.png")
    with Image.open(tmp_path / "generated_0.png") as draft, \
            Image.open(tmp_path / "step_0_3.png") as preview:
        assert ImageChops.difference(draft, preview).getbbox() is None
        assert preview.getextrema() != ((0, 0), (0, 0), (0, 0))
    assert meta["preview_source"] == "final_pipeline_decode"


def test_explicit_support_verb_is_not_weakened_to_above():
    from src.intent.extractor import IntentExtractor

    data = {
        "entities": [{"class_name": "cat"}, {"class_name": "chair"}],
        "spatial_relationships": [
            {"subject": "cat", "predicate": "above", "object": "chair"},
        ],
        "constraints": [{
            "type": "spatial_relation",
            "value": {"subject": "cat", "predicate": "above", "object": "chair"},
            "target": "cat->chair", "priority": "high",
        }],
    }
    extractor = object.__new__(IntentExtractor)
    extractor.uncountable_stoplist = set()
    extractor.router = SimpleNamespace(generate_json=lambda **kwargs: data)
    contract = extractor.extract("A red cat sitting on a blue chair")
    assert contract["spatial_relationships"][0]["predicate"] == "on"
    assert all(c["value"]["predicate"] == "on" for c in contract["constraints"])
    assert contract["extraction_corrections"][0]["from"] == "above"
    assert data["spatial_relationships"][0]["predicate"] == "above"

    negated = extractor.extract("A cat not sitting on a chair")
    assert negated["spatial_relationships"][0]["predicate"] == "above"


def test_offline_model_resolution_uses_cached_marker_not_full_snapshot(
        monkeypatch, tmp_path):
    marker = tmp_path / "snapshot" / "model_index.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")
    calls = []

    def cached_file(**kwargs):
        calls.append(kwargs)
        return str(marker)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", cached_file)
    assert resolve_model_source("stabilityai/sdxl-turbo") == str(marker.parent)
    assert calls == [{"repo_id": "stabilityai/sdxl-turbo",
                      "filename": "model_index.json", "local_files_only": True}]
    assert resolve_model_source(str(marker.parent)) == str(marker.parent.resolve())
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_source(str(tmp_path / "missing"))


def test_vision_fallback_tries_each_provider_once_without_hidden_retries(
        monkeypatch):
    from src.core.llm_router import LLMRouter, ProviderConfig

    router = object.__new__(LLMRouter)
    router.providers = [
        ProviderConfig(name="Local", base_url="http://local/v1", model="text",
                       vision_model="vision", priority=0),
        ProviderConfig(name="Backup", base_url="http://backup/v1", model="text",
                       vision_model="vision", priority=1),
    ]
    router.available_providers = router.providers
    calls = []

    class FailingClient:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.base_url = kwargs["base_url"]
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(self.base_url)
            raise TimeoutError("unavailable")

    monkeypatch.setattr("src.core.llm_router.OpenAI", FailingClient)
    with pytest.raises(RuntimeError, match="All vision LLM providers failed"):
        router.generate_vision([{"role": "user", "content": "Describe image"}])
    assert calls == ["http://local/v1", "http://backup/v1"]
    with pytest.raises(RuntimeError, match="All vision LLM providers failed"):
        router.generate_vision([{"role": "user", "content": "Describe image"}])
    assert len(calls) == 2  # cooldown prevents an immediate repeat


def test_accepted_report_has_no_invented_root_causes(tmp_path):
    report = Reporter(tmp_path).generate_report({}, {
        "accepted": True, "hard_failures": [],
        "signals": {"detected_counts": {"cat": 1}, "expected_counts": {"cat": 1}},
    })
    assert report["root_causes"] == []
    assert report["actions"] == []


def test_benchmark_error_is_retryable_and_excluded_from_scores(tmp_path):
    from pathlib import Path

    prompt = json.loads(
        (Path(__file__).resolve().parent.parent / "benchmark_subset.json").read_text()
    )[0]["prompt"]
    db = RunTrackerDB(str(tmp_path / "runs.db"))
    try:
        error_ctx = RunContext(prompt=prompt)
        error_ctx.config_name = "full"
        error_ctx.is_benchmark = True
        error_ctx.benchmark_seed = 42
        error_ctx.cache_key = "benchmark-v2:quality-test"
        error_ctx.verification_report = {"accepted": False, "error": "provider timeout"}
        db.log_run(error_ctx)
        assert db.has_run(prompt, "full", seed=42) is False
        assert db.all_benchmark_rows() == []

        failed_ctx = RunContext(prompt=prompt)
        failed_ctx.config_name = "full"
        failed_ctx.is_benchmark = True
        failed_ctx.benchmark_seed = 42
        failed_ctx.cache_key = "benchmark-v2:quality-test"
        failed_ctx.verification_report = {
            "accepted": False, "hard_failures": [{"type": "count_mismatch"}],
        }
        db.log_run(failed_ctx)
        assert db.has_run(prompt, "full", seed=42) is True
        assert db.has_run(prompt, "full", seed=42,
                          experiment_key="benchmark-v2:another-suite") is False
        assert len(db.all_benchmark_rows()) == 1
    finally:
        db.close()


def test_benchmark_dashboard_uses_one_fingerprinted_suite(tmp_path):
    db = RunTrackerDB(str(tmp_path / "suite.db"))
    try:
        for key, started in (("benchmark-v2:old", 1.0),
                             ("benchmark-v2:new", 2.0)):
            ctx = RunContext(prompt="a cat")
            ctx.config_name = "baseline"
            ctx.is_benchmark = True
            ctx.benchmark_seed = 42
            ctx.cache_key = key
            ctx.start_time = started
            ctx.is_successful = key.endswith("old")
            ctx.verification_report = {"accepted": ctx.is_successful,
                                       "hard_failures": [] if ctx.is_successful else
                                       [{"type": "count_mismatch"}]}
            db.log_run(ctx)
        assert db.latest_benchmark_experiment() == "benchmark-v2:new"
        rows = db.all_benchmark_rows()
        assert len(rows) == 1 and rows[0][2] == 0
        assert db.has_run("a cat", "baseline", seed=42,
                          experiment_key="benchmark-v2:new")
    finally:
        db.close()


def test_interactive_cache_requires_current_code_and_config(tmp_path):
    db = RunTrackerDB(str(tmp_path / "cache.db"))
    try:
        ctx = RunContext(prompt="a red cat")
        ctx.config_name = "interactive"
        ctx.is_successful = True
        ctx.verification_report = {"accepted": True, "hard_failures": []}
        ctx.cache_key = "old-policy"
        db.log_run(ctx)
        assert db.get_run_by_prompt(ctx.prompt, cache_key="old-policy") is not None
        assert db.get_run_by_prompt(ctx.prompt, cache_key="new-policy") is None
    finally:
        db.close()

    config = {"models": {"generator": "model-a"}}
    original = stage_module.current_cache_key(config)
    assert original != stage_module.current_cache_key(
        {"models": {"generator": "model-b"}}
    )


def test_lora_activation_failure_reaches_generation_metadata():
    class Engine:
        def register_pre_offload_hook(self, hook):
            self.hook = hook

    class Pipe:
        def load_lora_weights(self, source, adapter_name):
            pass

        def set_adapters(self, names, adapter_weights):
            raise RuntimeError("activation failed")

    stage = AdaptStage({"lora": {"adapters": [
        {"name": "style", "source": "local/style.safetensors"},
    ]}})
    engine = Engine()
    stage.bind_engine(engine)
    ctx = SimpleNamespace(generation_meta={})
    stage.run(ctx)
    engine.hook(Pipe())
    report = ctx.generation_meta["lora_load_report"]
    assert report[0]["loaded"] is True
    assert report[0]["activated"] is False
    assert "activation failed" in report[0]["activation_error"]


def test_lora_ablation_explicitly_enables_its_adapter():
    from benchmark import ALL_STAGES, benchmark_overrides

    baseline = benchmark_overrides("baseline", ALL_STAGES, 42)
    lora = benchmark_overrides("rag_lora", ALL_STAGES, 42)
    assert "lora" not in baseline
    assert len(lora["lora"]["adapters"]) == 1
    assert lora["lora"]["adapters"][0]["enabled"] is True
    assert lora["generation"]["seed"] == 42


def test_benchmark_execution_error_does_not_become_failed_image(monkeypatch):
    import benchmark

    class DB:
        def has_run(self, prompt, config_name, seed=None, experiment_key=None):
            return False

        def log_run(self, ctx):
            self.logged = ctx

    db = DB()

    class Pipeline:
        config = {}

        def __init__(self):
            self.db = db

        def run(self, *args, **kwargs):
            raise OSError("model unavailable")

    monkeypatch.setattr(benchmark, "FrameGenXPipeline", lambda **kwargs: Pipeline())
    monkeypatch.setattr(benchmark, "save_benchmark_manifest", lambda *args: "test-manifest")
    monkeypatch.setattr(benchmark, "load_subset", lambda path: [
        {"prompt": "a red cat", "category": "color_constraints"},
    ])
    result = benchmark.run_benchmark(config_name="baseline", limit=1)
    assert result["done"] == 0
    assert result["errors"] == 1
    assert result["by_category"] == {}
    assert db.logged.verification_report["error"] == "model unavailable"


def test_benchmark_manifest_changes_with_selected_cases():
    from benchmark import ALL_STAGES, build_benchmark_manifest

    first = build_benchmark_manifest([{"prompt": "a cat"}], "baseline", ALL_STAGES)
    same = build_benchmark_manifest([{"prompt": "a cat"}], "baseline", ALL_STAGES)
    other = build_benchmark_manifest([{"prompt": "a dog"}], "baseline", ALL_STAGES)
    assert first["experiment_key"] == same["experiment_key"]
    assert first["experiment_key"] != other["experiment_key"]
    assert first["checkpoint_revisions_verified"] is False


def test_diagnostic_scripts_abstain_when_reward_is_unavailable(monkeypatch, tmp_path):
    from scripts.tau_sweep import simulate_tau
    from scripts import reflect_diagnosis

    run_data = [{"attempts": [
        {"hard_failures": [], "reward": None},
    ]}]
    assert simulate_tau(run_data, 0.5)["accepted"] == 0

    attempts = [
        {"hard_failures": [], "reward": None, "accepted": False,
         "tau": 0.5, "prompt_edits": ["cat"], "final_prompt": "cat"},
        {"hard_failures": [], "reward": 0.7, "accepted": True,
         "tau": 0.5, "prompt_edits": [], "final_prompt": "cat"},
    ]
    monkeypatch.setattr(reflect_diagnosis, "load_attempt_artifacts",
                        lambda path: attempts)
    analysis = reflect_diagnosis.analyze_run(tmp_path, "cat", "full")
    assert analysis["transitions"][0]["reward_delta"] is None


def test_inactive_lora_cannot_be_scored_as_lora_ablation(mock_config):
    config = {**mock_config, "lora": {"adapters": [
        {"name": "style", "source": "local/style.safetensors", "enabled": True},
    ]}}
    stage = GenerateStage(config)
    stage.engine = SimpleNamespace(generate=lambda *args, **kwargs: {})
    ctx = RunContext(prompt="a cat", config=config)
    ctx.config_name = "rag_lora"
    ctx.is_benchmark = True
    ctx.final_prompt = "a cat"
    ctx.generation_meta["lora_load_report"] = [
        {"name": "style", "loaded": True, "activated": False},
    ]
    with pytest.raises(RuntimeError, match="LoRA ablation invalid"):
        stage.run(ctx)
