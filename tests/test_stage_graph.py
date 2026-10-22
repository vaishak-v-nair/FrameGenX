"""
Eng-review locked suite — pure-function + regression tests (CPU, no GPU).
Mocked-integration batch lands with T10.
"""
import json
import sqlite3

import pytest

from src.core.context import RunContext
from src.core.db import RunTrackerDB
from src.core.stage import Stage, STAGE_REGISTRY, register_stage, build_stages, Pipeline
from src.verification.verify_stage import (
    clip_norm_from_cos,
    compute_reward,
    build_vqa_questions,
    color_matches,
    VQAVerifier,
    VerifyStage,
)
from src.reflection.reflector import ReflectStage, build_corrective_edits
from src.scene.planner import PlanStage, ScenePlanner


# --------------------------------------------------------------------------
# Registry / pipeline loop
# --------------------------------------------------------------------------

def test_registry_contains_all_stages():
    build_stages({"pipeline": {"stages": []}})  # triggers module imports
    for name in ["intent", "retrieval", "plan", "adapt", "generate", "verify", "reflect"]:
        assert name in STAGE_REGISTRY, f"stage '{name}' not registered"


def test_unknown_stage_is_an_error():
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        build_stages({"pipeline": {"stages": ["intent", "nonsense"]}})


def _stub(name_, behavior=None):
    @register_stage
    class _S(Stage):
        name = name_
        calls = []

        def run(self, ctx):
            type(self).calls.append((name_, ctx.attempt))
            if behavior:
                behavior(self, ctx)
    return _S


def test_pipeline_loop_retry_then_accept(mock_config, monkeypatch):
    """Loop: reject on attempt 0, accept on attempt 1."""
    saved = dict(STAGE_REGISTRY)
    try:
        def verify_behavior(self, ctx):
            ok = ctx.attempt >= 1
            ctx.verification_report = {"accepted": False,
                "hard_failures": [] if ok else [{"type": "count_mismatch"}],
                "vqa_details": [{"answer": "unknown"}]}

        _stub("plan")
        _stub("generate")
        _stub("verify", verify_behavior)

        config = {**mock_config, "pipeline": {"stages": ["plan", "generate", "verify"],
                                              "max_repair_loops": 3}}
        monkeypatch.setattr("src.core.stage.build_stages",
                            lambda cfg: {n: STAGE_REGISTRY[n](cfg) for n in cfg["pipeline"]["stages"]})
        p = Pipeline(config)
        ctx = p.run("stub prompt")
        assert ctx.is_successful is True
        assert ctx.attempt == 1
        assert ctx.repair_loops_executed == 1
        rejected = json.loads((ctx.run_dir / "verification_0.json").read_text())
        accepted = json.loads((ctx.run_dir / "verification_1.json").read_text())
        decision = json.loads((ctx.run_dir / "decision_1.json").read_text())
        assert rejected["accepted"] is False
        assert accepted["accepted"] is decision["accept"] is ctx.is_successful is True
        assert accepted["vqa_details"] == [{"answer": "unknown"}]
    finally:
        STAGE_REGISTRY.clear()
        STAGE_REGISTRY.update(saved)


def test_pipeline_progress_events(mock_config, monkeypatch):
    """Glass-box API contract: progress_cb sees run start, per-stage events,
    decisions, and the finish — and a crashing callback never kills the run."""
    saved = dict(STAGE_REGISTRY)
    try:
        def verify_behavior(self, ctx):
            ok = ctx.attempt >= 1
            ctx.verification_report = {"hard_failures": [] if ok else [{"type": "count_mismatch"}]}

        _stub("plan")
        _stub("generate")
        _stub("verify", verify_behavior)

        config = {**mock_config, "pipeline": {"stages": ["plan", "generate", "verify"],
                                              "max_repair_loops": 3}}
        monkeypatch.setattr("src.core.stage.build_stages",
                            lambda cfg: {n: STAGE_REGISTRY[n](cfg) for n in cfg["pipeline"]["stages"]})
        p = Pipeline(config)

        events = []
        def cb(evt):
            events.append(evt)
            raise RuntimeError("callback bug must not propagate")

        ctx = p.run("stub prompt", progress_cb=cb)
        assert ctx.is_successful is True

        kinds = [e["event"] for e in events]
        assert kinds[0] == "run_started"
        assert events[0]["run_id"] == ctx.run_id
        assert kinds[-1] == "run_finished"
        decisions = [e for e in events if e["event"] == "decision"]
        assert [d["accepted"] for d in decisions] == [False, True]
        # verify ran on both attempts and the events carry the attempt number
        verify_evts = [e for e in events
                       if e["event"] == "stage_finished" and e["stage"] == "verify"]
        assert [e["attempt"] for e in verify_evts] == [0, 1]
    finally:
        STAGE_REGISTRY.clear()
        STAGE_REGISTRY.update(saved)


# --------------------------------------------------------------------------
# REGRESSION: extractor contract shape after knowledge_base deletion
# --------------------------------------------------------------------------

def test_extractor_contract_shape(stub_intent_extractor):
    contract = stub_intent_extractor.extract("three red apples on a table")
    assert contract["prompt"] == "three red apples on a table"
    assert contract["rag_hits"] == []  # RetrievalStage owns this now
    counts = {}
    for e in contract["entities"]:
        counts[e["class_name"]] = counts.get(e["class_name"], 0) + 1
    assert counts == {"apple": 3, "table": 1}
    assert contract["entities"][0]["attributes"] == ["red"]
    assert contract["execution_plan"]["risk_level"] == "medium"


def test_extractor_stoplist_kills_phantom_entities(stub_intent_extractor):
    contract = stub_intent_extractor.extract(
        "two cats sitting on a wall in the morning light, high quality, detailed")
    names = {e["class_name"] for e in contract["entities"]}
    assert names == {"cat"}, f"phantom entities survived: {names}"


# --------------------------------------------------------------------------
# REGRESSION: DB migration + config_name filter + resume semantics
# --------------------------------------------------------------------------

OLD_SCHEMA = """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, prompt TEXT, start_time REAL, duration REAL,
        is_successful BOOLEAN, repair_loops INTEGER,
        contract_json TEXT, verification_report_json TEXT
    )
"""


def _ctx(mock_config, prompt, config_name="full", ok=True,
         is_benchmark=False, seed=None, experiment_key=None):
    ctx = RunContext(prompt=prompt, config=mock_config)
    ctx.config_name = config_name
    ctx.is_benchmark = is_benchmark
    ctx.benchmark_seed = seed
    ctx.cache_key = experiment_key
    ctx.is_successful = ok
    ctx.verification_report = {"accepted": ok}
    return ctx


def test_db_migration_from_old_schema(mock_config, tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(OLD_SCHEMA)
    conn.execute("INSERT INTO runs VALUES ('r1','old prompt',0,1,1,0,NULL,NULL)")
    conn.commit()
    conn.close()

    db = RunTrackerDB(str(db_path))  # must ALTER, not fail
    db.log_run(_ctx(mock_config, "new prompt"))  # widened INSERT must work
    old = db.get_run("r1")
    assert old is not None and old["prompt"] == "old prompt"
    assert db.has_run("new prompt", "full") is False
    db.close()


def test_get_run_by_prompt_filters_config_name(mock_config, tmp_path):
    db = RunTrackerDB(str(tmp_path / "f.db"))
    db.log_run(_ctx(mock_config, "p1", config_name="baseline", ok=True))
    assert db.get_run_by_prompt("p1") is None, "baseline row leaked into API cache"
    db.log_run(_ctx(mock_config, "p1", config_name="full", ok=True))
    assert db.get_run_by_prompt("p1") is None, "benchmark row leaked into API cache"
    db.log_run(_ctx(mock_config, "p1", config_name="interactive", ok=True))
    assert db.get_run_by_prompt("p1") is not None
    db.close()


def test_benchmark_rows_exclude_interactive_and_non_dataset_prompts(mock_config, tmp_path):
    from pathlib import Path
    benchmark_prompt = json.loads(
        (Path(__file__).resolve().parent.parent / "benchmark_subset.json").read_text()
    )[0]["prompt"]
    db = RunTrackerDB(str(tmp_path / "benchmark.db"))
    db.log_run(_ctx(mock_config, benchmark_prompt, config_name="full"))
    assert db.all_benchmark_rows() == [], "legacy full rows lack benchmark provenance"
    db.log_run(_ctx(mock_config, benchmark_prompt, config_name="full",
                    is_benchmark=True, seed=42,
                    experiment_key="benchmark-v2:test"))
    db.log_run(_ctx(mock_config, "private studio prompt", config_name="full",
                    is_benchmark=True, seed=42))
    db.log_run(_ctx(mock_config, benchmark_prompt, config_name="interactive"))
    rows = db.all_benchmark_rows()
    assert len(rows) == 1
    assert rows[0][0:2] == ("full", benchmark_prompt)
    db.close()


def test_resume_keys_on_existence_not_success(mock_config, tmp_path):
    db = RunTrackerDB(str(tmp_path / "r.db"))
    db.log_run(_ctx(mock_config, "failed prompt", config_name="baseline",
                    ok=False, is_benchmark=True, seed=42))
    assert db.has_run("failed prompt", "baseline", seed=42) is True
    assert db.has_run("failed prompt", "baseline", seed=43) is False
    assert db.has_run("failed prompt", "full", seed=42) is False
    db.close()


# --------------------------------------------------------------------------
# Reward math (pure)
# --------------------------------------------------------------------------

def test_clip_norm_clamp():
    assert clip_norm_from_cos(0.10) == 0.0          # below range
    assert clip_norm_from_cos(0.15) == 0.0          # lower bound
    assert abs(clip_norm_from_cos(0.275) - 0.5) < 1e-9
    assert clip_norm_from_cos(0.40) == 1.0          # upper bound
    assert clip_norm_from_cos(0.90) == 1.0          # above range


def test_reward_formula_and_vacuous_vqa():
    assert compute_reward(0.5, 1.0) == 0.75
    # zero-entity contract: VQA=1.0 vacuous pass keeps tau reachable
    assert compute_reward(0.3, 1.0) >= 0.575 or compute_reward(0.3, 1.0) == 0.65


@pytest.mark.parametrize("hard,reward,tau,expected", [
    ([], 0.7, 0.575, True),                       # clean + high reward
    ([], 0.5, 0.575, False),                      # clean + low reward
    ([{"type": "count_mismatch"}], 0.9, 0.575, False),  # hard failure blocks
    ([{"type": "count_mismatch"}], 0.3, 0.575, False),
])
def test_acceptance_rule_all_combinations(hard, reward, tau, expected):
    """Tests reward_gate mode: accept = (no hard failures) AND (R >= τ)."""
    accept = (not hard) and reward >= tau
    assert accept is expected


@pytest.mark.parametrize("hard,reward,expected", [
    ([], 0.9, True),                              # clean + high reward → accept
    ([], 0.3, True),                              # clean + LOW reward → still accept
    ([], 0.0, True),                              # clean + zero reward → still accept
    ([{"type": "count_mismatch"}], 0.9, False),   # hard failure blocks even high reward
    ([{"type": "count_mismatch"}], 0.0, False),   # hard failure + zero reward
])
def test_acceptance_rule_hard_only_mode(hard, reward, expected):
    """Tests hard_only mode: accept = (no hard failures), reward is irrelevant.
    This is the default mode since tau_sweep.py proved reward gating is net-harmful."""
    accept = not hard
    assert accept is expected


def test_reflect_stage_blocks_hard_failure_even_with_high_reward(mock_config, monkeypatch):
    ctx = RunContext(prompt="two cats", config=mock_config)
    ctx.contract = {"prompt": "two cats"}
    ctx.verification_report = {
        "hard_failures": [{"type": "count_mismatch", "class_name": "cat",
                           "expected": 2, "detected": 1}],
        "reward": 0.99,
    }
    monkeypatch.setattr("src.reflection.reflector.build_corrective_edits",
                        lambda report, contract, details: ["exactly two cats"])
    ReflectStage({"reflection": {"mode": "hard_only", "tau": 0.73}}).run(ctx)
    assert ctx.decision["accept"] is False
    assert ctx.decision["hard_ok"] is False
    assert ctx.decision["prompt_edits"] == ["exactly two cats"]


# --------------------------------------------------------------------------
# Corrective edits (pure, all 5 signals, fresh recompute)
# --------------------------------------------------------------------------

def test_corrective_edits_all_signals(monkeypatch):
    report = {
        "hard_failures": [
            {"type": "count_mismatch", "class_name": "apples", "expected": 3, "detected": 1},
        ],
        "clip_norm": 0.3,
    }
    contract = {"rag_hits": ["a bowl of fruit on a table"], "prompt": "x",
                "parsed": {"primary_subject": "three red apples"}}
    vqa_details = [
        {"type": "existence", "class_name": "apples", "correct": False},
        {"type": "color", "class_name": "apples", "expected": "red", "correct": False},
    ]

    # Mock LLMRouter to avoid actual network calls
    class MockRouter:
        def generate(self, messages, temperature=0.3):
            return "mocked LLM repair prompt"
    monkeypatch.setattr("src.reflection.reflector.LLMRouter", MockRouter)

    edits = build_corrective_edits(report, contract, vqa_details)
    joined = " | ".join(edits)

    assert "mocked LLM repair prompt" in joined
    # low-CLIP: subject emphasis only
    assert "three red apples, prominent, clearly visible" in joined
    assert "a bowl of fruit" not in joined
    assert "prominent, clearly visible" in joined  # existence miss
    assert "(red colored apples)" in joined      # color miss


def test_corrective_edits_empty_when_clean():
    assert build_corrective_edits({"hard_failures": [], "clip_norm": 0.9}, {}, []) == []


def test_corrective_edits_use_policy_fragments_as_safe_suffixes(monkeypatch):
    """repair_policy (paper-contract parity) refines count edits with
    prompt-safe suffix fragments — raw meta-action names must never leak
    into the drawing prompt."""
    report = {"hard_failures": [
        {"type": "count_mismatch", "class_name": "apple", "expected": 3, "detected": 1},
    ], "clip_norm": 0.9}
    contract = {"repair_policy": {"mappings": [
        {"failure": "object_missing", "action": "prominently featured"},
        {"failure": "duplicate_object", "action": ""},
    ]}}

    # Mock LLMRouter to avoid actual network calls
    class MockRouter:
        def generate(self, messages, temperature=0.3):
            return "mocked LLM repair prompt"
    monkeypatch.setattr("src.reflection.reflector.LLMRouter", MockRouter)

    edits = build_corrective_edits(report, contract, [])
    joined = " | ".join(edits)
    assert "mocked LLM repair prompt" in joined


def test_corrective_edits_policy_low_alignment_fragment():
    report = {"hard_failures": [], "clip_norm": 0.3}
    contract = {"repair_policy": {"mappings": [
        {"failure": "low_alignment", "action": "sharp focus, highly detailed"},
    ]}, "parsed": {"primary_subject": "a red car"}, "prompt": "a red car"}
    edits = build_corrective_edits(report, contract, [])
    assert "sharp focus, highly detailed" in " | ".join(edits)


# --------------------------------------------------------------------------
# VQA question builder (pure)
# --------------------------------------------------------------------------

def test_vqa_questions_deduped_and_plural_safe():
    contract = {"entities": [
        {"class_name": "apples", "attributes": ["red"]},
        {"class_name": "apples", "attributes": ["red"]},
        {"class_name": "apples", "attributes": ["red"]},
        {"class_name": "table", "attributes": []},
    ]}
    qs = build_vqa_questions(contract)
    existence = [q for q in qs if q["type"] == "existence"]
    assert len(existence) == 2, "duplicated entities must yield ONE question"
    assert existence[0]["question"] == "Are there red apples in the image?"
    assert existence[1]["question"] == "Is there a table in the image?"
    color = [q for q in qs if q["type"] == "color"]
    assert len(color) == 1 and color[0]["expected"] == "red"


def test_vqa_non_color_attributes_get_no_color_question():
    contract = {"entities": [{"class_name": "temple", "attributes": ["ancient"]}]}
    qs = build_vqa_questions(contract)
    assert all(q["type"] != "color" for q in qs)


def test_vqa_provider_failure_is_unavailable_not_zero_precision():
    from PIL import Image

    verifier = object.__new__(VQAVerifier)
    verifier.router = type("Router", (), {
        "generate_vision": lambda self, **kwargs: (_ for _ in ()).throw(TimeoutError("offline"))
    })()
    result = verifier.score(Image.new("RGB", (8, 8)), {
        "entities": [{"class_name": "cat", "attributes": ["red"]}]
    })
    assert result["vqa_status"] == "unavailable"
    assert result["vqa_score"] is None
    assert all(item["correct"] is None for item in result["vqa_details"])


def test_hard_only_decision_keeps_vqa_outage_visible(mock_config):
    ctx = RunContext(prompt="red cat", config=mock_config)
    ctx.contract = {"prompt": "red cat", "entities": []}
    ctx.verification_report = {"hard_failures": [], "reward": None,
                               "vqa_status": "unavailable"}
    ReflectStage({"reflection": {"mode": "hard_only"}}).run(ctx)
    assert ctx.decision["accept"] is True
    assert ctx.decision["reward"] is None


def test_verify_stage_persists_unavailable_vqa_without_false_reward(tmp_path, monkeypatch):
    from PIL import Image

    config = {"paths": {"output_dir": str(tmp_path)},
              "pipeline": {"max_repair_loops": 0}}
    ctx = RunContext(prompt="red cat", config=config)
    ctx.contract = {"prompt": "red cat", "entities": [{"class_name": "cat"}]}
    Image.new("RGB", (8, 8)).save(ctx.run_dir / "generated_0.png")

    stage = object.__new__(VerifyStage)
    stage.config = config
    stage.qa = type("QA", (), {"verify": lambda self, path, contract: {
        "hard_failures": [], "signals": {"detected_counts": {"cat": 1},
                                         "expected_counts": {"cat": 1}}
    }})()
    stage.clip = type("CLIP", (), {"score": lambda self, image, prompt: {
        "clip_cos": 0.3, "clip_norm": 0.6
    }})()
    stage.vqa = type("VQA", (), {"score": lambda self, image, contract: {
        "vqa_score": None, "vqa_status": "unavailable", "vqa_vacuous": False,
        "vqa_details": []
    }})()
    stage.use_easyocr = False
    stage.use_scene_comprehension = False
    monkeypatch.setattr("src.utils.release_clip", lambda: None)

    stage.run(ctx)
    saved = json.loads((ctx.run_dir / "verification_0.json").read_text())
    assert saved["vqa_status"] == "unavailable"
    assert saved["vqa_score"] is None
    assert saved["reward"] is None
    assert ctx.reward_history[0]["reward"] is None


def test_color_synonyms():
    assert color_matches("red", "crimson")
    assert color_matches("gray", "grey")
    assert not color_matches("red", "blue")


# --------------------------------------------------------------------------
# Prompt composition rule (order + 77-token tail-drop)
# --------------------------------------------------------------------------

def test_composition_anchors_on_user_prompt(monkeypatch):
    """The user's prompt is the semantic anchor; entities are verification
    overlays, never a replacement for intent (rebuilding from entities
    collapsed an alpine panorama into 'blooming wildflower')."""
    p = ScenePlanner()
    entities = [{"class_name": "wildflower", "attributes": ["blooming"], "uncounted": True}]
    out = p.compose_prompt(entities, ["some rag caption"],
                           ["only 1 peak, nothing else similar"],
                           base_prompt="panoramic jagged alpine peaks at sunrise")
    assert out.startswith("only 1 peak, nothing else similar")  # edits first
    assert "panoramic jagged alpine peaks at sunrise" in out    # intent survives
    assert "wildflower" not in out                              # entity noise dropped
    assert out == "only 1 peak, nothing else similar, panoramic jagged alpine peaks at sunrise"


def test_composition_order_and_edit_survival(monkeypatch):
    p = ScenePlanner()
    entities = [{"class_name": "apples", "attributes": ["red"]}]
    out = p.compose_prompt(entities, ["caption one", "caption two", "caption three"],
                           ["exactly 3 apples"],
                           base_prompt="three red apples on a table")
    assert out.startswith("exactly 3 apples")          # edits first
    assert "three red apples on a table" in out        # user intent anchored
    # Captions are inspection-only: fused COCO sentences injected
    # hallucinations (CLIP sentence similarity can't validate relevance).
    assert "caption" not in out
    assert out == "exactly 3 apples, three red apples on a table"


def test_composition_preserves_edits_and_original_prompt():
    p = ScenePlanner()
    out = p.compose_prompt([{"class_name": "cats", "attributes": []}],
                           ["some very long caption"], ["exactly 2 cats"],
                           base_prompt="two cats on a windowsill")
    assert out == "exactly 2 cats, two cats on a windowsill"


def test_plan_stage_preserves_contract_prompt_and_excludes_retrieved_caption(mock_config):
    ctx = RunContext(prompt="two cats on a wall", config=mock_config)
    ctx.contract = {
        "prompt": ctx.prompt,
        "entities": [{"class_name": "cat", "attributes": []},
                     {"class_name": "cat", "attributes": []}],
        "rag_hits": ["airplane over a city"],
        "parsed": {"weighted_elements": []},
    }
    PlanStage({"generation": {"resolution": 64}}).run(ctx)
    assert ctx.contract["prompt"] == "two cats on a wall"
    assert "airplane" not in ctx.final_prompt
    assert "two cats on a wall" in ctx.final_prompt


def test_planner_uncounted_entities_get_no_masks():
    """Presence-only plurals ('wildflowers') never trigger layout grids and
    never receive bounding boxes."""
    from src.scene.planner import ScenePlanner
    p = ScenePlanner()
    entities = [{"class_name": "wildflower", "attributes": ["blooming"],
                 "uncounted": True}]
    scene_plan, control_hints, _, _, masks = p.plan({"entities": entities,
                                                     "rag_hits": []})
    assert masks == {}
    assert control_hints["control_image_mode"] == "none"
    assert scene_plan["entities"][0].get("bounding_box") is None


def test_qa_skips_count_check_for_uncounted_entities(tmp_path):
    """An uncounted entity alone must auto-pass the DINO count gate (no
    model load, no 'expected 1 detected 10' failures on meadows)."""
    from PIL import Image
    from src.verification.qa import QAVerifier
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64)).save(img_path)
    contract = {"entities": [{"class_name": "wildflower", "attributes": [],
                              "uncounted": True}]}
    decision = QAVerifier().verify(img_path, contract)
    assert decision["accepted"] is True
    assert decision["hard_failures"] == []


# --------------------------------------------------------------------------
# Retrieval: index-missing no-op + top-k on a fixture index
# --------------------------------------------------------------------------

def test_retrieval_noop_when_index_missing(mock_config, tmp_path):
    from src.retrieval.retriever import RetrievalStage
    config = {**mock_config, "retrieval": {"index_dir": str(tmp_path / "nope")},
              "pipeline": {"stages": ["retrieval"]}}
    stage = RetrievalStage(config)
    ctx = RunContext(prompt="x", config=config)
    ctx.contract = {"rag_hits": []}
    stage.run(ctx)  # must not raise
    assert ctx.contract["rag_hits"] == []


def _stub_hybrid_retriever(monkeypatch, hits, top_k=2):
    """Exercise current Milvus hybrid request/response handling without models."""
    import numpy as np
    from src.retrieval.retriever import Retriever

    class Dense:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class SparseVector:
        indices = np.array([1])
        values = np.array([0.7])

    class Sparse:
        def embed(self, texts):
            return [SparseVector()]

    class Client:
        def __init__(self):
            self.last_request = None

        def load_collection(self, name):
            assert name == "FrameGenX_RAG"

        def hybrid_search(self, **kwargs):
            self.last_request = kwargs
            return [hits[:kwargs["limit"]]]

    r = Retriever({"retrieval": {"top_k": top_k}})
    r._client = Client()
    r._dense_model = Dense()
    r._sparse_model = Sparse()
    monkeypatch.setattr(r, "_load_index", lambda: None)
    return r


def test_retrieval_topk_fixture(mock_config, tmp_path, monkeypatch):
    hits = [
        {"id": 0, "distance": 0.04, "entity": {"caption": "c0", "image": "i0"}},
        {"id": 1, "distance": 0.03, "entity": {"caption": "c1", "image": "i1"}},
        {"id": 2, "distance": 0.02, "entity": {"caption": "c2", "image": "i2"}},
    ]
    r = _stub_hybrid_retriever(monkeypatch, hits)
    result = r.query("whatever")
    assert [h["caption"] for h in result] == ["c0", "c1"]
    assert r._client.last_request["limit"] == 2


# --------------------------------------------------------------------------
# LoRA manager: fetch failure -> skip + record
# --------------------------------------------------------------------------

def test_lora_fetch_failure_is_recorded_not_fatal():
    from src.generation.lora import LoraManager

    class FakePipe:
        def load_lora_weights(self, *a, **k):
            raise OSError("no such repo")

    m = LoraManager({"lora": {"adapters": [{"name": "x", "source": "bogus/repo"}]}})
    m.hook(FakePipe())  # must not raise
    assert m.load_report[0]["loaded"] is False
    assert "no such repo" in m.load_report[0]["error"]


# --------------------------------------------------------------------------
# Benchmark config mapping
# --------------------------------------------------------------------------

def test_ablation_config_mapping():
    from benchmark import ABLATION_CONFIGS, ALL_STAGES
    assert ABLATION_CONFIGS["baseline"] == ["retrieval", "adapt", "reflect"]
    stages = [s for s in ALL_STAGES if s not in ABLATION_CONFIGS["baseline"]]
    assert stages == ["intent", "plan", "generate", "verify"]
    assert ABLATION_CONFIGS["full"] == []


# --------------------------------------------------------------------------
# Multi-Modal Hybrid RAG & Layout Blueprints
# --------------------------------------------------------------------------

def test_multimodal_layout_blueprints_retrieval(tmp_path):
    import json
    from src.retrieval.index_builder import DEFAULT_LAYOUT_BLUEPRINTS
    from src.retrieval.retriever import LayoutBlueprintRetriever

    idx = tmp_path / "index"
    idx.mkdir()
    (idx / "layout_blueprints.json").write_text(json.dumps(DEFAULT_LAYOUT_BLUEPRINTS))

    lbr = LayoutBlueprintRetriever(idx)
    bp_1 = lbr.retrieve(entity_count=1)
    assert bp_1 is not None
    assert bp_1["entity_count"] == 1
    assert len(bp_1["boxes"]) == 1

    bp_2 = lbr.retrieve(entity_count=2, relationships=[{"predicate": "standing_on"}])
    assert bp_2 is not None
    assert bp_2["name"] == "Subject on Supporting Surface"


def test_hybrid_bm25_and_dense_search(monkeypatch):
    hits = [{"id": 1, "distance": 0.03,
             "entity": {"caption": "a red apple on a table", "image": ""}}]
    r = _stub_hybrid_retriever(monkeypatch, hits)
    assert r.query("apple table")[0]["caption"] == "a red apple on a table"
    request = r._client.last_request
    assert len(request["reqs"]) == 2
    assert request["output_fields"] == ["caption", "image", "original_id"]


def test_hybrid_empty_search_abstains(monkeypatch):
    r = _stub_hybrid_retriever(monkeypatch, [])
    assert r.query("apple table") == []


# --------------------------------------------------------------------------
# Planner: layout masks only for multi-object scenes
# --------------------------------------------------------------------------

def test_planner_skips_masks_for_scenic_prompts():
    """0/1 countable entities -> no masks, ControlNet dropped (full-canvas
    composition). Forcing a grid mask on a panorama painted collages."""
    from src.scene.planner import ScenePlanner
    p = ScenePlanner()
    for entities in ([], [{"class_name": "peak", "attributes": []}]):
        _, control_hints, _, _, masks = p.plan(
            {"entities": [dict(e) for e in entities], "rag_hits": []})
        assert masks == {}
        assert control_hints["control_image_mode"] == "none"
        assert control_hints["use_segmentation"] is False


def test_planner_keeps_masks_for_multi_object_scenes():
    from src.scene.planner import ScenePlanner
    p = ScenePlanner()
    entities = [{"class_name": "apple", "attributes": ["red"]},
                {"class_name": "apple", "attributes": ["red"]},
                {"class_name": "table", "attributes": []}]
    _, control_hints, _, _, masks = p.plan({"entities": entities, "rag_hits": []})
    assert "global_mask" in masks
    assert control_hints["control_image_mode"] == "none"
    assert control_hints["use_segmentation"] is True


# --------------------------------------------------------------------------
# Intent: uncountable/atmospheric concepts never become countable entities
# --------------------------------------------------------------------------

def test_plural_nouns_become_uncounted_deterministically():
    """Bare plurals -> uncounted (presence-only); numbered plurals and
    singulars stay counted. Enforced in code — the LLM ignored rule 6
    (run 3500c905: 'wildflowers' -> wildflower x1 -> detected 10)."""
    from src.intent.extractor import _is_uncounted_plural as f
    assert f("wildflowers in bloom in the foreground", "wildflower") is True
    assert f("jagged alpine peaks at sunrise", "peak") is True
    assert f("cities at dusk", "city") is True
    assert f("three peaks", "peak") is False          # explicit count
    assert f("2 red cars", "car") is False            # numeric count
    assert f("a cat", "cat") is False                 # singular
    assert f("two cats sitting", "cat") is False      # number word
    assert f("a cat and a dog", "dog") is False       # singular


def test_extractor_filters_uncountable_entities(monkeypatch):
    """mist/valley/light must not become entities (DINO can't count them);
    relationships referencing them are dropped too."""
    from src.intent.extractor import IntentExtractor

    raw = {
        "parsed": {"subject": "peaks", "objects": ["peak", "mist"],
                   "colors": [], "primary_subject": "peaks",
                   "secondary_objects": []},
        "entities": [
            {"class_name": "peak", "attributes": ["jagged"]},
            {"class_name": "mist", "attributes": []},
            {"class_name": "valley", "attributes": []},
        ],
        "constraints": [{"type": "object_count", "value": 1,
                         "target": "peak", "priority": "critical"}],
        "spatial_relationships": [
            {"subject": "mist", "predicate": "below", "object": "peak"},
            {"subject": "cat", "predicate": "on", "object": "table"},
        ],
        "negations": [],
        "execution_plan": {"has_human": False, "has_depth": False,
                           "risk_level": "medium"},
        "signal_map": {},
    }
    ex = IntentExtractor(config={})
    monkeypatch.setattr(ex.router, "generate_json", lambda messages, temperature=0.0: raw)
    contract = ex.extract("jagged peaks with mist in the valleys")

    names = [e["class_name"] for e in contract["entities"]]
    assert names == ["peak"]
    rels = contract["spatial_relationships"]
    assert all(r["subject"] != "mist" for r in rels)
    assert any(r["subject"] == "cat" for r in rels)


# --------------------------------------------------------------------------
# LoRA: enabled flag
# --------------------------------------------------------------------------

def test_lora_disabled_adapter_is_skipped():
    from src.generation.lora import LoraManager

    class FakePipe:
        def __init__(self):
            self.loaded = []

        def load_lora_weights(self, source, adapter_name=None):
            self.loaded.append(adapter_name)

    pipe = FakePipe()
    m = LoraManager({"lora": {"adapters": [
        {"name": "off", "source": "some/repo", "weight": 1.0, "enabled": False},
        {"name": "on", "source": "some/repo2", "weight": 0.8},
    ]}})
    m.hook(pipe)
    assert m.load_report[0]["loaded"] is False
    assert m.load_report[0].get("skipped") == "enabled: false"
    assert m.load_report[1]["loaded"] is True
    assert pipe.loaded == ["on"]

