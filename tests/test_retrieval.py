"""
Tests for the Retrieval subsystem — validates the current Milvus hybrid
response contract, layout blueprint selection, and style preset matching.

Uses synthetic fixtures (no CLIP model loading, no GPU required).
"""
import json
import io
from types import SimpleNamespace
from contextlib import redirect_stdout
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.index_builder import (
    DEFAULT_LAYOUT_BLUEPRINTS,
    DEFAULT_STYLE_PRESETS,
)
from src.retrieval.retriever import (
    LayoutBlueprintRetriever,
    Retriever,
    StylePresetRetriever,
    tokenize,
    RetrievalStage,
)


def test_unicode_rewrite_survives_windows_console_encoding():
    from src.retrieval.agent import RAGAgent

    agent = object.__new__(RAGAgent)
    agent.llm_router = type("Router", (), {
        "generate": lambda self, **kwargs: "a red cat ‑ on a blue chair"
    })()
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with redirect_stdout(stream):
        result = agent.rewrite_query({"prompt": "cat on chair"})
    assert result["rewritten_query"] == "a red cat ‑ on a blue chair"


# ============================================================================
# 1. Hybrid retrieval response mapping
# ============================================================================

class TestHybridRetriever:
    @pytest.fixture
    def retriever(self, monkeypatch):
        import numpy as np

        class Dense:
            def encode(self, texts, convert_to_numpy=True):
                assert texts == ["two cats"]
                return np.array([[0.8, 0.2]], dtype=np.float32)

        class SparseVector:
            indices = np.array([4, 9])
            values = np.array([0.4, 0.9])

        class Sparse:
            def embed(self, texts):
                assert texts == ["two cats"]
                return [SparseVector()]

        class Client:
            def __init__(self):
                self.requests = None
                self.closed = False
                self.results = [[
                    {"id": 7, "distance": 0.03,
                     "entity": {"caption": "two cats on a wall", "image": "local-image"}},
                ]]

            def load_collection(self, name):
                assert name == "FrameGenX_RAG"

            def hybrid_search(self, **kwargs):
                self.requests = kwargs
                return self.results

            def close(self):
                self.closed = True

        retriever = Retriever({"retrieval": {"top_k": 2}})
        retriever._client = Client()
        retriever._dense_model = Dense()
        retriever._sparse_model = Sparse()
        monkeypatch.setattr(retriever, "_load_index", lambda: None)
        return retriever

    def test_dense_and_sparse_requests_are_sent_together(self, retriever):
        hits = retriever.query("two cats")
        requests = retriever._client.requests
        assert len(requests["reqs"]) == 2
        assert requests["limit"] == 2
        assert requests["output_fields"] == ["caption", "image", "original_id"]
        assert hits == [{
            "id": "7", "caption": "two cats on a wall", "image": "local-image",
            "score": 0.03, "retrieval_mode": "hybrid_milvus_rrf",
        }]

    def test_empty_hybrid_result_abstains(self, retriever):
        retriever._client.results = [[]]
        assert retriever.query("two cats") == []

    def test_unload_releases_milvus_connection(self, retriever):
        client = retriever._client
        retriever.unload()
        assert client.closed is True
        assert retriever._client is None

    def test_missing_index_abstains(self, tmp_path):
        retriever = Retriever({"retrieval": {"index_dir": str(tmp_path)}})
        assert retriever.index_available() is False
        assert retriever.query("two cats") == []


# ============================================================================
# 2. Tokenizer
# ============================================================================

class TestTokenizer:
    def test_lowercase(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = tokenize("cat, dog. bird!")
        assert "cat" in tokens
        assert "dog" in tokens
        assert "bird" in tokens

    def test_handles_numbers(self):
        tokens = tokenize("3 red apples")
        assert "3" in tokens
        assert "red" in tokens

    def test_empty_string(self):
        tokens = tokenize("")
        assert tokens == []


# ============================================================================
# 3. Layout Blueprint Retriever
# ============================================================================

class TestLayoutBlueprintRetriever:
    @pytest.fixture
    def retriever(self, tmp_path):
        bp_path = tmp_path / "layout_blueprints.json"
        bp_path.write_text(json.dumps(DEFAULT_LAYOUT_BLUEPRINTS), encoding="utf-8")
        return LayoutBlueprintRetriever(tmp_path)

    def test_single_entity_returns_centered(self, retriever):
        bp = retriever.retrieve(1)
        assert bp is not None
        assert bp["entity_count"] == 1
        assert len(bp["boxes"]) == 1

    def test_two_entities_returns_side_by_side(self, retriever):
        bp = retriever.retrieve(2)
        assert bp is not None
        assert bp["entity_count"] == 2
        assert len(bp["boxes"]) == 2

    def test_three_entities_returns_trio(self, retriever):
        bp = retriever.retrieve(3)
        assert bp is not None
        assert bp["entity_count"] == 3

    def test_four_entities_returns_grid(self, retriever):
        bp = retriever.retrieve(4)
        assert bp is not None
        assert bp["entity_count"] == 4

    def test_five_entities_returns_cluster(self, retriever):
        bp = retriever.retrieve(5)
        assert bp is not None
        assert bp["entity_count"] == 5

    def test_fallback_for_large_count(self, retriever):
        """Entity counts beyond templates should fallback to closest."""
        bp = retriever.retrieve(10)
        assert bp is not None  # Should not crash

    def test_support_relation_selects_table_layout(self, retriever):
        bp = retriever.retrieve(2, relationships=[
            {"predicate": "on_top_of", "subject": "apple", "object": "table"}
        ])
        assert bp is not None
        assert "table" in bp.get("name", "").lower() or "support" in bp.get("name", "").lower()

    @pytest.mark.parametrize("names", [("cat", "chair"), ("chair", "cat")])
    def test_support_boxes_follow_entity_roles_in_either_noun_order(self, retriever, names):
        entities = [{"id": name, "class_name": name} for name in names]
        bp = retriever.retrieve(2, relationships=[
            {"predicate": "on", "subject": "cat", "object": "chair"},
        ], entities=entities)
        box_by_name = dict(zip(names, bp["boxes"]))
        assert box_by_name["cat"]["ymin"] < box_by_name["chair"]["ymin"]
        assert box_by_name["chair"]["ymax"] == 1.0
        assert bp["entity_bindings"] == list(names)
        assert retriever.blueprints["2_entities_subject_table"]["boxes"][0]["ymax"] == 1.0

    def test_empty_blueprints(self, tmp_path):
        bp_path = tmp_path / "layout_blueprints.json"
        bp_path.write_text("{}", encoding="utf-8")
        retriever = LayoutBlueprintRetriever(tmp_path)
        bp = retriever.retrieve(1)
        assert bp is None

    def test_missing_file(self, tmp_path):
        retriever = LayoutBlueprintRetriever(tmp_path)
        bp = retriever.retrieve(1)
        assert bp is None  # graceful fallback


def test_retrieval_stage_uses_spatial_contract_and_only_counted_entities(
        tmp_path, monkeypatch):
    from src.core.context import RunContext

    (tmp_path / "layout_blueprints.json").write_text(
        json.dumps(DEFAULT_LAYOUT_BLUEPRINTS), encoding="utf-8")
    stage = object.__new__(RetrievalStage)
    stage.retriever = SimpleNamespace(
        is_available=lambda: True, index_dir=tmp_path,
        unload=lambda: None,
        layout_retriever=LayoutBlueprintRetriever(tmp_path),
        style_retriever=SimpleNamespace(retrieve=lambda prompt: {}),
    )
    stage.rag_agent = SimpleNamespace(invoke=lambda prompt: [])
    monkeypatch.setattr("src.retrieval.memory.FailureMemoryDB", lambda: SimpleNamespace(
        retrieve_warnings=lambda *args, **kwargs: [], unload=lambda: None,
    ))
    ctx = RunContext(prompt="a cat on a chair with birds", config={
        "paths": {"output_dir": str(tmp_path / "runs")},
    })
    ctx.contract = {
        "entities": [{"class_name": "cat"}, {"class_name": "chair"},
                     {"class_name": "bird", "uncounted": True}],
        "spatial_relationships": [{"subject": "cat", "predicate": "on", "object": "chair"}],
    }
    stage.run(ctx)
    assert ctx.contract["retrieved_layout"]["name"] == "Subject on Supporting Surface"
    assert ctx.contract["retrieved_layout"]["entity_bindings"] == ["cat", "chair"]


def test_local_milvus_requests_dedicated_channel_without_idle_pings(monkeypatch, tmp_path):
    from src.retrieval.milvus_client import open_local_milvus

    kwargs_seen = {}
    monkeypatch.setattr("pymilvus.MilvusClient", lambda **kwargs: kwargs_seen.update(kwargs))
    open_local_milvus(tmp_path / "memory.db")
    assert kwargs_seen["dedicated"] is True
    assert kwargs_seen["grpc_options"]["grpc.keepalive_permit_without_calls"] is False
    assert kwargs_seen["grpc_options"]["grpc.keepalive_time_ms"] >= 60000


# ============================================================================
# 4. Style Preset Retriever
# ============================================================================

class TestStylePresetRetriever:
    @pytest.fixture
    def retriever(self, tmp_path):
        sp_path = tmp_path / "style_presets.json"
        sp_path.write_text(json.dumps(DEFAULT_STYLE_PRESETS), encoding="utf-8")
        return StylePresetRetriever(tmp_path)

    def test_photorealistic_default(self, retriever):
        result = retriever.retrieve("a red car")
        assert result["preset_name"] == "photorealistic"

    def test_cinematic_match(self, retriever):
        result = retriever.retrieve("a cinematic shot of a castle")
        assert result["preset_name"] == "cinematic"

    def test_pixel_art_match(self, retriever):
        result = retriever.retrieve("pixel art character")
        assert result["preset_name"] == "pixel_art"

    def test_cyberpunk_match(self, retriever):
        result = retriever.retrieve("a cyberpunk city at night")
        assert result["preset_name"] == "cyberpunk_neon"

    def test_result_contains_descriptors(self, retriever):
        result = retriever.retrieve("a photo of a cat")
        assert "descriptors" in result
        assert len(result["descriptors"]) > 0

    def test_missing_file_returns_default(self, tmp_path):
        retriever = StylePresetRetriever(tmp_path)
        result = retriever.retrieve("anything")
        assert result["preset_name"] == "photorealistic"


# ============================================================================
# 5. Box Geometry Validation
# ============================================================================

class TestBlueprintBoxGeometry:
    """Validates that layout blueprint boxes have valid normalized coordinates."""

    def test_all_boxes_within_unit_square(self):
        for name, bp in DEFAULT_LAYOUT_BLUEPRINTS.items():
            for i, box in enumerate(bp["boxes"]):
                assert 0.0 <= box["xmin"] <= 1.0, f"{name} box {i}: xmin={box['xmin']}"
                assert 0.0 <= box["ymin"] <= 1.0, f"{name} box {i}: ymin={box['ymin']}"
                assert 0.0 <= box["xmax"] <= 1.0, f"{name} box {i}: xmax={box['xmax']}"
                assert 0.0 <= box["ymax"] <= 1.0, f"{name} box {i}: ymax={box['ymax']}"

    def test_boxes_have_positive_area(self):
        for name, bp in DEFAULT_LAYOUT_BLUEPRINTS.items():
            for i, box in enumerate(bp["boxes"]):
                w = box["xmax"] - box["xmin"]
                h = box["ymax"] - box["ymin"]
                assert w > 0, f"{name} box {i}: zero width"
                assert h > 0, f"{name} box {i}: zero height"

    def test_entity_count_matches_boxes(self):
        for name, bp in DEFAULT_LAYOUT_BLUEPRINTS.items():
            assert bp["entity_count"] == len(bp["boxes"]), \
                f"{name}: entity_count={bp['entity_count']} != {len(bp['boxes'])} boxes"
