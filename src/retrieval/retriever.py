"""
RetrievalStage — Multi-Modal Visual Knowledge Retriever (NVIDIA NeMo RAG-inspired).

Provides:
1. Hybrid Dense + Sparse BM25 Semantic Search
2. Spatial Layout Blueprint Retrieval
3. Style & Lighting Preset Retrieval
4. Visual Concept & Affordance Grounding
5. Failure Memory & Corrective Heuristics
"""
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.core.stage import Stage, register_stage
from src.core.context import RunContext
from src.retrieval.milvus_client import open_local_milvus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())


class Retriever:
    """
    Core Dense Retriever using ChromaDB and sentence-transformers.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        r_cfg = config.get("retrieval", {}) or {}
        self.index_dir = PROJECT_ROOT / r_cfg.get("index_dir", "data/index")
        self.top_k = int(r_cfg.get("top_k", 3))

        self.milvus_path = self.index_dir / "milvus.db"
        self.collection_name = "FrameGenX_RAG"

        self._client = None

    def index_available(self) -> bool:
        return self.milvus_path.exists()

    def _load_index(self):
        if self._client is None and self.milvus_path.exists():
            from sentence_transformers import SentenceTransformer
            from fastembed import SparseTextEmbedding

            self._client = open_local_milvus(self.milvus_path)
            self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def unload(self):
        client, self._client = self._client, None
        try:
            if client is not None:
                client.close()
        finally:
            self._dense_model = None
            self._sparse_model = None

    def query(self, text: str) -> List[Dict[str, Any]]:
        self._load_index()
        if self._client is None:
            return []

        from pymilvus import AnnSearchRequest, RRFRanker

        self._client.load_collection(self.collection_name)

        dense = self._dense_model.encode([text], convert_to_numpy=True).tolist()[0]
        sparse_gen = list(self._sparse_model.embed([text]))[0]
        sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_gen.indices.tolist(), sparse_gen.values.tolist())}

        req_dense = AnnSearchRequest(
            data=[dense],
            anns_field="dense",
            param={"metric_type": "IP"},
            limit=self.top_k * 2
        )
        req_sparse = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=self.top_k * 2
        )

        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=RRFRanker(),
            limit=self.top_k,
            output_fields=["caption", "image", "original_id"]
        )

        ret = []
        if not results or not len(results):
            return ret

        for hit in results[0]:
            doc = hit["entity"].get("caption", "")
            meta = hit["entity"]

            ret.append({
                "id": str(hit["id"]),
                "caption": doc,
                "image": meta.get("image", ""),
                "score": float(hit["distance"]),
                "retrieval_mode": "hybrid_milvus_rrf"
            })
        return ret



class LayoutBlueprintRetriever:
    """Retrieves 2D spatial layout blueprints matching entity counts and relationships."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.blueprints = {}
        self._load()

    def _load(self):
        bp_path = self.index_dir / "layout_blueprints.json"
        if bp_path.exists():
            with open(bp_path, "r", encoding="utf-8") as f:
                self.blueprints = json.load(f)

    def retrieve(self, entity_count: int, relationships: List[Dict[str, Any]] = None,
                 entities: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        relationships = relationships or []
        # Check specific relationship-driven layouts first
        support = next((r for r in relationships if r.get("predicate") in
                        {"on", "atop", "standing_on", "placed_on", "on_top_of"}), None)

        if entity_count == 2 and support and "2_entities_subject_table" in self.blueprints:
            blueprint = deepcopy(self.blueprints["2_entities_subject_table"])
            # This catalog stores the support surface first and subject
            # second; bind boxes to entities instead of relying on noun order.
            if entities and len(entities) == 2 and len(blueprint.get("boxes", [])) == 2:
                subject = str(support.get("subject", "")).lower()
                obj = str(support.get("object", "")).lower()
                names = [str(e.get("class_name", "")).lower() for e in entities]
                if subject != obj and set(names) == {subject, obj}:
                    surface_box, subject_box = blueprint["boxes"]
                    blueprint["boxes"] = [subject_box if name == subject else surface_box
                                          for name in names]
                    blueprint["entity_bindings"] = [e.get("id", e.get("class_name"))
                                                     for e in entities]
            return blueprint

        # Otherwise match by entity count
        for key, bp in self.blueprints.items():
            if bp.get("entity_count") == entity_count:
                return bp

        # Fallback to closest available blueprint
        if self.blueprints:
            sorted_bps = sorted(self.blueprints.values(), key=lambda x: abs(x.get("entity_count", 1) - entity_count))
            return sorted_bps[0]
        return None


class StylePresetRetriever:
    """Retrieves style, lighting, and camera presets."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.presets = {}
        self._load()

    def _load(self):
        style_path = self.index_dir / "style_presets.json"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                self.presets = json.load(f)

    def retrieve(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        for name, preset in self.presets.items():
            if name in query_lower or any(w in query_lower for w in name.split("_")):
                return {"preset_name": name, **preset}
        # Default to photorealistic
        return {"preset_name": "photorealistic", **self.presets.get("photorealistic", {})}


class MultiModalKnowledgeRetriever:
    """
    Unified Façade for all Visual Knowledge Retrieval facets.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        r_cfg = config.get("retrieval", {}) or {}
        self.index_dir = PROJECT_ROOT / r_cfg.get("index_dir", "data/index")
        self.top_k = int(r_cfg.get("top_k", 3))

        self.retriever = Retriever(config)
        self.layout_retriever = LayoutBlueprintRetriever(self.index_dir)
        self.style_retriever = StylePresetRetriever(self.index_dir)

    def is_available(self) -> bool:
        return self.retriever.index_available()

    def unload(self):
        self.retriever.unload()

    def query(self, query: str, entity_count: int = 1,
              relationships: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes multi-modal query across captions, layout blueprints, and style presets.
        """
        caption_hits = self.retriever.query(query)
        layout_bp = self.layout_retriever.retrieve(entity_count, relationships)
        style_preset = self.style_retriever.retrieve(query)

        return {
            "caption_hits": caption_hits,
            "layout_blueprint": layout_bp,
            "style_preset": style_preset,
        }


from src.core.llm_router import LLMRouter
from src.retrieval.agent import RAGAgent

@register_stage
class RetrievalStage(Stage):
    name = "retrieval"

    def __init__(self, config):
        super().__init__(config)
        self.retriever = MultiModalKnowledgeRetriever(config)
        self.router = LLMRouter()
        self.rag_agent = None

    def run(self, ctx: RunContext) -> None:
        if not self.retriever.is_available():
            print(f"[retrieval] WARNING: multi-modal index missing at {self.retriever.index_dir} — no-op skip")
            return

        if self.rag_agent is None:
            self.rag_agent = RAGAgent(self.retriever.retriever, self.router)

        entities = ctx.contract.get("entities", [])
        counted_entities = [e for e in entities if not e.get("uncounted")]
        entity_count = len(counted_entities)
        relationships = ctx.contract.get("spatial_relationships", [])

        print(f"[retrieval] executing LangGraph Agent CRAG retrieval (entities={entity_count})...")

        # 1. Advanced LangGraph CRAG Agent for caption hits (with Web Search Fallback)
        try:
            caption_hits = self.rag_agent.invoke(ctx.prompt)
        finally:
            self.retriever.unload()

        # 2. Layout and Style retrieval (still synchronous/rule-based)
        layout_bp = self.retriever.layout_retriever.retrieve(
            entity_count, relationships, entities=counted_entities,
        ) if entity_count >= 2 else None
        style_preset = self.retriever.style_retriever.retrieve(ctx.prompt)

        knowledge = {
            "caption_hits": caption_hits,
            "layout_blueprint": layout_bp,
            "style_preset": style_preset,
        }

        hits = knowledge["caption_hits"]
        captions = [h["caption"] for h in hits]
        ctx.contract["rag_hits"] = captions
        ctx.ref_image_paths = [h.get("image", "") for h in hits]
        ctx.contract["retrieved_layout"] = knowledge.get("layout_blueprint")
        ctx.contract["retrieved_style"] = knowledge.get("style_preset")

        # Grounding influences
        ctx.contract["rag_influence"] = {
            "depth_required": any("depth" in c.lower() for c in captions),
            "lighting_complex": any("light" in c.lower() for c in captions),
            "layout_preset_applied": knowledge.get("layout_blueprint") is not None,
            "style_preset_applied": knowledge.get("style_preset", {}).get("preset_name"),
        }

        ep = ctx.contract.get("execution_plan", {})
        ep["has_depth"] = ctx.contract["rag_influence"]["depth_required"]
        ep["risk_level"] = "high" if len(captions) > 2 else "medium"
        ctx.contract["execution_plan"] = ep

        # --- SRRL Memory Retrieval ---
        try:
            from src.retrieval.memory import FailureMemoryDB
            mem_db = FailureMemoryDB()
            try:
                warnings = mem_db.retrieve_warnings(ctx.prompt, top_k=2)
            finally:
                mem_db.unload()
            if warnings:
                ctx.contract["retrieved_warnings"] = warnings
                print(f"[retrieval] Found {len(warnings)} past failure warnings for similar prompts!")
        except Exception as e:
            print(f"[retrieval] Warning: Failed to retrieve SRRL warnings: {e}")
        # -----------------------------

        ctx.save_artifact("rag_hits.json", hits)
        ctx.save_artifact("retrieved_knowledge.json", knowledge)
        print(f"[retrieval] {len(captions)} hybrid hits; blueprint: {(knowledge.get('layout_blueprint') or {}).get('name', 'none')}")
