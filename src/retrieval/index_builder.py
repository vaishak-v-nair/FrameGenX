"""
Multi-Modal Visual Knowledge Index Builder (NVIDIA NeMo RAG-inspired).

Builds structured visual knowledge indices:
1. Dense Semantic Index (CLIP Embeddings)
2. Sparse Lexical Index (BM25 Inverted Term-Frequency)
3. Spatial Layout Blueprint Catalog (Normalized Bounding Boxes & Anchors)
4. Style & Aesthetic Preset Index (Lighting, Camera, Lenses, Palettes)
5. Visual Concept & Affordance Catalog (Scale Priors, Object Attributes)
6. Failure Memory & Repair Heuristics Catalog
"""
import io
import json
import math
import re
import urllib.request
import zipfile
from contextlib import closing
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from src.retrieval.milvus_client import open_local_milvus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# ============================================================================
# 1. Canonical Layout Blueprints
# ============================================================================
DEFAULT_LAYOUT_BLUEPRINTS = {
    "1_entity_centered": {
        "entity_count": 1,
        "name": "Single Subject Centered",
        "description": "Dominant single foreground subject centered in frame",
        "boxes": [
            {"anchor": "center", "ymin": 0.15, "xmin": 0.15, "ymax": 0.85, "xmax": 0.85, "z_index": 1}
        ],
    },
    "2_entities_side_by_side": {
        "entity_count": 2,
        "name": "Dual Horizontal Split",
        "description": "Two distinct entities placed side-by-side with clear separation",
        "boxes": [
            {"anchor": "center_left", "ymin": 0.20, "xmin": 0.08, "ymax": 0.80, "xmax": 0.46, "z_index": 1},
            {"anchor": "center_right", "ymin": 0.20, "xmin": 0.54, "ymax": 0.80, "xmax": 0.92, "z_index": 1}
        ],
    },
    "2_entities_subject_table": {
        "entity_count": 2,
        "name": "Subject on Supporting Surface",
        "description": "Primary subject resting on a supporting table/ground surface",
        "boxes": [
            {"anchor": "bottom_center", "ymin": 0.45, "xmin": 0.0, "ymax": 1.0, "xmax": 1.0, "z_index": 0},
            {"anchor": "center", "ymin": 0.25, "xmin": 0.25, "ymax": 0.65, "xmax": 0.75, "z_index": 1}
        ],
    },
    "3_entities_horizontal_trio": {
        "entity_count": 3,
        "name": "Horizontal Trio",
        "description": "Three subjects aligned horizontally with even spacing",
        "boxes": [
            {"anchor": "left", "ymin": 0.25, "xmin": 0.05, "ymax": 0.75, "xmax": 0.32, "z_index": 1},
            {"anchor": "center", "ymin": 0.20, "xmin": 0.36, "ymax": 0.80, "xmax": 0.64, "z_index": 1},
            {"anchor": "right", "ymin": 0.25, "xmin": 0.68, "ymax": 0.75, "xmax": 0.95, "z_index": 1}
        ],
    },
    "3_entities_triangular_cluster": {
        "entity_count": 3,
        "name": "Triangular Composition",
        "description": "One dominant top/center subject flanked by two lower supporting subjects",
        "boxes": [
            {"anchor": "top_center", "ymin": 0.12, "xmin": 0.30, "ymax": 0.52, "xmax": 0.70, "z_index": 1},
            {"anchor": "bottom_left", "ymin": 0.50, "xmin": 0.08, "ymax": 0.88, "xmax": 0.46, "z_index": 1},
            {"anchor": "bottom_right", "ymin": 0.50, "xmin": 0.54, "ymax": 0.88, "xmax": 0.92, "z_index": 1}
        ],
    },
    "4_entities_quad_grid": {
        "entity_count": 4,
        "name": "2x2 Quad Grid",
        "description": "Four balanced quadrants for high-density multi-object scenes",
        "boxes": [
            {"anchor": "top_left", "ymin": 0.08, "xmin": 0.08, "ymax": 0.46, "xmax": 0.46, "z_index": 1},
            {"anchor": "top_right", "ymin": 0.08, "xmin": 0.54, "ymax": 0.46, "xmax": 0.92, "z_index": 1},
            {"anchor": "bottom_left", "ymin": 0.54, "xmin": 0.08, "ymax": 0.92, "xmax": 0.46, "z_index": 1},
            {"anchor": "bottom_right", "ymin": 0.54, "xmin": 0.54, "ymax": 0.92, "xmax": 0.92, "z_index": 1}
        ],
    },
    "5_entities_crowded_cluster": {
        "entity_count": 5,
        "name": "Five Object Dynamic Array",
        "description": "Center focal entity surrounded by 4 satellite objects",
        "boxes": [
            {"anchor": "center", "ymin": 0.30, "xmin": 0.30, "ymax": 0.70, "xmax": 0.70, "z_index": 2},
            {"anchor": "top_left", "ymin": 0.06, "xmin": 0.06, "ymax": 0.38, "xmax": 0.38, "z_index": 1},
            {"anchor": "top_right", "ymin": 0.06, "xmin": 0.62, "ymax": 0.38, "xmax": 0.94, "z_index": 1},
            {"anchor": "bottom_left", "ymin": 0.62, "xmin": 0.06, "ymax": 0.94, "xmax": 0.38, "z_index": 1},
            {"anchor": "bottom_right", "ymin": 0.62, "xmin": 0.62, "ymax": 0.94, "xmax": 0.94, "z_index": 1}
        ],
    }
}


# ============================================================================
# 2. Style & Lighting Presets Catalog
# ============================================================================
DEFAULT_STYLE_PRESETS = {
    "photorealistic": {
        "descriptors": "photograph, 35mm lens, sharp focus, natural lighting, subsurface scattering, authentic textures",
        "lighting": "soft directional daylight with realistic ambient bounce",
        "camera": "eye-level 50mm f/2.8 lens, shallow depth of field",
        "negative": "cartoon, illustration, 3d render, distorted, oversaturated, plastic"
    },
    "cinematic": {
        "descriptors": "anamorphic widescreen, cinematic lighting, rim lighting, atmospheric volumetric haze, 8k resolution",
        "lighting": "dramatic chiaroscuro lighting with warm key and cool fill",
        "camera": "panavision 70mm lens, cinematic color grading, teal and orange",
        "negative": "flat lighting, amateur snapshot, oversaturated, watermark"
    },
    "studio_product": {
        "descriptors": "commercial studio photography, clean neutral backdrop, commercial lighting, crisp highlights",
        "lighting": "three-point softbox studio lighting, diffused shadows",
        "camera": "macro 85mm lens, crisp product focus, f/8 aperture",
        "negative": "blurry, dark, noisy, cluttered background, distorted geometry"
    },
    "pixel_art": {
        "descriptors": "pixel art, 16-bit retro game aesthetic, sharp pixel contours, vibrant restricted color palette",
        "lighting": "dithered shading, clean pixel highlights",
        "camera": "orthographic retro perspective",
        "negative": "photorealistic, smooth gradient, 3d render, blurry, high poly"
    },
    "isometric_3d": {
        "descriptors": "isometric 3D render, blender 3D, octane render, clean ambient occlusion, clay/matte material",
        "lighting": "soft global illumination, smooth soft shadows",
        "camera": "orthographic isometric camera angle at 30 degrees",
        "negative": "perspective distortion, 2D flat, messy, low resolution"
    },
    "cyberpunk_neon": {
        "descriptors": "cyberpunk aesthetic, vibrant neon reflections, rain-slicked surfaces, high contrast glowing lights",
        "lighting": "vivid cyan and magenta neon rim lights against deep shadows",
        "camera": "low-angle dynamic street photography",
        "negative": "sunlight, pastel colors, monochrome, washed out"
    }
}


# ============================================================================
# 3. Visual Concept & Affordance Catalog
# ============================================================================
DEFAULT_CONCEPT_CATALOG = {
    "furniture": {
        "classes": ["table", "chair", "sofa", "desk", "bed", "bench"],
        "is_supporting_surface": True,
        "default_layer": "background_or_midground",
        "typical_scale": "large"
    },
    "fruits_and_food": {
        "classes": ["apple", "banana", "orange", "cake", "sandwich", "pizza", "bread"],
        "is_supporting_surface": False,
        "typical_host": "table",
        "typical_scale": "small",
        "common_attributes": ["fresh", "glossy", "ripe", "colorful"]
    },
    "utensils_and_tableware": {
        "classes": ["fork", "knife", "spoon", "cup", "bottle", "plate", "glass", "bowl"],
        "is_supporting_surface": False,
        "typical_host": "table",
        "typical_scale": "small",
        "common_attributes": ["metallic", "polished", "ceramic", "transparent"]
    },
    "vehicles": {
        "classes": ["car", "bicycle", "motorcycle", "bus", "truck", "boat", "airplane"],
        "is_supporting_surface": False,
        "typical_host": "street_or_water",
        "typical_scale": "large"
    },
    "animals_and_pets": {
        "classes": ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant"],
        "is_supporting_surface": False,
        "typical_scale": "medium",
        "common_attributes": ["furry", "alert", "sitting", "standing"]
    }
}


# ============================================================================
# 4. Failure Memory & Repair Knowledge Base
# ============================================================================
DEFAULT_FAILURE_MEMORY = {
    "count_under_generation": {
        "trigger": "detected_count < expected_count",
        "repair_strategy": "Increase spatial bounding box separation, enforce explicit per-item numbering in prompt ('item 1, item 2, item 3 each completely separate'), and increase ControlNet edge conditioning weight."
    },
    "count_over_generation": {
        "trigger": "detected_count > expected_count",
        "repair_strategy": "Inject negative prompt 'duplicate, clones, extra objects, cluttered background' and decrease density in bounding box mask."
    },
    "attribute_leakage": {
        "trigger": "attribute_bleeding_detected",
        "repair_strategy": "Isolate noun-adjective pairs into distinct regional tokens, insert explicit contrast descriptors ('a red apple strictly next to a green bottle, no color mixing')."
    },
    "spatial_inversion": {
        "trigger": "spatial_relation_inverted",
        "repair_strategy": "Reinforce asymmetric directional cues ('on the extreme far left side', 'placed on the upper horizontal surface')."
    }
}


# BM25 Sparse Lexical Index Builder removed in favor of ChromaDB semantic search.


# ============================================================================
# 6. Master Indexing Pipeline Orchestrator
# ============================================================================
class VisualKnowledgeIndexBuilder:
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (PROJECT_ROOT / "data" / "index")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_coco_captions(self, n: int = 1000) -> List[Dict[str, str]]:
        cache = PROJECT_ROOT / "data" / "captions_val2017.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            print(f"[indexer] Downloading COCO annotations ({ANN_URL})...")
            buf = io.BytesIO(urllib.request.urlopen(ANN_URL).read())
            with zipfile.ZipFile(buf) as z:
                with z.open("annotations/captions_val2017.json") as f:
                    cache.write_bytes(f.read())
            print(f"[indexer] Cached annotations -> {cache}")

        with open(cache, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_urls = {img["id"]: img.get("coco_url", "") for img in data.get("images", [])}
        records = []
        seen_imgs = set()
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in seen_imgs:
                continue
            seen_imgs.add(img_id)
            records.append({
                "caption": ann["caption"].strip(),
                "image": img_urls.get(img_id, ""),
                "id": f"coco_{img_id}"
            })
            if len(records) >= n:
                break
        return records

    def build_dense_embeddings(self, records: List[Dict[str, str]], batch_size: int = 128):
        from pymilvus import MilvusClient, DataType
        from sentence_transformers import SentenceTransformer
        from fastembed import SparseTextEmbedding
        
        milvus_path = self.output_dir / "milvus.db"
        print(f"[indexer] Initializing Milvus Hybrid Index at {milvus_path}...")
        dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        sparse_model = SparseTextEmbedding("Qdrant/bm25")
        with closing(open_local_milvus(milvus_path)) as client:
            collection_name = "FrameGenX_RAG"
            if client.has_collection(collection_name=collection_name):
                client.drop_collection(collection_name=collection_name)

            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=384)
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="caption", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="image", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="original_id", datatype=DataType.VARCHAR, max_length=128)

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
            index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

            for i in range(0, len(records), batch_size):
                batch_records = records[i:i + batch_size]
                batch_texts = [r["caption"] for r in batch_records]

                dense_embs = dense_model.encode(batch_texts, convert_to_numpy=True).tolist()
                sparse_embs = list(sparse_model.embed(batch_texts))

                data = []
                for j, (r, dense, sparse) in enumerate(zip(batch_records, dense_embs, sparse_embs)):
                    sparse_dict = {int(idx): float(val) for idx, val in zip(sparse.indices.tolist(), sparse.values.tolist())}
                    data.append({
                        "id": i + j,
                        "dense": dense,
                        "sparse": sparse_dict,
                        "caption": r["caption"],
                        "image": r["image"],
                        "original_id": r["id"]
                    })

                client.insert(collection_name=collection_name, data=data)
                print(f"  hybrid embedded {min(i + batch_size, len(records))}/{len(records)}", end="\r")

        print(f"\n[indexer] Hybrid index created in Milvus -> {milvus_path}")

    def build_layout_blueprints(self):
        print("[indexer] Saving spatial layout blueprints catalog...")
        with open(self.output_dir / "layout_blueprints.json", "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LAYOUT_BLUEPRINTS, f, indent=2)
        print(f"[indexer] Layout blueprints saved -> {self.output_dir / 'layout_blueprints.json'}")

    def build_style_presets(self):
        print("[indexer] Saving style presets catalog...")
        with open(self.output_dir / "style_presets.json", "w", encoding="utf-8") as f:
            json.dump(DEFAULT_STYLE_PRESETS, f, indent=2)
        print(f"[indexer] Style presets saved -> {self.output_dir / 'style_presets.json'}")

    def build_concept_catalog(self):
        print("[indexer] Saving visual concept catalog...")
        with open(self.output_dir / "concept_catalog.json", "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONCEPT_CATALOG, f, indent=2)
        print(f"[indexer] Concept catalog saved -> {self.output_dir / 'concept_catalog.json'}")

    def build_failure_memory(self):
        print("[indexer] Saving failure memory & repair heuristics catalog...")
        with open(self.output_dir / "failure_memory.json", "w", encoding="utf-8") as f:
            json.dump(DEFAULT_FAILURE_MEMORY, f, indent=2)
        print(f"[indexer] Failure memory saved -> {self.output_dir / 'failure_memory.json'}")

    def build_all(self, n: int = 1000):
        print(f"=== FrameGenX Multi-Modal Knowledge Indexer (Target: {n} documents) ===")
        records = self.fetch_coco_captions(n=n)
        self.build_dense_embeddings(records)
        self.build_layout_blueprints()
        self.build_style_presets()
        self.build_concept_catalog()
        self.build_failure_memory()
        print(f"=== Multi-Modal Visual Knowledge Base successfully built at {self.output_dir} ===")
