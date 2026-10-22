"""
Build the complete Multi-Modal Visual Knowledge Index (NVIDIA NeMo RAG-inspired).

Builds:
1. Dense Semantic Embeddings (CLIP ViT-B/32) -> data/index/embeddings.npy
2. Dense Records & Image URLs -> data/index/captions.json
3. Sparse BM25 Inverted Index & Term Statistics -> data/index/bm25_index.json
4. Spatial Layout Blueprints -> data/index/layout_blueprints.json
5. Style & Lighting Presets -> data/index/style_presets.json
6. Visual Concept & Affordance Catalog -> data/index/concept_catalog.json
7. Failure Memory & Repair Heuristics -> data/index/failure_memory.json

Usage: python scripts/build_index.py [--n 1000] [--batch 64]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.index_builder import VisualKnowledgeIndexBuilder


def main():
    parser = argparse.ArgumentParser(description="FrameGenX Multi-Modal Knowledge Index Builder")
    parser.add_argument("--n", type=int, default=1000, help="Number of COCO captions to index")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for dense embedding")
    args = parser.parse_args()

    builder = VisualKnowledgeIndexBuilder()
    builder.build_all(n=args.n)


if __name__ == "__main__":
    main()
