# 05_CONTRACT_SPEC.md — Formal Contract Specification (v2.1)

> **Design target, not the current runtime schema.** The Pydantic model in
> `src/intent/models.py` is smaller and does not implement the JSON Schema
> below. Runtime extraction also appends spatial relationships and negations
> after model validation. Use recorded `contract_v2.json` artifacts and the
> runtime model when checking current behavior.

---

## 1. Philosophical Foundation: Why Natural Language Fails as a Contract

In traditional computer science, no compiler allows source code to be specified as an ambiguous English paragraph. Compilers require a formal grammar (EBNF) that resolves deterministically to an Abstract Syntax Tree (AST).

Generative text-to-image systems historically bypassed formal intermediate representations, attempting to map continuous natural language strings directly to high-dimensional latent distributions. This causes:
1. **Syntactic Ambiguity**: `"Two red cars and bikes"` $\to$ Are the bikes red? Are there two bikes or two cars?
2. **Spatial Indeterminacy**: `"A cat next to a dog"` $\to$ Does "next to" imply left, right, foreground, background, or overlapping?
3. **Unfalsifiable Specifications**: An automated test suite cannot evaluate whether an image satisfies `"a vibrant cinematic atmosphere"`.

**The Contract is the single source of truth in FrameGenX.** The Contract decouples human/agent intent from the underlying rendering engine, enabling exact mathematical verification.

---

## 2. Complete Formal JSON Schema (ContractV2)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "FrameGenXContractV2",
  "type": "object",
  "required": ["metadata", "global_canvas", "entities", "verification_rules"],
  "properties": {
    "metadata": {
      "type": "object",
      "required": ["version", "contract_id", "timestamp", "source_prompt"],
      "properties": {
        "version": { "type": "string", "enum": ["2.0", "2.1"] },
        "contract_id": { "type": "string", "format": "uuid" },
        "timestamp": { "type": "string", "format": "date-time" },
        "source_prompt": { "type": "string" },
        "author": { "type": "string" }
      }
    },
    "global_canvas": {
      "type": "object",
      "required": ["width", "height", "aspect_ratio"],
      "properties": {
        "width": { "type": "integer", "default": 512, "minimum": 256, "maximum": 2048 },
        "height": { "type": "integer", "default": 512, "minimum": 256, "maximum": 2048 },
        "aspect_ratio": { "type": "string", "enum": ["1:1", "16:9", "4:3", "9:16", "21:9"] },
        "style_preset": { "type": "string", "default": "photorealistic" },
        "lighting_environment": { "type": "string" },
        "background_description": { "type": "string" },
        "negative_prompt": { "type": "string" }
      }
    },
    "entities": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["id", "class_name", "count", "attributes", "spatial_constraint"],
        "properties": {
          "id": { "type": "string", "pattern": "^ent_[a-z0-9_]+$" },
          "class_name": { "type": "string", "description": "Normalized noun for zero-shot detection" },
          "count": { "type": "integer", "minimum": 1, "maximum": 50 },
          "attributes": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Adjectival descriptors bound exclusively to this entity"
          },
          "spatial_constraint": {
            "type": "object",
            "required": ["bounding_box", "z_index"],
            "properties": {
              "bounding_box": {
                "type": "array",
                "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                "minItems": 4,
                "maxItems": 4,
                "description": "[ymin, xmin, ymax, xmax] normalized coordinates"
              },
              "z_index": { 
                "type": "integer", 
                "default": 0, 
                "description": "Depth layer: higher values are closer to foreground" 
              },
              "anchor": {
                "type": "string",
                "enum": ["top_left", "top_center", "top_right", "center_left", "center", "center_right", "bottom_left", "bottom_center", "bottom_right"]
              }
            }
          },
          "conditioning_mask": {
            "type": "object",
            "properties": {
              "mask_type": { "type": "string", "enum": ["box_mask", "canny_edge", "depth_map", "openpose"] },
              "mask_path": { "type": "string" }
            }
          }
        }
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["subject_id", "predicate", "object_id"],
        "properties": {
          "subject_id": { "type": "string" },
          "predicate": {
            "type": "string",
            "enum": ["left_of", "right_of", "above", "below", "inside", "holding", "standing_on", "occluding", "adjacent_to"]
          },
          "object_id": { "type": "string" }
        }
      }
    },
    "verification_rules": {
      "type": "object",
      "required": ["hard_constraints", "soft_constraints", "thresholds"],
      "properties": {
        "hard_constraints": {
          "type": "array",
          "items": { "type": "string", "enum": ["entity_counts", "spatial_overlap", "forbidden_entities"] },
          "description": "Violations trigger immediate generation rejection"
        },
        "soft_constraints": {
          "type": "array",
          "items": { "type": "string", "enum": ["clip_semantic_alignment", "vqa_attribute_confidence", "aesthetic_score"] },
          "description": "Contribute continuously to composite reward R"
        },
        "thresholds": {
          "type": "object",
          "required": ["dino_confidence_threshold", "composite_reward_gate_tau"],
          "properties": {
            "dino_confidence_threshold": { "type": "number", "default": 0.35, "minimum": 0.1, "maximum": 0.9 },
            "clip_cosine_threshold": { "type": "number", "default": 0.25 },
            "vqa_confidence_threshold": { "type": "number", "default": 0.70 },
            "composite_reward_gate_tau": { "type": "number", "default": 0.70, "minimum": 0.0, "maximum": 1.0 }
          }
        }
      }
    }
  }
}
```

---

## 3. Concrete Contract Example (`contract_sample.json`)

```json
{
  "metadata": {
    "version": "2.1",
    "contract_id": "c98f4b62-7934-4b51-9e2c-38914b1d601a",
    "timestamp": "2026-08-14T12:00:00Z",
    "source_prompt": "Three red apples on a rustic wooden table with a silver fork to the left."
  },
  "global_canvas": {
    "width": 512,
    "height": 512,
    "aspect_ratio": "1:1",
    "style_preset": "photorealistic",
    "lighting_environment": "soft natural morning sunlight from right",
    "background_description": "warm neutral kitchen backdrop, shallow depth of field"
  },
  "entities": [
    {
      "id": "ent_table",
      "class_name": "wooden table",
      "count": 1,
      "attributes": ["rustic", "dark wood", "textured surface"],
      "spatial_constraint": {
        "bounding_box": [0.4, 0.0, 1.0, 1.0],
        "z_index": 0,
        "anchor": "bottom_center"
      }
    },
    {
      "id": "ent_apples",
      "class_name": "apple",
      "count": 3,
      "attributes": ["red", "glossy", "fresh", "unblemished"],
      "spatial_constraint": {
        "bounding_box": [0.35, 0.35, 0.75, 0.85],
        "z_index": 1,
        "anchor": "center"
      }
    },
    {
      "id": "ent_fork",
      "class_name": "fork",
      "count": 1,
      "attributes": ["silver", "metallic", "polished"],
      "spatial_constraint": {
        "bounding_box": [0.45, 0.1, 0.8, 0.25],
        "z_index": 1,
        "anchor": "center_left"
      }
    }
  ],
  "relationships": [
    { "subject_id": "ent_apples", "predicate": "standing_on", "object_id": "ent_table" },
    { "subject_id": "ent_fork", "predicate": "standing_on", "object_id": "ent_table" },
    { "subject_id": "ent_fork", "predicate": "left_of", "object_id": "ent_apples" }
  ],
  "verification_rules": {
    "hard_constraints": ["entity_counts"],
    "soft_constraints": ["clip_semantic_alignment", "vqa_attribute_confidence"],
    "thresholds": {
      "dino_confidence_threshold": 0.35,
      "clip_cosine_threshold": 0.26,
      "vqa_confidence_threshold": 0.75,
      "composite_reward_gate_tau": 0.72
    }
  }
}
```

---

## 4. Contract Validation & Integrity Rules

Prior to executing any generative inference, the Contract Parser executes the following deterministic validation passes:

```
                  CONTRACT INTEGRITY VALIDATION PIPELINE
                  
  [Raw Contract JSON]
          │
          ▼
  [Pass 1: Schema Conformance] ──> Validates JSON Schema draft 2020-12
          │
          ▼
  [Pass 2: Coordinate Bounds]  ──> Asserts 0.0 <= ymin < ymax <= 1.0 & 0.0 <= xmin < xmax <= 1.0
          │
          ▼
  [Pass 3: Cardinality Check]  ──> Asserts all entity.count >= 1 (integer)
          │
          ▼
  [Pass 4: Graph Integrity]    ──> Verifies subject_id & object_id exist in entities list;
          │                        detects and rejects circular spatial paradoxes (A left of B, B left of A)
          │
          ▼
  [Validated Contract State]   ──> Dispatched to Scene Planner
```

---

## 5. Versioning & Evolution Strategy

- **Major Version Bumps (`v2.0 -> v3.0`)**: Introduced when the spatial dimensionality changes (e.g., adding 3D camera matrices $[R | T]$, mesh references, or multi-frame temporal tracks).
- **Minor Version Bumps (`v2.0 -> v2.1`)**: Backward-compatible additions of new verification heuristics, new relationship predicates, or custom LoRA hooks.
- **Contract Hash / Immutability**: Each contract is assigned a deterministic SHA-256 hash computed over normalized JSON keys. This hash forms the canonical root of all downstream verification receipts.
