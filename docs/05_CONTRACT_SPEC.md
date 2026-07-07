# Contract Specification

## 1. Prompt-Driven vs. Contract-Driven Generation
The core differentiator of FrameGenX is its abandonment of Prompt-Driven Generation.

**Prompt-Driven Generation**: A user writes a paragraph describing a scene. The text encoder compresses this into embeddings, inevitably losing spatial logic, counts, and attribute isolation. If the diffusion model hallucinates, the only recourse is to "prompt engineer"—adding words like "highly detailed," "exactly three," or "4k" in a desperate attempt to steer the probability distribution.

**Contract-Driven Generation**: The Contract is the single source of truth in the FrameGenX architecture. It is a strictly typed JSON schema that represents the user's intent mathematically and logically, completely stripping away the ambiguity of natural language. The diffusion model is not allowed to "guess" the layout; the layout is dictated by the contract. What becomes impossible without a Contract is **deterministic verifiability**—an autonomous agent cannot verify a "vibe," but it can mathematically verify a JSON constraint.

## 2. Core Schema Structure (v2)
The `contract_v2.json` schema is divided into the following primary domains:

```json
{
  "metadata": {
    "version": "2.0",
    "prompt_id": "uuid-1234"
  },
  "global_parameters": {
    "style": "photorealistic",
    "lighting": "golden hour",
    "aspect_ratio": "16:9"
  },
  "entities": [
    {
      "id": "entity_1",
      "class_name": "car",
      "attributes": ["red", "parked"],
      "bounding_box": [100, 200, 300, 400], 
      "z_index": 1
    }
  ],
  "relationships": []
}
```

## 3. Constraints and Validation Rules
- **Entity Uniqueness**: Every entity must possess a unique UUID. This prevents the generation engine from conflating two similar objects.
- **Spatial Resolution**: Bounding boxes must mathematically validate spatial relationships before generation begins.
- **Attribute Isolation**: Attributes are strictly bound to their parent entity's ID.

If a constraint is not in the contract, it does not exist. If the generated image violates the contract, the image is rejected.
