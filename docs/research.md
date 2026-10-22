# FrameGenX Research Primer

> The measured table below is a historical snapshot from an earlier code
> revision. Current acceptance checks object counts and supported spatial
> relations; it does not establish every attribute in the contract. See
> `README.md` and a fresh benchmark before making a current CSR claim.

## The Problem

Text-to-image models interpret prompts probabilistically. When given:

> "A red car parked beside a blue house with two people standing in front."

The model might produce:
- One person instead of two
- A blue car instead of red
- Car in front of house instead of beside it
- Missing house entirely

The model has no strict concept of:
```
car.color = red
person.count = 2
car.relation = beside(house)
```

## The Hypothesis

> Explicit contracts, scene planning, structural conditioning, and post-generation
> verification can improve **measurable constraint satisfaction** in text-to-image generation.

This is a testable research question. The success metric is **Constraint Satisfaction Rate (CSR)** — the percentage of generated images that satisfy all specified constraints.

## What FrameGenX Is

FrameGenX inserts an engineering layer between human intent and image generation:

```
Normal:     Prompt → Generator → Image (hope for the best)

FrameGenX:  Prompt → Contract Extraction → Scene Planning →
            Generation → Verification → [Accept | Repair + Retry]
```

The key insight: **generation is a compilation problem, not a creative act**.
Contracts are source-of-truth. Prompts are implementation details.

## What FrameGenX Is NOT

- Not another image generator (it wraps existing generators as backends)
- Not another SDXL wrapper (it provides decoupled verification)
- Not claiming to "eliminate hallucinations" (it claims measurable reduction)

## Measured Results

Ablation study on 15-prompt balanced subset, RTX 3050 6GB, SDXL-Turbo 512px, seed 42:

| Config | CSR | CLIP_norm | VQA | Reward | Avg Attempts |
|--------|-----|-----------|-----|--------|-------------|
| Baseline | 53% | 0.522 | 0.860 | 0.691 | 2.40 |
| + RAG | 40% | 0.538 | 0.913 | 0.725 | 3.20 |
| + RAG + LoRA | 73% | 0.554 | 0.923 | 0.739 | 2.60 |
| Full (+ Reflect) | 33% | 0.601 | 0.917 | 0.759 | 2.53 |

### Honest Assessment

- **RAG + LoRA achieves 73% CSR** — the best configuration so far
- **Full config (with Reflect) drops to 33%** — the reflect stage is actively harming CSR, likely due to τ threshold being too aggressive (0.730). This is a known regression under active investigation.
- These are small-scale numbers (N=15). Statistical confidence requires larger benchmarks.
- Reward increases monotonically (0.691 → 0.759), but CSR does not — demonstrating that reward ≠ constraint satisfaction.

### Caveats

1. **Sample size**: 15 prompts is too small for strong claims. Expansion to 60+ prompts is planned.
2. **Single seed**: Seed 42 only. Multi-seed runs needed for confidence intervals.
3. **Category coverage**: Primarily object_count and color_constraints. Spatial relations and text rendering are under-tested.

## Research Design Rules

1. **Reproducibility over cleverness** — fixed seeds, deterministic outputs, logged artifacts
2. **Benchmarks before claims** — no feature ships without CSR measurement
3. **Modular systems over notebooks** — stages are independent, testable units
4. **Verification before optimization** — a fast wrong answer is still wrong
5. **Deterministic outputs whenever possible** — same input → same output

## Key Metrics

- **CSR (Constraint Satisfaction Rate)**: Primary metric. Percentage of images satisfying ALL contract constraints.
- **CLIP_norm**: Normalized cosine similarity between generated image and ORIGINAL user prompt.
- **VQA_score**: Mean confidence of BLIP-VQA on per-entity attribute questions.
- **Reward**: `R = 0.5 × CLIP_norm + 0.5 × VQA_score`
- **τ (Acceptance Threshold)**: Reward threshold for the reflect stage acceptance gate.

## Verification Principle

The generator and verifier must use **decoupled** model families:
- Generator: Diffusion-based (SDXL-Turbo, score matching loss)
- Verifier 1: Detection-based (GroundingDINO-tiny, box regression loss)
- Verifier 2: VLM-based (LLM-vision VQA via router — Gemini/NIM; originally local BLIP-VQA, pivoted 2026-08: frontier-grade answers, 0 VRAM, still a different architecture family and training objective than the generator)
- Verifier 3: Contrastive (CLIP, cosine similarity loss)

If the verifier shares the same training data and loss function as the generator, it will confirm the generator's biases rather than catching them.

### Provenance note (2026-08)
The ablation table above was measured BEFORE the 2026-08-24 quality
revisions (prompt-anchored composition replacing entity-rebuilt prompts,
uncounted scenery entities, dense-confirmed retrieval). Those changes
primarily affect scenic prompts, which the object-count-centric benchmark
under-represents — expect the next benchmark run to shift numbers. Treat
the table as the historical baseline; re-measure before comparing.
