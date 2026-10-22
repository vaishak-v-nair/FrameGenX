# FrameGenX — Measured Ablation Results

> Historical snapshot. These rows predate explicit benchmark provenance and
> adapter activation records. The percentages below are the old hard-check
> pass rates, not verified satisfaction of every contract constraint. They
> should not be used as a current benchmark claim.

Measured on RTX 3050 Laptop 6GB, SDXL-Turbo 512px, 15-prompt balanced subset
(from 30 curated scenes in `benchmark_dataset.json`; one variant per scene), fixed seed 42.
Scoring verifiers (CLIP, VQA) run in every config; `reflect` toggles only
reward gating + corrective prompts. CLIP scored against the ORIGINAL user
prompt. These are honest small-scale numbers — NOT the values reported in
the published paper (see README). +LoRA rows run the pixel-art style
adapter (nerijs/pixel-art-xl) — a visible style shift is expected and
intended; it demonstrates plug-and-play adaptation, not quality gain.

| Config | N | Historical hard-check pass | CLIP_norm | VQA | Reward | Avg attempts | Avg latency (s) |
|---|---|---|---|---|---|---|---|
| Baseline (SDXL-Turbo + count-repair) | 15 | 53% | 0.522 | 0.860 | 0.691 | 2.40 | 197.5 |
| + Retrieval (RAG) | 15 | 40% | 0.538 | 0.913 | 0.725 | 3.20 | 277.2 |
| + RAG + LoRA | 15 | 73% | 0.554 | 0.923 | 0.739 | 2.60 | 239.3 |
| Full (RAG + LoRA + Reflect) | 15 | 33% | 0.601 | 0.917 | 0.759 | 2.53 | 229.0 |

Paper-reported values (different system scale, different eval protocol —
reported for transparency, not comparison): FID 7.2, CLIP 0.63, VQA 0.68,
81% human preference.
