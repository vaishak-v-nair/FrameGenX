# FrameGenX Architecture

> **FrameGenX** is a contract-driven visual generation and verification system.
> It transforms natural language image requirements into measurable visual outputs.

## Core Principle

```
Generate → Verify → Repair → Accept
```

Verification is first-class. Generation alone is insufficient.

## Stage Graph

```
config.yaml: pipeline.stages: [intent, retrieval, plan, adapt, generate, verify, reflect]

 prompt ──> INTENT ──> RETRIEVAL ──┐  (executed once per run)
                                   ▼
        ┌────────── repair loop (attempt k in 0 .. max_repair_loops) ───────┐
        │  PLAN ──> ADAPT ──> GENERATE ──> VERIFY ───────────> REFLECT      │
        │  masks    LoRA      SDXL-Turbo   DINO count check   hard pass?    │
        │  prompt   inject    +ControlNet  CLIP score         corrective    │
        │  compose  (once)    512px, 4stp  vision QA probes  prompt edits ─┘
        └─────────────────────────────────────────────────────────│─────────
                                                                  ▼
                                            accepted artifact + audit receipt
```

### Stage Descriptions

| Stage | Module | Purpose |
|-------|--------|---------|
| **Intent** | `src/intent/extractor.py` | LLM contract extraction (Groq gpt-oss-120b → NIM → Gemini via `src/core/llm_router.py`) → ContractV2 (entities, counts, attributes, constraints) with deterministic backstops: uncountable-concept stoplist, bare-plural → `uncounted` (presence-only) entities, and a rule-based `repair_policy`. spaCy survives only in `src/intent/relationships.py` (spatial predicates) |
| **Retrieval** | `src/retrieval/retriever.py` | Hybrid Dense (CLIP) + Sparse (BM25) RRF search → layout blueprints, style presets, failure memory. Captions are **inspection-only** (UI evidence) — they are never fused into the drawing prompt (see Verification note below) |
| **Plan** | `src/scene/planner.py` | Copies contract entities into a separate scene plan; binds support layouts to subject/object roles; makes masks only for ≥2 counted entities; keeps the original prompt with corrective edits prepended |
| **Adapt** | `src/generation/lora.py` | Dynamic LoRA loading (plug-and-play style adaptation, `enabled` flag per adapter) |
| **Generate** | `src/generation/engine.py` | SDXL-Turbo with regional attention, optional ControlNet, 512px/4 steps/cfg 0; sequential CPU offload; cached checkpoint resolution supports offline loading |
| **Verify** | `src/verification/verify_stage.py` | GroundingDINO-tiny counting + CLIP alignment + LLM-vision VQA + CPU spatial geometry + optional diagnostic plugins |
| **Reflect** | `src/reflection/reflector.py` | `hard_only` acceptance (zero count/spatial failures) + rule-based corrective edit synthesis (τ gating retired — tau_sweep proved it net-harmful) |
| **Refine** | `src/generation/refine_stage.py` | Disabled by default; opt-in texture polish reuses the generate pipe where possible; the derivative is explicitly unverified |

### Why captions are no longer fused into prompts
Measured CLIP sentence-to-sentence similarity cannot discriminate caption
relevance (any two English sentences score ~0.27–0.48; "city skyline and
train yard" outscored "field of wildflowers" against an alpine query).
Fusing unvalidated COCO captions *injected* hallucinations — the model drew
a metal structure a retrieved caption described. Retrieval still grounds
generation through curated layout blueprints and style presets; caption
hits remain visible as glass-box evidence (UI Step 2, `rag_hits.json`).

## Zero-Concurrency Memory Lifecycle (6GB VRAM Invariant)

Only one model group occupies GPU memory at any time:

```
[Time]    [Active Stage]            [Device]    [VRAM]       [RAM]
T0        LLM Intent Extraction     API         0 MB         ~400 MB
T1        Hybrid Dense+BM25 Search  CPU         0 MB         ~600 MB
          (CLIP text singleton,
          CPU-resident, shared
          with verify)
T2        Layout & Mask Generation  CPU         0 MB         ~600 MB
T3        SDXL-Turbo (+ControlNet)  GPU         ~3,500 MB    ~3-4 GB
          sequential CPU offload:               (peaks)      (streamed)
          layers stream RAM->GPU
T4        free_memory()             CUDA Flush  < 100 MB     —
T5        GroundingDINO-tiny        GPU         ~1,800 MB    ~2,200 MB
          (fp32 CPU-resident
          between uses)
T6        CLIP score + LLM-vision   GPU/API     ~1,200 MB    ~2,500 MB
          VQA (router call, 0 VRAM              (CLIP
          for the LLM part)                     transient)
T7        Reward & Edits            CPU         0 MB         ~1,200 MB
T8        Refine (opt-in)           GPU         reuses T3 pipe
```

Measured offload policy (2026-08-24, RTX 3050 Laptop): **sequential** CPU
offload wins — the laptop GPU also drives the Windows display, leaving only
~4.6-5.1GB of the 6GB actually free, so model-offload's 4.9GB UNet spills
into driver-managed system memory and thrashes (70s/step vs 44s/step). The
CLIP singleton is evicted before pipe construction and reloads from OS page
cache in seconds (RAM budget discipline on the 16GB host / 8GB WSL VM).

## Verification Architecture

The verification suite uses **decoupled** discriminative models:

| Component | Architecture | Objective | Purpose |
|-----------|-------------|-----------|---------|
| Generator (SDXL-Turbo) | Diffusion UNet / DMD | Score Matching | Image synthesis |
| Verifier 1 (GroundingDINO-tiny) | Open-Set Detector | Box Regression | Object counting |
| Verifier 2 (LLM-vision VQA) | Frontier VLM via router (Gemini / NIM) | Autoregressive VQA | Attribute + existence probes |
| Verifier 3 (CLIP ViT-B/32) | Contrastive | Cosine Similarity | Semantic alignment |
| Verifier 4 (Spatial geometry) | Pure CPU box math | Deterministic predicates | Relational topology |

Note: BLIP-VQA (local) was the original design; VQA now rides the LLM
vision router — frontier-grade answers, 0 VRAM, no local model lifecycle.
Florence-2 was evaluated and retired (HF checkpoint incompatible with
transformers 5.x); its scene-comprehension role is covered by the optional
LLM-vision plugin (`verification.plugins.scene_comprehension`), which —
like the EasyOCR plugin — is a **diagnostic soft signal** (lands in
`verification_report.signals`, never hard-rejects, final attempt only).

### Composite Reward & Acceptance

```
R = 0.5 × CLIP_norm + 0.5 × VQA_score
Acceptance = (¬count_failures) ∧ (¬spatial_failures)      # mode: hard_only
```

VQA and reward are unavailable when answers are incomplete or providers fail.
`contract_evidence` records operational passes, failures, and unverified
requirements per attempt. Colors, negations, and ambiguous instance/depth
checks cannot become full-contract passes from count evidence alone. The
ledger covers extracted requirements; extraction recall and visual judges
still require calibration against human labels.

`mode: hard_only` is the default since tau_sweep.py (2026-08-15) proved
reward gating net-harmful: Hard-only CSR=57%, τ=0.730 CSR=38%. R is still
computed and logged every attempt — it feeds reporting and future
experiments, not the acceptance gate.

## Key Abstractions

### ContractV2 (`src/intent/models.py`)
The machine-readable specification extracted from natural language. Contains:
- **Entities**: objects with class names, counts, attributes, and positions
- **Constraints**: typed requirements (object_count, attribute_binding, etc.)
- **Execution Plan**: human detection, depth requirement, risk level

### RunContext (`src/core/context.py`)
The stateful artifact bag flowing through all stages. Contains:
- The contract, generated images, verification reports
- Timing, attempt counter, and the final accept/reject decision

### Pipeline (`src/core/stage.py`)
Config-driven stage executor with registry pattern. Stages are registered via
`@register_stage` decorator and executed in the order specified by `config.yaml`.

## Benchmark System

The ablation benchmark (`benchmark.py`) tests 4 configurations:
- **Baseline**: SDXL-Turbo with the same verifiers and bounded attempts; no retrieval, LoRA, or reflective prompt edits
- **+ RAG**: Adds retrieved layout/style and inspection evidence
- **+ RAG + LoRA**: Adds plug-and-play style adaptation
- **Full**: Adds reflection with hard-check acceptance and corrective edits

All runs are logged to SQLite (`runs/framegenx.db`) for longitudinal analysis.
Each new benchmark suite has a code/config/case/package fingerprint and saved
manifest. Resume and reports use one suite; historical unlabeled rows stay
archival. Floating model and adapter references still limit reproducibility.

## Resource Constraints

- **Primary hardware**: RTX 3050 Laptop GPU, 6GB VRAM, 16GB RAM
- **All architecture decisions must work within this constraint**
- Zero OOM errors is a hard requirement
