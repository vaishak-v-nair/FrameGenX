# 00_MASTER_INDEX.md — The FrameGenX Research & Architecture Compendium

```
========================================================================================
                              FRAMEGENX ARCHITECTURAL COMPENDIUM
              Contract-Driven Visual Generation and Verification Platform
========================================================================================
```

---

## Compendium Structure & Document Directory

This compendium represents the complete architectural and scientific specification for **FrameGenX**. It transitions visual AI generation from probabilistic prompt heuristics into a deterministic, verifiable, and benchmarked engineering discipline.

| Document | Title & Focus | Key Concepts & Contributions |
|---|---|---|
| [**01_VISION.md**](01_VISION.md) | **Mission & Strategic Direction** | Transitioning from prompt-driven to contract-driven generation; 5 core axioms; long-term visual compiler vision. |
| [**02_PROBLEM_STATEMENT.md**](02_PROBLEM_STATEMENT.md) | **Problems & Failure Modes** | The cost of unverified generation; taxonomy of failure modes (numeracy collapse, prompt bleed, spatial inversion). |
| [**03_RESEARCH_HYPOTHESIS.md**](03_RESEARCH_HYPOTHESIS.md) | **Research Hypotheses & Validation** | Primary hypothesis ($H_1$), secondary hypotheses, statistical testing protocols, and formal falsification criteria. |
| [**04_SYSTEM_ARCHITECTURE.md**](04_SYSTEM_ARCHITECTURE.md) | **System Architecture & Data Flow** | Config-driven stage graph; module breakdown; 6GB VRAM memory lifecycle; SQLite persistence architecture. |
| [**05_CONTRACT_SPEC.md**](05_CONTRACT_SPEC.md) | **Formal Contract Specification** | Complete JSON Schema (ContractV2); entity bounding boxes, z-index, relationships, and integrity rules. |
| [**06_VERIFICATION_SPEC.md**](06_VERIFICATION_SPEC.md) | **Multi-Modal Verification Suite** | Decoupled verification principle; GroundingDINO count checks; BLIP-VQA attribute scoring; reward gating ($\tau$). |
| [**07_BENCHMARKING.md**](07_BENCHMARKING.md) | **Benchmarking & Measured Results** | 30-scene benchmark suite; ablation study tables on consumer GPUs; statistical evaluation and reproducibility steps. |
| [**08_FAILURE_CASES.md**](08_FAILURE_CASES.md) | **Failure Cases & Risk Mitigation** | High-cardinality saturation; micro-scale misses; verifier shared bias; infinite loop defense mechanisms. |
| [**09_ROADMAP.md**](09_ROADMAP.md) | **Phased Technical Roadmap** | Phase 1 (MVP) $\to$ Phase 2 (Research) $\to$ Phase 3 (Production) $\to$ Phase 4 (Enterprise & Temporal Scale). |
| [**10_COMPETITOR_ANALYSIS.md**](10_COMPETITOR_ANALYSIS.md) | **Competitor Analysis & Differentiation**| Deep technical comparison against Midjourney, Flux, SDXL, DALL-E 3, and ComfyUI workflows. |
| [**11_PRODUCT_POSITIONING.md**](11_PRODUCT_POSITIONING.md) | **Product Strategy & Value Proposition**| ICPs (Autonomous Agents, Synthetic Data CV Teams, Pre-Viz Studios); the 10-minute developer promise. |
| [**12_FUTURE_DIRECTIONS.md**](12_FUTURE_DIRECTIONS.md) | **Spatio-Temporal & Multi-Agent Horizons**| Video generation contracts ($3D+T$); character identity persistence; multi-agent visual swarms; regional inpainting. |
| [**13_IMPLEMENTATION_STRATEGY.md**](13_IMPLEMENTATION_STRATEGY.md) | **Engineering Priorities & Standards** | Scientific validity, reproducibility, quality gates, automated testing, and CI/CD benchmarks. |
| [**14_INDEX_AND_RETRIEVAL_ARCHITECTURE.md**](14_INDEX_AND_RETRIEVAL_ARCHITECTURE.md) | **Index-First Multi-Modal Visual RAG** | NeMo RAG-inspired multi-modal indexing, hybrid dense-sparse BM25, spatial layout blueprints, and query decomposition. |
| [**15_CONSTRAINT_TRAINING.md**](15_CONSTRAINT_TRAINING.md) | **Local Constraint-Adapter Training** | PEFT NF4 QLoRA on 6 GB, grouped datasets, local MLflow, resumable evidence and qualification gates. |
| [**16_CONSTRAINT_CONTROL.md**](16_CONSTRAINT_CONTROL.md) | **Instance Control & Local Verification** | Regional denoising, mask centering, local count refinement, chromatic evidence and bounded repair. |

---

## Core System Architecture at a Glance

```
config.yaml: pipeline.stages: [intent, retrieval, plan, adapt, generate, verify, reflect, refine]

 prompt ──> INTENT ──> RETRIEVAL ──┐  (executed once per run)
                                   ▼
        ┌────────── repair loop (attempt k in 0 .. max_repair_loops) ───────┐
        │  PLAN ──> ADAPT ──> GENERATE ──> VERIFY ───────────> REFLECT      │
        │  masks    LoRA      SDXL-Turbo   DINO count check   hard_only?    │
        │  prompt   inject    +ControlNet  CLIP score         corrective    │
        │  compose  (once)    512px, 4stp  LLM-vision VQA     prompt edits ─┘
        └─────────────────────────────────────────────────────────│─────────
                                                                  ▼
                          accepted artifact ──> REFINE (opt-in) + audit receipt
```

---

## The Six Non-Negotiable Invariants

1. **Deterministic Contracts**: Contracts are source-of-truth; prompts are transient compilation artifacts.
2. **First-Class Verification**: Visual generation is incomplete without independent, decoupled verification.
3. **Optimized for Constraint Satisfaction (CSR)**: We optimize for mathematical correctness, not aesthetic saturation.
4. **Honest Measurement**: Claims are paid for in reproducible measurements. No unsupported marketing claims.
5. **No Hallucination Elimination Claims**: We claim measurable hallucination reduction supported by benchmarks.
6. **Consumer Hardware Feasibility**: The reference implementation must execute completely within $\le 6\text{GB}$ VRAM.
