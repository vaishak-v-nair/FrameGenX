# 09_ROADMAP.md — Phased Technical Roadmap (MVP to Enterprise Scale)

---

## 1. Roadmap Architecture Overview

The FrameGenX engineering lifecycle is organized into four sequential, milestone-gated phases. Each phase is defined by strict quantitative validation criteria rather than calendar deadlines.

```
       PHASE 1: MVP               PHASE 2: RESEARCH           PHASE 3: PRODUCTION          PHASE 4: ENTERPRISE
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ • Discrete Numeracy     │ │ • Attribute Isolation   │ │ • FastAPI + Glass-Box UI│ │ • Temporal Continuity   │
│ • GroundingDINO Verifier│ │ • Florence-2 VLM Stack  │ │ • Regional Inpainting   │ │ • Character Persistence │
│ • SDXL-Turbo 6GB Engine │ │ • Attention Steering    │ │ • SQLite Run Database   │ │ • Synthetic Data Factory│
│ • CLI & Benchmark Suite │ │ • Formal Whitepaper     │ │ • Python SDK Packaging  │ │ • Cloud Swappable Backends
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
             │                           │                           │                           │
             ▼                           ▼                           ▼                           ▼
     [Gate: CSR > 50%]          [Gate: CSR > 70%]          [Gate: Latency < 60s]       [Gate: Multi-Frame CSR]
```

---

## 2. Detailed Phase Breakdown

### Phase 1: MVP — Discrete Cardinality & Closed-Loop Baseline *(Completed)*
- **Primary Objective**: Prove that contract-driven generation and decoupled verification resolve discrete object counting on consumer hardware (6GB VRAM).
- **Key Deliverables**:
  - `src/intent/`: spaCy-based `ContractV2` extraction for entities and integer counts.
  - `src/scene/`: Grid layout mask generation and 77-token prompt composer.
  - `src/generation/`: SDXL-Turbo 4-step latent generator with CPU offloading.
  - `src/verification/`: GroundingDINO zero-shot object counter.
  - `src/reflection/`: Baseline count-mismatch prompt repair loop ($K \le 3$).
  - `benchmark.py`: 30-scene benchmark suite with reproducible ablation execution.
- **Phase Gate**: Achieve $>50\%$ Constraint Satisfaction Rate on the 15-prompt balanced test suite within $<6\text{GB}$ VRAM.

---

### Phase 2: Research Version — Fine-Grained Attribute & Spatial Disentanglement *(Current)*
- **Primary Objective**: Eliminate prompt bleed and enforce strict spatial topological relationships (e.g., `"left_of"`, `"inside"`).
- **Key Deliverables**:
  - **Florence-2 & BLIP-VQA Integration**: Unified phrase grounding, OCR reading, and fine-grained visual question answering.
  - **Regional Latent Conditioning**: Implementing spatial cross-attention steering (e.g., MultiDiffusion / BoxDiff) to confine adjectives strictly to their allocated bounding boxes.
  - **Automated Reward Calibration**: Dynamic tuning of the acceptance threshold $\tau$ using Pareto-optimal frontier search.
  - **Open Research Publication**: Comprehensive ablation paper open-sourcing all benchmarks, datasets, and verification weights.
- **Phase Gate**: Achieve $\text{CSR} \ge 70\%$ across spatial and attribute benchmark categories with zero regression in numeracy.

---

### Phase 3: Production Version — High-Throughput Developer Platform
- **Primary Objective**: Package FrameGenX into a production-grade developer platform with asynchronous execution, glass-box observability, and regional repair optimization.
- **Key Deliverables**:
  - **Asynchronous FastAPI Engine**: Non-blocking job submission, WebSocket live progress streaming, and task cancellation.
  - **Interactive Glass-Box Frontend**: React/TypeScript web dashboard showing live intermediate masks, bounding box overlays, VQA logs, and repair history.
  - **Sub-Canvas Regional Inpainting Repair**: Instead of regenerating full 512px images on repair loops, crop and inpaint only the failed bounding box region to cut repair latency by $60\%$.
  - **Python SDK (`pip install framegenx`)**: Clean programmatic interface:
    ```python
    from framegenx import VisualCompiler
    compiler = VisualCompiler(config="full")
    result = compiler.compile("three red apples on a wooden table")
    result.save("output.png")
    print(result.receipt.verification_report)
    ```
- **Phase Gate**: Median end-to-end generation latency $<60\text{s}$ on consumer GPUs for single-attempt passes; 100% test coverage on core stage interfaces.

---

### Phase 4: Enterprise Scale — Spatio-Temporal Consistency & Synthetic Data Factory
- **Primary Objective**: Extend the contract schema across the temporal axis for multi-frame storyboards, video generation, and industrial synthetic data pipelines.
- **Key Deliverables**:
  - **Spatio-Temporal Contract ($3D+T$)**: Adding persistent `entity_uuid`, cross-frame camera trajectories $[R | T]$, and depth continuity.
  - **Character Persistence Engine**: Storing character identity latent embeddings (e.g., InstantID / FaceID / subject LoRA) bound to entity contracts across multiple shots.
  - **Industrial Synthetic Dataset Factory**: Automated generation of 100,000+ image batches with zero-drift, pixel-perfect ground truth annotations exported directly into COCO, YOLO, and Pascal VOC formats.
  - **Enterprise Cloud Swappable Backends**: Pluggable support for remote high-performance generation backends (Flux.1-Dev, SD3.5-Large, Midjourney API) while retaining local verifier gates.
- **Phase Gate**: Zero identity drift across 10 consecutive storyboard frames; verified bounding box annotation precision $\ge 98.5\%$.
