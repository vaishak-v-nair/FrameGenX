# 04_SYSTEM_ARCHITECTURE.md — System Architecture, Data Flow, and Module Specifications

---

## 1. High-Level Architectural Topology

FrameGenX is architected as a **strictly decoupled, config-driven stage graph with closed-loop autonomous reflection**. 

Instead of a monolithic pipeline where prompts are directly fed into diffusion pipelines, FrameGenX separates semantic parsing, spatial planning, latent generation, multi-modal verification, and reflection into independent, testable stages communicating via an immutable in-memory state object (`RunContext`) and persistent storage.

```
                                  FRAMEGENX STAGE-GRAPH TOPOLOGY
                                  
   +───────────────────────────────────────────────────────────────────────────────+
   | PRE-LOOP STAGES (Executed Once per Run)                                       |
   |                                                                               |
   |  [User Prompt] ──> [1. INTENT STAGE]    ──> Generates ContractV2 (Entities,    |
   |                           │                 Counts, Attributes, Spatial Rels) |
   |                           ▼                                                   |
   |                    [2. RETRIEVAL STAGE] ──> Queries COCO Vector Store for     |
   |                                             Grounding Phrases (RAG)           |
   +───────────────────────────────────────────────────────────────────────────────+
                                       │
                                       ▼
   +───────────────────────────────────────────────────────────────────────────────+
   | AUTONOMOUS REPAIR LOOP (Executed for Attempt k in 0 .. max_repair_loops)      |
   |                                                                               |
   |  ┌─> [3. PLAN STAGE]     ──> Computes 2D Bounding Boxes, Segmentation Masks,  |
   |  │                           and Composes Token-Budgeted Prompt               |
   |  │        │                                                                   |
   |  │        ▼                                                                   |
   |  │   [4. ADAPT STAGE]    ──> Injects LoRA Weights into Diffusion Pipeline     |
   |  │        │                  (Pre-CPU-offload hook, loaded once)              |
   |  │        ▼                                                                   |
   |  │   [5. GENERATE STAGE] ──> SDXL-Turbo / Flow Matching Engine                |
   |  │        │                  (512px, 4 steps, CFG 0, with ControlNet)         |
   |  │        ▼                                                                   |
   |  │   [6. VERIFY STAGE]   ──> Multi-Modal Verification Suite:                  |
   |  │        │                  - GroundingDINO (Open-Vocabulary Count Check)    |
   |  │        │                  - CLIP Score (vs. ORIGINAL prompt)               |
   |  │        │                  - BLIP-VQA (Attribute Binding Questions)         |
   |  │        ▼                                                                   |
   |  │   [7. REFLECT STAGE]  ──> Calculates Composite Reward R:                   |
   |  │        │                  R = 0.5 * CLIP_norm + 0.5 * VQA                  |
   |  │        ▼                                                                   |
   |  │   [Decision Gate]     ──> Is Hard Failures == 0 AND R >= tau?              |
   |  │        │                                                                   |
   |  │        ├─── YES ──> [ACCEPT] ──> Emit Artifact Trail & Final Image         |
   |  │        │                                                                   |
   |  └────────└─── NO  ──> [REPAIR] ──> Synthesize Corrective Prompt & Mask Edits |
   +───────────────────────────────────────────────────────────────────────────────+
```

---

## 2. Core Subsystems & Module Breakdown

### 2.1 Intent Extractor (`src/intent/`)
- **Responsibility**: Translates unstructured natural language into the strict, machine-readable `ContractV2` schema.
- **Implementation**:
  - **LLM-based extraction** (2026-08 pivot from spaCy-only): Groq `gpt-oss-120b` → NVIDIA NIM → Gemini, behind `src/core/llm_router.py` with per-provider model env overrides and cooldown fallbacks.
  - Deterministic backstops enforced in code (never trusted to the LLM):
    uncountable-concept stoplist (`intent.uncountable_stoplist`), bare-plural
    → `uncounted` presence-only entities, and emission of the rule-based
    `repair_policy` (paper-contract parity).
  - spaCy (`en_core_web_sm`) survives in `src/intent/relationships.py` for spatial predicate extraction.
  - Normalizes class names into open-vocabulary target queries for the verification suite.

### 2.2 Semantic Retrieval Engine (`src/retrieval/`)
- **Responsibility**: Grounding user intent against proven photographic captions to eliminate out-of-distribution prompt phrasings.
- **Implementation**:
  - Pre-computed FAISS/vector index of 1,000+ COCO val2017 captions embedded with CLIP ViT-B/32.
  - Retrieves top-$K$ semantic neighbors to provide realistic contextual phrase grounding without modifying the declared contract entities.

### 2.3 Spatial Scene Planner (`src/scene/`)
- **Responsibility**: Translates semantic contract entities and relational constraints into physical 2D coordinates and prompts.
- **Implementation**:
  - Deterministic layout engine assigns normalized bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]^4$.
  - Generates binary spatial masks for regional attention or ControlNet conditioning.
  - Strict 77-token budget prompt composer: Prioritizes active entities and corrective edits while discarding redundant adjectives.

### 2.4 Adaptation & LoRA Layer (`src/generation/lora.py`)
- **Responsibility**: Dynamic style and domain adaptation without full model fine-tuning.
- **Implementation**:
  - Injects PEFT / Diffusers LoRA adapters directly into cross-attention projection layers.
  - Pre-offload hook ensures adapter weights reside in correct device memory before `enable_model_cpu_offload` executes.

### 2.5 Latent Generation Engine (`src/generation/`)
- **Responsibility**: Neural latent rendering under strict hardware constraints.
- **Implementation**:
  - **Backbone**: SDXL-Turbo (Adversarial Diffusion Distillation), generating 512x512 images in 4 inference steps with CFG 0.0.
  - **ControlNet Integration**: Canny edge and depth conditioning for structural boundary enforcement.
  - **Memory Safeguards**: Sequential CPU offloading, FP16 half-precision execution, and explicit VRAM garbage collection.

### 2.6 Multi-Modal Verification Suite (`src/verification/`)
- **Responsibility**: Decoupled, independent evaluation of the generated artifact against the initial contract.
- **Components**:
  1. **GroundingDINO (`QAVerifier`, grounding-dino-**tiny**)**: Open-vocabulary zero-shot detector. Counts detected instances per counted entity class; `uncounted` entities (bare plurals) are presence-checked, never counted. fp32, CPU-resident between uses.
  2. **CLIP Verifier (`CLIPScoreVerifier`)**: Cosine similarity between image embedding and **original user prompt** (preventing prompt-compiler drift). Process-lifetime CPU singleton shared with retrieval.
  3. **LLM-vision VQA (`VQAVerifier`)**: Batched existence/color questions answered by a frontier vision model via `src/core/llm_router.py` (Gemini flash-latest / NIM). Replaced local BLIP-VQA (2026-08 pivot): 0 VRAM, stronger answers, still architecturally decoupled from the generator.
  4. **Spatial Verifier (`SpatialRelationVerifier`)**: Pure CPU geometry over DINO boxes — 0 VRAM.
  5. **Optional plugins** (`config: verification.plugins`): `scene_comprehension` (LLM-vision scene description cross-check) and `easyocr` (text-hallucination diagnostic). Both are **soft signals** — they populate `verification_report.signals`, never hard-reject, and run only on the final attempt.

### 2.7 Reflection & Repair Engine (`src/reflection/`)
- **Responsibility**: Closed-loop diagnostic scoring and corrective action synthesis.
- **Implementation**:
  - Computes composite reward: $R = 0.5 \cdot \text{CLIP}_{\text{norm}} + 0.5 \cdot \text{VQA}_{\text{score}}$.
  - Compares $R$ against empirical acceptance threshold $\tau$ calibrated via validation pilots.
  - If rejected, extracts discrepancy vector $\Delta = \text{Contract} \setminus \text{Detections}$ and generates structured prompt edits (e.g., `"add 1 apple, clearly separated"` or `"increase contrast between red apple and green table"`).

---

## 3. Hardware & Memory Lifecycle Architecture (6GB VRAM Budget)

To ensure full reproducibility on consumer laptops (RTX 3050 6GB VRAM, 16GB RAM), FrameGenX implements a **Zero-Concurrency VRAM Model Lifecycle**:

```
[Time Step]       [Active Module]          [Device]    [VRAM Allocated]    [System RAM]
────────────────────────────────────────────────────────────────────────────────────────
T0 (Init)         Intent Extraction (LLM)  API         0 MB               ~400 MB
T1 (Retrieval)    CLIP+BM25 Search         CPU         0 MB               ~600 MB
T2 (Planning)     Layout Calculation       CPU         0 MB               ~600 MB
T3 (Generation)   SDXL-Turbo (+LoRA)       GPU (CUDA)  ~3,500 MB peaks    ~3-4 GB
                  sequential CPU offload:              (layers stream     (weights
                  RAM->GPU per layer)                  through RAM)       streamed)
T4 (Flush 1)      free_memory()            CUDA Flush  < 100 MB           —
T5 (Count Check)  GroundingDINO-tiny       GPU (CUDA)  ~1,800 MB          ~2,200 MB
T6 (Flush 2)      free_memory()            CUDA Flush  < 100 MB           —
T7 (VQA Check)    LLM-vision via router    API         0 MB               ~2,500 MB
                  (+CLIP score on GPU)                 ~1,200 MB transient
T8 (Reflection)   Reward & Edits           CPU         0 MB               ~1,200 MB
T9 (Refine, opt)  Reuses T3 pipe           GPU (CUDA)  ~3,500 MB peaks    unchanged
```

**Sequential offload rationale (measured 2026-08-24)**: the RTX 3050 Laptop
drives the Windows display, leaving only ~4.6-5.1GB VRAM actually free.
Model-offload's 4.9GB UNet spills into driver-managed system memory and
thrashes (70s/step); sequential streaming never spills and wins (44s/step).

### Memory Management Primitives (`src/utils.py`):
```python
def free_memory():
    """Forces garbage collection and flushes PyTorch CUDA memory allocator."""
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

---

## 4. RunContext State Machine & Persistence Architecture

Every execution creates a unique run directory `runs/<run_id>/` and updates a local SQLite database (`runs/framegenx.db`).

### 4.1 On-Disk Artifact Manifest
```
runs/run_20260814_123456_a1b2c3/
├── contract_v2.json           # Initial parsed contract
├── plan.json                  # Spatial bounding boxes & masks
├── attempt_0/
│   ├── draft.png              # Raw generated image attempt 0
│   ├── dino_detections.json   # GroundingDINO boxes and scores
│   ├── vqa_results.json       # BLIP-VQA question/answer scores
│   └── decision_0.json        # Pass/Fail decision, reward, and edits
├── attempt_1/                 # (If repair loop triggered)
│   ├── draft.png
│   ├── dino_detections.json
│   ├── vqa_results.json
│   └── decision_1.json
├── final_accepted.png         # Final accepted artifact
└── receipt.json               # Cryptographic/provenance execution receipt
```

### 4.2 SQLite Database Schema (`src/core/db.py`)
```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT NOT NULL,
    config_name TEXT NOT NULL,
    is_successful BOOLEAN NOT NULL,
    total_attempts INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    final_reward REAL,
    clip_norm REAL,
    vqa_score REAL,
    contract_json TEXT NOT NULL,
    decision_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    hard_failures TEXT,
    clip_norm REAL,
    vqa_score REAL,
    reward REAL,
    accepted BOOLEAN NOT NULL,
    prompt_edits TEXT
);
```
