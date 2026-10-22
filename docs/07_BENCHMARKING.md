# 07_BENCHMARKING.md — Benchmarking Methodology, Metrics, and Measured Empirical Results

> **Historical benchmark snapshot.** The ablation table below predates
> explicit benchmark provenance and LoRA activation records. Its percentages
> represent the old hard-check pass rate, not verified satisfaction of all
> contract constraints. Re-run the benchmark and inspect current artifacts
> before using the numbers as evidence for the present code.

---

## 1. The Flaw in Existing Evaluation Paradigms

Historically, generative visual AI evaluation has relied on metrics that prioritize perceptual beauty over logical correctness:
1. **Fréchet Inception Distance (FID) / Kernel Inception Distance (KID)**: Measures distance between feature distributions of generated vs. real image sets. FID is blind to prompt adherence, discrete counts, and spatial correctness.
2. **Human Preference ELO (e.g., LMSYS Chatbot Arena, Midjourney ratings)**: Raters consistently prefer vibrant, high-contrast, aesthetically pleasing images over less stylized images, even when the beautiful image completely violates numeric or spatial constraints.
3. **Automated Aesthetic Scoring (Aesthetic Predictor, ImageReward, PickScore)**: Trained on subjective human "likes," reinforcing prompt-bleed and hallucination blindspots.

**FrameGenX establishes a Benchmark for Constraint Adherence:**
Visual generation must be evaluated with the same rigor as an automated unit-test suite. Success is defined by whether the generation satisfies the contract.

---

## 2. Dataset Design: The FrameGenX Benchmark Suite

The FrameGenX benchmark suite comprises 30 curated multi-constraint scenes (`benchmark_dataset.json`), designed to systematically probe known failure boundaries across 5 orthogonal failure categories:

```
+─────────────────────────────────────────────────────────────────────────────+
|                     FRAMEGENX BENCHMARK TAXONOMY (30 SCENES)                |
+──────────────────────────+─────────────────────────+────────────────────────+
| CATEGORY                 | SAMPLE PROMPT           | FAILURE MECHANISM      |
+──────────────────────────+─────────────────────────+────────────────────────+
| 1. Discrete Numeracy     | "Exactly four coffee    | Cross-attention density|
|    (Cardinality)         | cups on a wooden tray"  | smearing; under-count  |
+──────────────────────────+─────────────────────────+────────────────────────+
| 2. Attribute Binding     | "A green bottle and a   | Color/material token   |
|    (Prompt Bleed)        | red apple on a desk"    | cross-talk             |
+──────────────────────────+─────────────────────────+────────────────────────+
| 3. Spatial Topology      | "A cat sitting on the   | Relational coordinate  |
|    (Relational Logic)    | left of a dog"          | inversion / omission   |
+──────────────────────────+─────────────────────────+────────────────────────+
| 4. Multi-Entity Crowding | "A fork, knife, spoon,  | Entity fusion and      |
|    (Composite Scenes)    | glass, and plate set"   | missing objects        |
+──────────────────────────+─────────────────────────+────────────────────────+
| 5. Foreground/Background | "A massive lighthouse   | Depth inversion;       |
|    (Occlusion/Depth)     | behind a small sailboat"| scale distortion       |
+──────────────────────────+─────────────────────────+────────────────────────+
```

---

## 3. Measured Empirical Results (Ablation Study)

All experiments were executed locally on standard consumer hardware (**RTX 3050 Laptop GPU 6GB VRAM, 16GB System RAM, SDXL-Turbo 512px, 4 steps, fixed seed 42**) using a 15-prompt balanced subset representing all 5 categories.

### 3.1 Empirical Ablation Table

| Configuration | $N$ | Constraint Satisfaction (CSR) | $\text{CLIP}_{\text{norm}}$ | VQA Score | Composite Reward $R$ | Avg Attempts | Avg Latency (s) |
|---|---|---|---|---|---|---|---|
| **1. Baseline (SDXL-Turbo + DINO count-repair)** | 15 | **53.3%** | 0.522 | 0.860 | 0.691 | 2.40 | 197.5s |
| **2. + Retrieval (RAG Grounding)** | 15 | **40.0%** | 0.538 | 0.913 | 0.725 | 3.20 | 277.2s |
| **3. + RAG + Style LoRA** | 15 | **73.3%** | 0.554 | 0.923 | 0.739 | 2.60 | 239.3s |
| **4. Full (RAG + LoRA + Reflection)** | 15 | **33.3%** | **0.601** | 0.917 | **0.759** | 2.53 | 229.0s |

### 3.2 Scientific Analysis of Empirical Trends

1. **Why LoRA Increased CSR (73.3%)**:
   The pixel-art adapter (`nerijs/pixel-art-xl`) enforces discrete visual boundaries, high object contrast, and stylized separation of individual items, making it significantly easier for GroundingDINO to isolate discrete object contours.
2. **Why Full Pipeline Shows 33.3% CSR under Strict Threshold $\tau = 0.70$**:
   In the Full configuration, acceptance requires *both* hard count satisfaction *and* reward gating ($R \ge \tau$). While semantic quality ($\text{CLIP}_{\text{norm}} = 0.601, R = 0.759$) reached its highest peak across all configurations, the dual-gate requirement rejected borderline images that baseline accepted. This provides a clean demonstration of the precision vs. recall tradeoff in verification gates.
3. **Published Paper Comparison Note**:
   Paper-reported values (IJRASET, DOI: 10.22214/ijraset.2025.76265: FID 7.2, CLIP 0.63, VQA 0.68, 81% human preference) were conducted under different evaluation scales and non-reproducible hardware environments. FrameGenX adheres strictly to open, measured local reproducibility.

---

## 4. Reproducibility Protocol

To reproduce the benchmark results on any system with Python $\ge 3.10$ and CUDA:

```powershell
# 1. Activate Environment
.\venv\Scripts\Activate.ps1

# 2. Run Automated Multi-Configuration Ablation Suite
scripts\run_ablation.bat

# Or run individual configurations manually:
python benchmark.py --config baseline --limit 15 --seed 42
python benchmark.py --config rag --limit 15 --seed 42
python benchmark.py --config rag_lora --limit 15 --seed 42
python benchmark.py --config full --limit 15 --seed 42

# 3. Generate Formatted Results Markdown
python scripts\make_results.py
```

### Reproducibility Guarantees:
- **Seed Invariance**: All latent noise tensors and model initializations are anchored to explicit seed integers.
- **Model Checkpoint Pinning**:
  - Generator: `stabilityai/sdxl-turbo` (Hugging Face Revision pinned)
  - Detector: `IDEA-Research/grounding-dino-tiny`
  - VQA: LLM-vision via `src/core/llm_router.py` (`gemini-flash-latest` / Groq / NIM — provider responses are non-deterministic; VQA is a scoring signal, not the acceptance gate, so CSR stays reproducible. Originally local `Salesforce/blip-vqa-base`, pivoted 2026-08)
  - Text Encoder: `openai/clip-vit-base-patch32`

> **Provenance**: the table above was measured before the 2026-08-24
> quality revisions (prompt-anchored composition, uncounted scenery
> entities, dense-confirmed retrieval). Re-run before comparing against
> current code — see docs/research.md provenance note.
