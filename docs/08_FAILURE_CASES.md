# 08_FAILURE_CASES.md — Failure Modes, Edge Cases, and Risk Mitigation

---

## 1. Taxonomic Analysis of Known Limitations

FrameGenX is explicitly engineered with open disclosure of its architectural boundaries. In accordance with rigorous scientific practice, the following limitations are formally documented:

```
                                  SYSTEM FAILURE BOUNDARY MATRIX
                                  
  HIGH COMPLEXITY ┼─────────────────────────────────────────────────────────────
                  │  [Zone 4: Physical Paradoxes]    [Zone 3: Cardinality Saturation]
                  │  - "Sphere inside a cube"        - "15 apples in 512px canvas"
                  │  - Result: Infinite Repair Loop  - Result: Object Merging / OOM
                  │                                  
  MEDIUM          ┼─────────────────────────────────────────────────────────────
                  │  [Zone 1: Nominal Operations]    [Zone 2: Small-Object Misses]
                  │  - "3 cars on a street"          - "Ring on left pinky finger"
                  │  - Result: High CSR (Pass)       - Result: Verifier False Negative
  LOW COMPLEXITY  ┼──────────────────────────────────┴──────────────────────────
                  LOW ENTITY COUNT                   HIGH ENTITY COUNT
```

---

## 2. Detailed Failure Scenarios & Behavioral Mechanics

### 2.1 Cardinality & Spatial Density Saturation ($N > 8$)
- **Description**: Requesting more than 8 distinct small objects within a 512x512 canvas resolution.
- **Root Cause**: The spatial resolution of SDXL-Turbo latent space ($64 \times 64$ latent grid) is insufficient to resolve independent cross-attention basins for $>8$ closely packed entities. Objects fuse into collective textures.
- **Observed Behavior**: GroundingDINO detects $N-2$ or $N-3$ bounding boxes because visual contours merge. The repair loop triggers repeated attempts without resolving pixel resolution limits.
- **Mitigation**: Introduce a dynamic canvas scaler that automatically expands canvas resolution to $1024 \times 1024$ or partitions generation into regional sub-canvas inpainting passes when entity count $N > 6$.

### 2.2 Micro-Scale Entities and Sub-Pixel Attributes
- **Description**: Constraints placed on fine-grained accessories or micro-features (e.g., `"a gold signet ring on the third finger of the left hand"`).
- **Root Cause**: Open-vocabulary object detectors (GroundingDINO) downsample input images through multi-scale feature pyramids. Tiny bounding boxes ($< 16 \times 16$ pixels) produce low feature activations, falling below detection threshold $\theta_{\text{dino}} = 0.35$.
- **Observed Behavior**: False-negative rejection. The image may visually contain the ring, but the verifier fails to detect it, burning compute on unnecessary repair iterations.
- **Mitigation**: Implement progressive multi-crop zooming (super-resolution cropping of candidate regions before passing to verifier).

### 2.3 Abstract & Non-Spatial Metaphors
- **Description**: Prompts lacking physical entities or spatial layouts (e.g., `"The feeling of melancholy on a rainy Sunday morning"`).
- **Root Cause**: The Intent Extractor attempts to parse nouns into bounding boxes. In abstract prompts, extracted nouns (`"feeling"`, `"melancholy"`) have no physical spatial extent.
- **Observed Behavior**: The Scene Planner outputs empty or invalid masks, bypassing structural enforcement and falling back to unconstrained diffusion.
- **Mitigation**: Fallback heuristic: If the Intent Extractor detects zero spatial entities, the pipeline automatically routes the generation to an unconstrained aesthetic mode, skipping spatial verification.

### 2.4 Physically Contradictory Contracts & Infinite Loops
- **Description**: Contracts specifying physically or topologically impossible scenes (e.g., `"A building larger than a mountain contained inside a teacup"`).
- **Root Cause**: The diffusion model cannot reconcile impossible lighting, perspective, and containment priors.
- **Observed Behavior**: The generator fails verification on every attempt. Without an iteration cap, the system would enter an infinite retry loop.
- **Mitigation**: Enforce a strict architectural limit: `max_repair_loops = 3`. If all 3 attempts fail, the pipeline returns the highest-reward attempt along with an explicit `FAILURE_RECEIPT` detailing the unsatisfied constraints.

---

## 3. Verifier Bias & Correlated Hallucinations

A critical research risk is **Verifier Shared Bias**:
- If both the generator (SDXL) and the verifier (BLIP-VQA / Florence-2) were trained on web-scraped datasets with identical cultural or visual biases (e.g., associating `"doctor"` exclusively with specific genders), the verifier may confirm an image as "correct" even when it violates fine-grained neutral prompt constraints.
- **Defense-in-Depth Strategy**:
  1. Multi-modal consensus: Requiring agreement between orthogonal detection architectures (GroundingDINO for boxes + BLIP for VQA + CLIP for semantics).
  2. Blinded human benchmark validation: Periodically evaluating verifier predictions against gold-standard human annotations to compute verifier precision and recall.

---

## 4. Risk Assessment & Engineering Mitigations

| Risk Factor | Probability | Impact | Engineering Mitigation Strategy |
|---|---|---|---|
| **VRAM Exhaustion (OOM Crash)** | Low | High | Mandatory sequential execution; strict `free_memory()` invocations between all stage transitions; model CPU offloading. |
| **Upstream Dependency Drift** | Medium | Medium | Pin all Hugging Face revisions, PyTorch CUDA versions, and spaCy language models in `requirements.txt` and `pyproject.toml`. |
| **High Compute Latency on Complex Repairs** | High | Low | Implement early-exit rewards and caching of intermediate latent tensors; limit repair loops to targeted regional inpainting. |
| **Silent False-Positive Verification** | Low | High | Independent verification threshold calibration ($\tau = 0.70$) validated via ablation sweeps. |
