# 06_VERIFICATION_SPEC.md — Multi-Modal Verification Architecture and Scoring Specifications

> **Specification with historical design sections.** Current acceptance in
> `config.yaml` is `hard_only`: object counts and supported box-based spatial
> relations gate acceptance. Attribute VQA and CLIP are diagnostic. Current
> count processing suppresses near-identical same-class boxes at IoU 0.90;
> the 0.45 NMS threshold, regional BLIP crops, and reward gate described below
> are historical design targets. The per-requirement ledger and current
> validation are documented in `reports/research_image_generation_2026-10-13.md`.

---

## 1. The Decoupled Verification Principle

In software engineering, testing a function with the same logic used to write the function leads to catastrophic false negatives. 

Similarly, in generative AI, verifying a diffusion model using another generative diffusion inversion or using internal cross-attention maps creates **correlated bias**. If the generator suffers from an attribute-leakage blindspot, its internal activations will share that blindspot.

FrameGenX enforces the **Decoupled Verification Principle**:
> **The Verification Layer must consist of independent, discriminative vision-language and object-detection architectures whose training objectives, loss functions, and spatial representations are completely decoupled from the generative diffusion backbone.**

```
+─────────────────────────────────────────────────────────────────────────────+
|                         DECOUPLED VERIFICATION MATRIX                       |
+───────────────────────────+───────────────────────+─────────────────────────+
| SYSTEM COMPONENT          | ARCHITECTURAL FAMILY  | OBJECTIVE FUNCTION      |
+───────────────────────────+───────────────────────+─────────────────────────+
| Generator (Latent Engine) | SDXL-Turbo / Flow     | Score Matching / DMD    |
|                           | Diffusion (UNet/DiT)  | Continuous Denoising    |
+───────────────────────────+───────────────────────+─────────────────────────+
| Verifier 1 (Cardinality)  | GroundingDINO-tiny    | Open-Set Box Regression |
|                           | Transformer Detector  | Cross-Modality Contrast |
+───────────────────────────+───────────────────────+─────────────────────────+
| Verifier 2 (Attributes)   | LLM-vision VLM via    | Autoregressive Sequence |
| (2026-08: was local       | router (Gemini / NIM; | Language Modeling Loss  |
| BLIP-VQA; Florence-2      | originally BLIP-VQA)  |                         |
| evaluated, retired —      |                       |                         |
| transformers 5.x          |                       |                         |
| incompatible)             |                       |                         |
+───────────────────────────+───────────────────────+─────────────────────────+
| Verifier 3 (Semantics)    | CLIP ViT-B/32 or L/14 | InfoNCE Dual Encoder    |
|                           | Multimodal Embedding  | Contrastive Cosine Sim  |
+───────────────────────────+───────────────────────+─────────────────────────+
```

---

## 2. Multi-Tier Verification Pipeline

The verification engine evaluates candidate images through four sequential tiers of increasing semantic abstraction:

```
                               VERIFICATION PIPELINE FLOW
                               
   [Candidate Image I] + [Contract C]
            │
            ▼
   +───────────────────────────────────+
   | TIER 1: DISCRETE COUNT VALIDATION | ──> GroundingDINO Open-Vocabulary Count Check
   +───────────────────────────────────+
            │
            ├─── Count Mismatch? ──> [HARD FAILURE] ──> Reject Immediately
            ▼
   +───────────────────────────────────+
   | TIER 2: SPATIAL TOPOLOGY CHECK    | ──> Centroid Coordinate & Bounding Box Relational Graph
   +───────────────────────────────────+
            │
            ├─── Spatial Violation? ─> [HARD FAILURE] ──> Reject Immediately
            ▼
   +───────────────────────────────────+
   | TIER 3: ATTRIBUTE BINDING (VQA)   | ──> BLIP-VQA Probes on Regional Crops & Global Image
   +───────────────────────────────────+
            │
            ▼ Computes VQA Score in [0, 1]
   +───────────────────────────────────+
   | TIER 4: GLOBAL SEMANTIC ALIGNMENT | ──> CLIP Score vs. ORIGINAL User Prompt
   +───────────────────────────────────+
            │
            ▼ Computes CLIP_norm in [0, 1]
   +───────────────────────────────────+
   | DECISION: REWARD COMPILATION      | ──> R = 0.5 * CLIP_norm + 0.5 * VQA >= tau ?
   +───────────────────────────────────+
```

---

## 3. Mathematical Formulations of Verification Metrics

### 3.1 Tier 1: Object Count & Cardinality Verification
For each contracted entity $e_i \in \mathcal{E}$ with declared target count $N(e_i)$ and normalized class name $c(e_i)$:
1. Query GroundingDINO with text prompt $c(e_i)$ and box confidence threshold $\theta_{\text{dino}} = 0.35$.
2. Apply Non-Maximum Suppression (NMS) with IoU threshold $\theta_{\text{nms}} = 0.45$ to eliminate duplicate detections.
3. Count surviving bounding boxes: $\hat{N}(e_i) = |\{\mathbf{b} \in \mathcal{B} \mid \text{class}(\mathbf{b}) = c(e_i) \land \text{conf}(\mathbf{b}) \ge \theta_{\text{dino}}\}|$.
4. Define Cardinality Hard Failure $H_{\text{count}}$:
$$H_{\text{count}} = \bigvee_{e_i \in \mathcal{E}} \left( \hat{N}(e_i) \neq N(e_i) \right)$$

### 3.2 Tier 2: Spatial Topology & Relational Verification
For each relational tuple $(e_j, \text{predicate}, e_k) \in \mathcal{R}$:
1. Compute centroids $(\bar{x}_j, \bar{y}_j)$ and $(\bar{x}_k, \bar{y}_k)$ of detected bounding boxes.
2. Evaluate geometric truth predicates:
$$\text{Rel}(\text{"left\_of"}, \mathbf{b}_j, \mathbf{b}_k) = \begin{cases} 1 & \text{if } x_{\max}(\mathbf{b}_j) < x_{\min}(\mathbf{b}_k) + \delta \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Rel}(\text{"standing\_on"}, \mathbf{b}_j, \mathbf{b}_k) = \begin{cases} 1 & \text{if } y_{\max}(\mathbf{b}_j) \approx y_{\min}(\mathbf{b}_k) \land \text{IoU}_{x}(\mathbf{b}_j, \mathbf{b}_k) > 0.3 \\ 0 & \text{otherwise} \end{cases}$$
3. Define Spatial Hard Failure $H_{\text{spatial}}$:
$$H_{\text{spatial}} = \bigvee_{r \in \mathcal{R}} \left( \text{Rel}(r) == 0 \right)$$

### 3.3 Tier 3: Attribute Binding & VQA Scoring
For each entity $e_i$ and each declared attribute $a_{i,m} \in \text{Attributes}(e_i)$:
1. Formulate structured visual question:
$$q_{i,m} = \text{"What color/material is the "} + c(e_i) + \text{"?"}$$
2. Query BLIP-VQA on image crop $\mathcal{I}[\mathbf{b}_i]$ or global image:
$$p_{i,m} = P_{\text{VQA}}(\text{answer} = a_{i,m} \mid \mathcal{I}, q_{i,m})$$
3. Compute mean attribute score:
$$\text{VQA}_{\text{score}} = \frac{1}{|\mathcal{A}|} \sum_{a_{i,m} \in \mathcal{A}} p_{i,m}$$

### 3.4 Tier 4: Normalized CLIP Semantic Score
To prevent "prompt-compiler drift" (where the system scores the image against its own internally compiled prompt rather than what the user asked for), CLIP cosine similarity is **strictly evaluated against the original user prompt $P_{\text{orig}}$**:
$$S_{\text{clip}} = \cos\left(\mathbf{E}_{\text{img}}(\mathcal{I}), \mathbf{E}_{\text{text}}(P_{\text{orig}})\right)$$
$$S_{\text{clip\_norm}} = \text{clip}\left(\frac{S_{\text{clip}} - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}, 0.0, 1.0\right)$$
where empirical calibration constants on COCO/SDXL are $\mu_{\text{baseline}} = 0.20, \sigma_{\text{baseline}} = 0.15$.

### 3.5 Composite Reward & Acceptance Gating
The candidate generation $\mathcal{I}_k$ is scored with the composite reward:
$$R_k = 0.5 \cdot S_{\text{clip\_norm}} + 0.5 \cdot \text{VQA}_{\text{score}}$$

**Acceptance (current default — `mode: hard_only`, set 2026-08-15):**
$$\text{Acceptance}(\mathcal{I}_k) = (\neg H_{\text{count}}) \land (\neg H_{\text{spatial}})$$

The $\tau$ reward gate is RETIRED: `tau_sweep.py` measured Hard-only
CSR=57% vs τ=0.730 CSR=38% — the dual gate rejected borderline-good images
and actively degraded constraint satisfaction. R remains computed and
logged every attempt for reporting and future experiments
(`mode: reward_gate` re-enables the legacy gate).

### 3.6 Optional Diagnostic Plugins (soft signals)
`verification.plugins.scene_comprehension` (LLM-vision scene-description
cross-check) and `verification.plugins.easyocr` (text-hallucination OCR)
are **evidence-only**: their findings populate
`verification_report.signals` and never extend hard failures. Both run
once per run, on the final attempt, bounding their latency cost. Rationale:
a vision model's phrasing ("a feline") or incidental background signage
must not burn repair loops. Hard-gating them for specific benchmark
categories (e.g. `text_rendering`) is a measured decision for the future,
not a default.

---

## 4. Empirical Threshold Calibration ($\tau$)

The acceptance threshold $\tau$ determines the sensitivity/specificity balance of the repair loop. Setting $\tau$ too high induces infinite repair loops on difficult prompts; setting $\tau$ too low permits low-fidelity outputs.

```
       ACCEPTANCE THRESHOLD (tau) TRADEOFF CURVE (Empirical RTX 3050 Pilot)
       
  100% ┼─────────────────────────────────────────────────────────────
       │                  \                     / (Avg Latency & Loops)
   80% ┼─── (CSR) ─────────\                   /─────────────────────
       │                    \                 /
   60% ┼─────────────────────\───────────────/───────────────────────
       │                      \             /
   40% ┼───────────────────────\───────────/─────────────────────────
       │                        \         /
   20% ┼─────────────────────────\───────/───────────────────────────
       │                          \     /
    0% ┼───────┬─────────┬─────────\───/─────────┬─────────┬─────────
              0.50      0.60      0.70 (tau*)   0.80      0.90
```

- **Optimal Calibrated Operating Point**: $\tau^* = 0.70$.
- Under $\tau^* = 0.70$, the pipeline achieves high constraint satisfaction while maintaining average repair attempts $\le 2.6$ per generation.
