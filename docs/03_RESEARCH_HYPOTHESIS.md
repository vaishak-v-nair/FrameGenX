# 03_RESEARCH_HYPOTHESIS.md — Hypotheses, Validation Protocols, and Falsification Criteria

---

## 1. Primary Research Hypothesis

$$\mathbf{H_1}: \quad \text{CSR}(\text{FrameGenX}_{\text{Closed-Loop}}) > \text{CSR}(\text{Baseline}_{\text{Open-Loop}}) \quad \text{with } p < 0.01$$

> **Formal Statement**: Decoupling visual generation into formal semantic contract extraction, deterministic 2D spatial layout projection, conditioned latent diffusion, and decoupled vision-language verification with autonomous reflection repair loops produces a **statistically significant increase in the Constraint Satisfaction Rate (CSR)** compared to monolithic, prompt-driven text-to-image foundation models.

Instead of expecting a continuous latent diffusion model to intrinsically perform discrete combinatorial reasoning, FrameGenX treats the diffusion model as a **constrained neural rendering engine** whose spatial distribution and cross-attention activations are governed by an external, verifiable contract.

---

## 2. Secondary Research Hypotheses

```
+─────────────────────────────────────────────────────────────────────────────+
|                         SECONDARY RESEARCH HYPOTHESES                       |
+─────────────────────────────────────────────────────────────────────────────+
|  H2.1 (Numeracy & Cardinality)                                              |
|  Explicit spatial partitioning via geometric bounding masks forces discrete|
|  object isolation, reducing object under-/over-counting by >= 40% vs SDXL.  |
+─────────────────────────────────────────────────────────────────────────────+
|  H2.2 (Attribute Disentanglement)                                           |
|  Region-bounded conditioning restricts cross-attention token mixing,        |
|  suppressing color/material prompt bleed across adjacent entities.          |
+─────────────────────────────────────────────────────────────────────────────+
|  H2.3 (Autonomous Repair Convergence)                                       |
|  Closed-loop discrepancy reflection with targeted corrective prompt and     |
|  mask edits monotonically increases composite reward R across K <= 3        |
|  repair attempts without catastrophic semantic drift.                       |
+─────────────────────────────────────────────────────────────────────────────+
|  H2.4 (Resource-Constrained Feasibility)                                    |
|  A strict sequential model lifecycle (staged CPU/VRAM swapping) enables     |
|  full contract extraction, SDXL generation, GroundingDINO verification, and |
|  BLIP-VQA scoring to execute stably within <= 6GB VRAM and <= 16GB RAM.     |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Validation Criteria & Statistical Testing Protocol

### 3.1 Experimental Design
The validation suite evaluates four standardized configurations across an identical set of evaluation prompts with fixed random seeds:

1. **Configuration A (Baseline)**: Raw SDXL-Turbo (512px, 4 steps, cfg 0.0, monolithic prompt).
2. **Configuration B (RAG-Augmented)**: SDXL-Turbo + Retrieval-Augmented Generation (COCO caption grounding).
3. **Configuration C (RAG + LoRA)**: SDXL-Turbo + RAG + Style/Domain LoRA adapter injection.
4. **Configuration D (Full FrameGenX Pipeline)**: Intent Extraction $\to$ Retrieval $\to$ Spatial Planning $\to$ SDXL-Turbo Generation $\to$ GroundingDINO Count Verification $\to$ BLIP-VQA Attribute Scoring $\to$ Closed-Loop Reflection & Repair Loop ($K \le 3$).

### 3.2 Quantitative Metrics

| Metric | Symbol | Mathematical Formulation | Evaluation Method |
|---|---|---|---|
| **Constraint Satisfaction Rate** | $\text{CSR}$ | $\frac{1}{N} \sum_{i=1}^N \prod_{c \in \mathcal{C}_i} \mathbb{I}(\text{Verifier}(c, \mathcal{I}_i) == 1)$ | Binary pass across all hard constraints (DINO object count matching). |
| **Normalized CLIP Score** | $\text{CLIP}_{\text{norm}}$ | $\frac{\cos(\mathbf{E}_{\text{img}}, \mathbf{E}_{\text{orig\_prompt}}) - \mu_{\text{clip}}}{\sigma_{\text{clip}}}$ | Evaluated strictly against the **original user prompt** (not the internal compiled prompt). |
| **Visual Question Answering Score** | $\text{VQA}$ | $\frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{Confidence}(\text{BLIP-VQA}(q, \mathcal{I}) == \text{ans}_{\text{expected}})$ | Probes attribute binding (e.g., `"Is the umbrella red?"`). |
| **Composite Quality Reward** | $R$ | $w_1 \cdot \text{CLIP}_{\text{norm}} + w_2 \cdot \text{VQA} \quad (w_1=0.5, w_2=0.5)$ | Used by the Reflection engine to evaluate candidate acceptance against gate $\tau$. |
| **Repair Convergence Rate** | $\eta_{\text{repair}}$ | $\frac{\text{Passed After Attempts } [1, 3]}{\text{Total Initial Rejections}}$ | Measures the percentage of initial failures successfully rescued by reflection. |
| **Latency & Compute Overhead** | $T_{\text{exec}}$ | $t_{\text{final}} - t_{\text{init}} \text{ (seconds)}$ | Wall-clock execution time per successful generation. |

### 3.3 Statistical Significance Framework
- **Null Hypothesis ($H_0$)**: $\text{CSR}(\text{Full Pipeline}) \le \text{CSR}(\text{Baseline})$.
- **Alternative Hypothesis ($H_1$)**: $\text{CSR}(\text{Full Pipeline}) > \text{CSR}(\text{Baseline})$.
- **Statistical Tests**:
  - Two-tailed Welch's t-test for continuous metrics ($R, \text{CLIP}_{\text{norm}}, \text{VQA}$).
  - McNemar's Test and Fisher's Exact Test for paired binary outcomes ($\text{CSR}_{\text{pass/fail}}$).
  - Significance threshold set at $\alpha = 0.05$ with $95\%$ confidence intervals calculated via bootstrapping ($B = 10,000$).

---

## 4. Falsification & Failure Conditions

To maintain scientific integrity, we explicitly define the empirical results that will **falsify** our core hypotheses:

```
+─────────────────────────────────────────────────────────────────────────────+
|                         FALSIFICATION CONDITIONS                            |
+─────────────────────────────────────────────────────────────────────────────+
|  1. Insignificant CSR Delta:                                                |
|     If FrameGenX fails to achieve a statistically significant increase in   |
|     CSR (p >= 0.05) over raw baseline diffusion models on the benchmark.    |
|                                                                             |
|  2. Verifier Degradation / Correlated Bias:                                 |
|     If the independent verification models (GroundingDINO, BLIP) exhibit    |
|     correlated failure modes with the generator, resulting in a false-      |
|     positive verification rate > 15% (verified by blinded human audit).     |
|                                                                             |
|  3. Negative Repair Drift:                                                  |
|     If autonomous repair iterations (attempts 1 -> 3) decrease semantic      |
|     alignment (CLIP_norm) or induce visual artifacts faster than they       |
|     resolve structural count/spatial errors.                                |
|                                                                             |
|  4. Infeasible Compute Overhead:                                            |
|     If the time-to-acceptance latency exceeds 10x baseline without achieving|
|     at least a 2x improvement in constraint satisfaction.                   |
+─────────────────────────────────────────────────────────────────────────────+
```
