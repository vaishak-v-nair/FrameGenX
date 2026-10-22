# 11_PRODUCT_POSITIONING.md — Product Strategy, Target Personas, and Value Proposition

---

## 1. Target Personas & Ideal Customer Profiles (ICPs)

FrameGenX is **not** designed for prompt artists seeking creative visual serendipity. It is engineered for **systems builders who treat visual generation as a deterministic software dependency**.

```
+─────────────────────────────────────────────────────────────────────────────+
|                         IDEAL CUSTOMER PROFILES (ICPs)                      |
+─────────────────────────────────────────────────────────────────────────────+
|  ICP 1: Autonomous Agent & LLM Tool Builders                                |
|  • Persona: AI Engineer building multi-agent autonomous applications.       |
|  • Job-to-be-Done: Grant LLM agents a "verifiable visual hand" to generate  |
|    storyboards, diagrams, and illustrations without human intervention.     |
|  • Core Pain: LLMs cannot trust visual model outputs; prompt hallucinations|
|    break automated agent workflows silently.                                |
+─────────────────────────────────────────────────────────────────────────────+
|  ICP 2: Synthetic Data & Computer Vision Engineers                         |
|  • Persona: ML Engineer training robotics, retail, or automotive CV models. |
|  • Job-to-be-Done: Generate 50,000 domain-specific training images with      |
|    perfectly aligned, zero-drift bounding box annotations.                 |
|  • Core Pain: 15-30% of standard generated images contain phantom objects or|
|    miss contracted targets, corrupting downstream model weights.            |
+─────────────────────────────────────────────────────────────────────────────+
|  ICP 3: Enterprise Automation & Brand Studios                               |
|  • Persona: Technical Director automating marketing asset generation.      |
|  • Job-to-be-Done: Generate ad variations adhering to strict composition,   |
|    logo placement, and product quantity constraints.                        |
|  • Core Pain: Human review teams spending hundreds of hours filtering out   |
|    non-compliant AI images.                                                 |
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 2. The 10-Minute Developer Value Proposition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE 10-MINUTE DEVELOPER CONTRACT                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Clone repo ──> Install requirements ──> Run CLI / API                      │
│                                                                             │
│  "Within 10 minutes of cloning, an engineer has a locally executable,       │
│   verified visual generation endpoint that returns an image alongside       │
│   a machine-readable verification receipt on standard consumer hardware."   │
└─────────────────────────────────────────────────────────────────────────────┘
```

When a developer submits:
```bash
python -m src.cli generate "three red apples on a rustic table"
```
They do not merely receive a PNG. They receive:
1. **The Accepted Render (`final_accepted.png`)**.
2. **The GroundingDINO Detection Matrix (`dino_detections.json`)** proving exact count = 3.
3. **The BLIP-VQA Attribute Confidence Log (`vqa_results.json`)** proving apples are red.
4. **The Full Execution Trail (`runs/<id>/`)** with timestamps, VRAM usage, and latency.

---

## 3. Value Proposition Canvas: Jobs, Pains, and Gains

```
+───────────────────────────────────+─────────────────────────────────────────+
| CUSTOMER PAINS (BEFORE FRAMEGENX) | VALUE CREATION (WITH FRAMEGENX)         |
+───────────────────────────────────+─────────────────────────────────────────+
| • Paying human QA testers to      | • Autonomous decoupled verification     |
|   filter hallucinated AI images   |   eliminates manual QA overhead         |
+───────────────────────────────────+─────────────────────────────────────────+
| • Re-rolling prompts 10x hoping   | • Bounded self-repair loop ($K \le 3$)   |
|   the model draws the right items |   diagnoses and corrects layout errors  |
+───────────────────────────────────+─────────────────────────────────────────+
| • Inconsistent annotations for    | • Machine-readable contracts export     |
|   synthetic dataset generation    |   perfect COCO/YOLO labels natively     |
+───────────────────────────────────+─────────────────────────────────────────+
| • Cloud API vendor lock-in and    | • 100% open-source, local-first runtime |
|   opaque non-deterministic updates|   guaranteed to execute on 6GB VRAM     |
+───────────────────────────────────+─────────────────────────────────────────+
```

---

## 4. Addressing Adoption Barriers & Strategic Objections

### Objection 1: "Isn't FrameGenX slower than calling raw SDXL or Midjourney?"
- **The Reality**: Generating an unverified image takes ~3 seconds. However, if that image has a 40% probability of violating constraints, an automated system must re-roll or require human review. A human review cycle takes 30–120 seconds.
- **The Counter**: FrameGenX trades a few seconds of local compute (VLM verification + repair) to eliminate minutes of human labor and prevent downstream data corruption.

### Objection 2: "Why not use ComfyUI?"
- **The Reality**: ComfyUI is an interactive canvas for human visual artists. It lacks declarative REST contracts, headless test runners, automated discrepancy reflection, and programmatic CI/CD integration.
- **The Counter**: FrameGenX is a **headless visual compiler for code and agents**, not a node canvas for human graphic designers.

### Objection 3: "Won't next-generation foundation models solve spatial reasoning anyway?"
- **The Reality**: Scaling foundation models improves average-case semantic understanding, but continuous latent diffusion models fundamentally lack discrete cardinality counting and coordinate bounds checking.
- **The Counter**: Even if foundation models improve, mission-critical systems still require independent unit tests. FrameGenX serves as the test harness and compiler regardless of the underlying foundation model.
