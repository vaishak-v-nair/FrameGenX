# System Architecture

## 1. Overview
FrameGenX operates as a strictly decoupled, sequential pipeline. Rather than feeding natural language directly into a generative model, the system introduces intermediary semantic and spatial calculation layers, followed by a deterministic verification layer.

## 2. Component Modules

### A. Intent Extractor (NLP Layer)
- **Function**: Translates natural language prompts into structured constraints.
- **Mechanism**: Utilizes spaCy for dependency parsing and entity extraction, coupled with an LLM (or robust local models) to resolve ambiguities.
- **Output**: The initial `contract_v2.json` containing distinct entities, relationships, styles, and negative constraints.

### B. Scene Planner (Spatial Engine)
- **Function**: Converts semantic constraints into physical, 2D spatial layouts.
- **Mechanism**: A logic engine that calculates bounding boxes, overlapping areas, and depth ordering (Z-index). It determines which ControlNets or structural conditioning is required.
- **Output**: An enriched contract containing coordinates, masks, and generated depth/pose maps.

### C. Latent Drafter (Generation Layer)
- **Function**: The raw rendering engine.
- **Mechanism**: A monolithic diffusion model (e.g., SDXL, Flux) conditioned strictly on the output of the Scene Planner, utilizing `enable_model_cpu_offload` for memory efficiency.
- **Output**: A raw `draft_image.png`.

### D. Structural Enforcer (Control Layer)
- **Function**: Forces the generation to respect the Scene Planner's layout.
- **Mechanism**: Applies ControlNet (Depth, Canny, Pose, Region-based prompting) over the generation step. Due to hardware constraints (6GB VRAM), this layer caches models sequentially and applies them aggressively to prevent prompt bleed.
- **Output**: A structurally enforced `constrained_draft.png`.

### E. QA Verifier (Verification Layer)
- **Function**: Independently verifies the generated image against the initial contract.
- **Mechanism**: Uses specialized Vision-Language Models and Object Detectors (e.g., GroundingDINO for precise bounding boxes, Florence-2 for semantic understanding, EasyOCR for text).
- **Output**: A boolean Pass/Fail status and an itemized discrepancy report.

### F. Pipeline Orchestrator (The Controller)
- **Function**: Manages the state machine.
- **Mechanism**: Moves the contract and artifacts through the pipeline. If the QA Verifier returns a Fail, the Orchestrator initiates an autonomous repair loop, sending the discrepancy report back to the Scene Planner for adjustment.

## 3. Data Flow
1. `User Prompt` -> **[Intent Extractor]** -> `contract.json`
2. `contract.json` -> **[Scene Planner]** -> `layout_masks` + `enriched_contract.json`
3. `layout_masks` + `enriched_contract` -> **[Structural Enforcer]** + **[Latent Drafter]** -> `generated_image.png`
4. `generated_image.png` + `enriched_contract.json` -> **[QA Verifier]** -> `verification_report.json`
5. If Pass: Return Image. If Fail: Return to **[Scene Planner]** with `verification_report.json`.

## 4. Future Extensibility
- **Swappable Backends**: The architecture allows swapping the Latent Drafter from SDXL to Flux or Midjourney (via API) without modifying the Planning or Verification layers.
- **Temporal Extension**: The Scene Planner can be extended into a 3D/Temporal Planner, creating sequential contracts for video generation or storyboard continuity.
