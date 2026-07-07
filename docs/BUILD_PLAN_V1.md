# FrameGenX Build Plan v1

This plan maps the shortest path to a working MVP: achieving Count Consistency and demonstrating a functioning Autonomous Repair Loop within a 6GB VRAM constraint.

## Phase 1: Repository Cleanup & Foundation
- **Deliverables**: Removal of legacy notebook files, implementation of `RunContext` state management, centralized configuration (`config.yaml`), setup `pytest`.
- **Dependencies**: None.
- **Risks**: Breaking the fragile `pipeline.py` during refactor.
- **Estimated Complexity**: Low.
- **Acceptance Criteria**: `pytest` runs successfully, configuration loads properly, memory management utilities are robust.

## Phase 2: Core Architecture Setup (Tracking & State)
- **Deliverables**: SQLite run tracking database, intermediate artifact persistence engine.
- **Dependencies**: Phase 1.
- **Risks**: High disk I/O overhead.
- **Estimated Complexity**: Low.
- **Acceptance Criteria**: A pipeline run logs its start, end, and metadata to `runs.db`.

## Phase 3: Contract Engine (The Source of Truth)
- **Deliverables**: Strict Pydantic models for `ContractV2`. Refactored `IntentExtractor` to parse a natural language prompt into the strict Pydantic model.
- **Dependencies**: Phase 2.
- **Risks**: NLP extraction is probabilistic; parsing failures.
- **Estimated Complexity**: Medium.
- **Acceptance Criteria**: The engine can take "3 red apples and 2 blue cars" and output a perfectly typed JSON contract with precise counts.

## Phase 4: Scene Planning
- **Deliverables**: Mathematical translation of `ContractV2` objects into 2D geometric bounding boxes. Generation of synthetic segmentation masks (black background, white boxes) to feed into ControlNet.
- **Dependencies**: Phase 3.
- **Risks**: Bounding box overlap causing ControlNet confusion.
- **Estimated Complexity**: High.
- **Acceptance Criteria**: Given a contract, the planner outputs a directory of PNG masks perfectly aligned with the requested spatial layout.

## Phase 5: Generation Engine (The Latent Drafter)
- **Deliverables**: A unified generation pass using a lightweight diffusion backbone (e.g., SD1.5 or optimized SDXL-Turbo) + ControlNet (Segmentation/Canny) to force the layout.
- **Dependencies**: Phase 4.
- **Risks**: 6GB VRAM OOM when loading both base model and ControlNet.
- **Estimated Complexity**: High.
- **Acceptance Criteria**: The generator produces an image strictly adhering to the synthetic masks provided by the Scene Planner, without OOM errors.

## Phase 6: Verification Engine (The QA Layer)
- **Deliverables**: Integration of GroundingDINO (or optimized alternative) to detect and count objects in the generated image. Comparison logic against `ContractV2`.
- **Dependencies**: Phase 5.
- **Risks**: Verification model hallucinating or missing objects (false negatives), skewing the benchmark.
- **Estimated Complexity**: Medium.
- **Acceptance Criteria**: The QA engine accurately reports "Failed: Expected 3 apples, found 2".

## Phase 7: Benchmark Framework
- **Deliverables**: `benchmark.py` script. A static JSON dataset of 50 hard numeracy prompts. Automated execution and CSR calculation.
- **Dependencies**: Phase 6.
- **Risks**: Long runtime.
- **Estimated Complexity**: Medium.
- **Acceptance Criteria**: Running `python benchmark.py` outputs a final Constraint Satisfaction Score across the dataset.

## Phase 8: Repair Loops (The Autonomous Compiler)
- **Deliverables**: A feedback loop in `pipeline.py`. If QA fails, the discrepancy report is fed to the Scene Planner. The Scene Planner adjusts bounding box sizes, spacing, or control weights, and re-triggers Generation.
- **Dependencies**: Phases 4, 5, 6.
- **Risks**: Infinite loops. Must be capped at `max_retries=3`.
- **Estimated Complexity**: Very High.
- **Acceptance Criteria**: A prompt initially failing with 2 apples mathematically adjusts its masks and succeeds with 3 apples on the second attempt, entirely autonomously.

## Phase 9: Production Optimization
- **Deliverables**: Code profiling, VRAM memory swapping optimization, packaging as a clean CLI and API.
- **Dependencies**: Phase 8.
- **Risks**: Platform-specific bugs (Windows/Linux).
- **Estimated Complexity**: Low.
- **Acceptance Criteria**: FrameGenX can be pip-installed and run via `framegenx generate "prompt"`.
