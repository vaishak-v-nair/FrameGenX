# Historical implementation audit

> This undated legacy audit predates the current pipeline. Its descriptions of
> missing repair loops, benchmarks, configuration, SQLite and tests, and its
> references to a YOLO verifier, are superseded. For current behavior use the
> [README](../README.md), [architecture](architecture.md),
> [approved task board](../reports/planning/EXECUTION_TASKS.md) and the dated
> execution reports. Preserved below as historical design context.

## Overview
FrameGenX has been successfully migrated from a series of monolithic Jupyter Notebooks (N0-N7) into a modular Python package (`src/`). The overarching sequential flow (Intent -> Scene -> Generation -> Control -> QA -> Reporting) exists structurally.

## 1. What Already Works (To Preserve)
- **Modular Pipeline Skeleton**: `src/pipeline.py` correctly orchestrates the sequential execution of the system.
- **VRAM Constraints (Basic)**: The sequential loading of models and the `free_memory()` utility successfully prevent parallel VRAM exhaustion, respecting the 6GB limit.
- **Separation of Concerns**: The distinct modules (`intent/`, `scene/`, `generation/`, `control/`, `verification/`, `reporting/`) correctly enforce the boundaries required by the architecture.

## 2. What is Partially Implemented (To Rewrite/Refine)
- **The Contract Engine (`src/intent/extractor.py`)**: Currently relies on lightweight NLP (SpaCy, Sentence-Transformers). This is insufficient for generating the strict, deeply nested `contract_v2.json` schema required for deterministic layout planning. It needs to be upgraded to a structured output engine (e.g., using a local LLM or strict API).
- **Scene Planning (`src/scene/planner.py`)**: Currently a stub. It must be heavily rewritten to mathematically translate the `contract_v2.json` bounding boxes into pixel-perfect 2D control masks (e.g., generating black/white segmentation masks for ControlNet).
- **Generation & Enforcement (`src/generation/`, `src/control/`)**: Splitting Draft (SDXL) and Enforcement (ControlNet) into two separate steps (N3 and N4) might be unnecessary tech debt. We should evaluate generating the image directly with ControlNets in a single pass to save VRAM and time, provided it adheres to the contract.
- **Verification (`src/verification/qa.py`)**: Implements basic object detection (YOLOv8). Needs to transition to a more robust, zero-shot open-vocabulary detector like GroundingDINO for exact count verification.

## 3. What is Missing (To Build)
- **Autonomous Repair Loops**: `pipeline.py` currently prints a rejection message and stops. It lacks the logic to feed the QA decision back into the Scene Planner and mathematically adjust the layout before triggering a re-generation.
- **Benchmark Engine**: There is no automated framework to run a dataset of prompts and calculate the Constraint Satisfaction Rate (CSR).
- **Run Tracking & Persistence**: Missing SQLite integration for tracking generation runs, latency, and success rates.
- **Configuration Management**: Missing structured configuration for model paths, device selection, and inference steps.
- **Unit Tests**: The repository currently lacks automated testing.

## 4. What Should Be Removed
- **Redundant Refinement (N5)**: The notebook N5 (Refinement/Upscaling) should be disabled or removed for the MVP. It introduces unnecessary VRAM overhead. The MVP focuses on *Count Consistency*, not high-resolution aesthetics.

## 5. Architectural Weaknesses
- **Error Handling**: The pipeline assumes perfect execution from step to step. If the Intent Extractor hallucinates an invalid JSON, the Scene Planner will crash.
- **Disk I/O Bottleneck**: Saving intermediate JSONs and PNGs to disk at every step (`output/`) is good for debugging, but we need a formalized `RunContext` object to pass data in memory to avoid excessive disk I/O, while optionally saving artifacts to an SQLite blob or structured directory.
