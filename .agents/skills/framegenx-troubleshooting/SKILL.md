---
name: framegenx-troubleshooting
description: |
  Cheatsheet for debugging common FrameGenX pipeline failures, including VRAM offload crashes, RAG prompt poisoning, corrupted SDXL latents, GAN refinement architecture, and Rule-Based Repair.
---

# FrameGenX Troubleshooting Guide

When debugging the FrameGenX generation pipeline, check the following known failure modes:

## 1. Corrupted, Noisy, or Blocky Images
**Symptom**: The final draft or intermediate steps look like 8-bit glitchy blocks or pure noise.
**Root Cause**: Mismatch between the base model and the sampler steps/CFG.
**Resolution**:
- If `config.yaml` specifies `steps: 4` and `cfg_scale: 0.0`, the `generator` MUST be `stabilityai/sdxl-turbo`.
- If using `stabilityai/stable-diffusion-xl-base-1.0`, you MUST increase `steps` to 25+ and `cfg_scale` to ~7.0.

## 2. RuntimeError: `Expected all tensors to be on the same device / Input type c10::Half vs float`
**Symptom**: Crash during the generation loop when decoding intermediate latents.
**Root Cause**: Under `enable_model_cpu_offload()`, diffusers maintains the VAE in `float32` while latents are in `float16`.
**Resolution**: Inside `engine.py`'s `step_callback`, always cast the latents to the VAE's dtype before decoding:
`latents = latents.to(dtype=next(iter(pipe.vae.parameters())).dtype)`

## 3. Hallucinated / Bizarre Image Content (Prompt Poisoning)
**Symptom**: The image generates things completely unrelated to the user's prompt (e.g., generating airplanes when asked for a sunset).
**Root Cause**: The RAG Retriever (`retriever.py`) is retrieving out-of-domain images from the small (1k) COCO dataset and forcefully fusing their captions into the final prompt.
**Resolution**: Ensure similarity thresholding is active in `query()` and `query_hybrid()`.
- Dense search (Milvus): Filter hits where `distance < 0.25`.
- Sparse search (BM25): Filter hits where `score < 1.0`.

## 4. Understanding Rule-Based Repair Policy
**Architecture**: The `contract_v2.json` contains a `repair_policy` object that maps specific failures (e.g. `duplicate_object`) to actionable prompt edits (e.g. `enforce_single_subject`).
**Implementation**: In `src/reflection/reflector.py`, this is implemented as a dual-layer strategy:
- **Python Override**: The reflector checks `verification_results` against the `repair_policy.mappings`. If a match is found, the mapped action is applied deterministically, overriding the LLM.
- **LLM Injection**: If the LLM is invoked, the `repair_policy` is injected into the LLM system prompt as explicit constraints.

## 5. GAN Refinement (Texture Enhancement)
**Architecture**: Post-acceptance texture enhancement is handled by `src/generation/refine_stage.py`.
**Implementation**: It uses `StableDiffusionXLImg2ImgPipeline` with extremely low strength (0.3) to enhance textures without altering the semantic structure or layout.
**Memory Constraint**: To satisfy 6GB VRAM limits, the refinement stage aggressively unloads the base generation pipeline before running.
