# Expanded constraint data: preregistered optimization feasibility study

Frozen before new fitting. This is a compatible SDXL PEFT/NF4 study; the prior
unconditional RGB VSM checkpoint is not installed into SDXL. Its negative
0/32-versus-0/32 result remains unchanged.

## Question and limits

Can the existing bounded adapter workflow process a larger, attributed corpus,
cover every training image in a deterministic epoch, preserve held-out images
and finish finite optimization within this laptop's resource budget?

This is an optimization and execution study. It does not establish hallucination
reduction, full-contract CSR, generalization at 512 px or adapter qualification.
The 256 px training resolution reduces compute and preserves full-image geometry
through letterboxing. Its fidelity and inference resolution transfer remain
empirical questions, not assumed improvements.

## Frozen profile

- `configs/training_expanded.yaml`: SDXL base 1.0, NF4 double quantization,
  rank/alpha 4, attention projections, learning rate 1e-5, Min-SNR gamma 5.
- 256 px, batch 1 with accumulation 2, 320 optimizer steps, seed 1729.
- Target acquisition: 100 complete CLEVR 3-4-object scenes, 300 licensed COCO
  count images, 300 consensus-positive VSR images. False VSR statements are
  evaluation-only records and never optimizer targets.
- Prior pilot image splits remain reserved. Exact byte and decoded RGB image
  leakage are checked before fitting. The actual dataset SHA and split counts
  are frozen in the run manifest, rather than guessed before download finishes.
- Test images and all negatives are excluded from frozen training caches.
  Training uses only the train split. Validation selection is deterministic,
  source-balanced, at most 12 images; the same fixed noise seeds are used before
  and after fitting. Validation never selects a favorite checkpoint.
- Checkpoint and validation every 80 steps. Keep the final planned step, or
  explicitly record failure/interruption. No hidden extra epochs or rerolling.
- GPU reserved/allocated <=5.5 GB, initial GPU free >=3.0 GB. Process RSS <=7 GB
  and host available >=1 GB sampled at preparation/optimization checkpoints.
  Optimization deadline 1800 s, total checked deadline 2700 s. These are checked
  budgets, not OS enforcement or continuous peak-memory guarantees.
- One shared device lease starts before CUDA probing, held through cleanup.
  Do not overlap the run with broad test suites, builds or other model workloads.

## Evidence and acceptance of execution

Record exact source snapshot including device/file helpers, package versions,
resolved backbone identity/hash, dataset SHA, cache identity, selected validation
IDs/noise seeds, finite losses/gradients, optimizer changes, all train image IDs
seen, source presentations, SafeTensors checkpoints, local MLflow lifecycle and
sampled host/GPU memory. A checkpoint retains optimizer/scaler/RNG/sampler state
for an explicit compatible resume. No automatic restart after failure.

Successful execution requires all 320 steps, finite changed adapter tensors,
intact artifacts and an honest train-image coverage count. A completed study with
missing images is reported incomplete for the coverage objective. Decreased
validation denoising MSE is a diagnostic and never substitutes for CSR.

Adapter status stays **unqualified / unverified** and default activation stays
off. A separate, disjoint paired image study, reviewed requirement extraction,
human-calibrated judges and the approved all-gates qualification protocol are
needed before any reduction or promotion claim. I-HallA is held out, not trained
on, and no paid VQA calls are authorized by this protocol.

## Primary research references

- [Diffusers SDXL training](https://huggingface.co/docs/diffusers/training/sdxl)
  supports mixed precision, gradient accumulation/checkpointing and caching
  frozen embeddings; the existing local NF4 implementation is separately tested.
- [Min-SNR weighting](https://arxiv.org/abs/2303.09556) motivates the existing
  objective. No paper speedup number is claimed for this hardware.
- [Noise schedule and sampling mismatch analysis](https://arxiv.org/abs/2305.08891)
  supports reviewing parameterization and terminal SNR for a future scratch
  diffusion study. Changing pretrained SDXL's prediction type/scheduler without
  compatible fitting is excluded here.
- [VSM source](https://github.com/bhosalems/VSM) requires a compatible learned
  variance output/sampler. That separate integration remains gated.
