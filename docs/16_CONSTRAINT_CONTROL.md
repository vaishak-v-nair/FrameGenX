# Instance control and local verification

This optional path addresses repeated-object counts, missing objects and
chromatic color binding using an explicit contract, regional denoising paths,
and independent detector/segmentation models. It does not establish general
hallucination elimination or qualify the earlier trained LoRA.

The default Turbo configuration remains separate. The explicit controller
profile uses SDXL base, NF4, 512px, 25 Euler steps and CFG 7. It currently
supports 2–6 explicitly counted foreground instances, simple background
surfaces, and unambiguous left/right relations. Contact/depth layouts and
unknown foreground counts require another controller. Unsupported requests
fail before loading the dense generation pipeline. Explicit foreground boxes
or canvas positions also require another controller; they are never silently
replaced by the generated row layout.

## How it works

1. Compile immutable contract entities into separate noun/attribute prompts
   and disjoint planned regions. Background surfaces have their own prompt.
2. During early denoising, replace non-object areas with encoded neutral
   backgrounds. Temporarily shift each object region to the image center,
   run its denoiser, and invert that shift before combining predictions.
3. Fuse predictions by mask and advance Euler once per timestep. Batches
   contain at most two regions; conditioning and latents stay within the
   configured GPU budget. A later global phase uses the original prompt.
4. Release generation models before verification. GroundingDINO detects
   candidate objects; pinned SAM2-tiny segments their actual pixels.
5. Suppress part/whole duplicates when high-quality masks agree and their
   detector boxes overlap by containment. Preserve raw counts and masks.
   The expected count never determines which detections survive.
6. Measure chromatic hue inside eroded object masks. Weak, mixed, achromatic
   and ambiguous same-class color binding remain unknown. Clear contradictions
   block acceptance. The explicit profile also blocks unknown required
   chromatic colors. An unavailable local verifier blocks acceptance.
7. Bounded repair uses new recorded seeds. It cannot repeat the exact same
   failed generation while claiming to have applied a repair.

This implements adaptations of [MultiDiffusion](https://multidiffusion.github.io/)
and [SemanticDraw's mask-centering idea](https://github.com/ironjr/semantic-draw)
for the existing NF4 SDXL/Euler stack. It is not a reproduction of their
published benchmarks. The uncontrolled sampler is checked against actual
Diffusers latents before interpreting experimental controller results.

## Setup and run (PowerShell)

Install the application's pinned requirements and the optional
[training stack](15_CONSTRAINT_TRAINING.md). SDXL base must already be cached.
The local color/count refiner adds a 155.9 MB
[SAM2-tiny checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny), pinned
to revision `de431c4043854a71d8101e17995dfe596bf101a5` and checked by SHA256.

```powershell
$env:HF_HOME='E:\hf_cache'
$env:HF_HUB_OFFLINE='0'
& .\venv\Scripts\python.exe scripts/cache_instance_verifier.py

$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:MLFLOW_DISABLE_AGENT_HINT='1'
& .\venv\Scripts\python.exe -m src.cli generate "Exactly two green bottles on a white table. No red bottle." --config configs/constraint_control.yaml --contract configs/examples/green_bottles_contract.json --seed 43
```

`--contract` preserves supplied requirements and bypasses intent extraction.
Its prompt must equal the requested prompt and its entities must validate.
Supplied contracts use a distinct cache key and cannot enter the ordinary
prompt-only cache or an existing benchmark suite. Without this option, the
configured intent provider still extracts the contract.
The explicit profile disables optional failure-memory writes, so repairs do
not initialize a retrieval database or an additional embedding model.

Run one GPU job at a time. The reference profile caps measured PyTorch GPU
reservation at 5.5 GB; the development fusion runs reserved about 2.582 GB.
Its 25 steps and multiple regions cost more than the default four-step Turbo
path. Available memory and measured runtime are recorded in generation
metadata. This is not a promise about total Windows display/driver memory.

## Evidence and outcomes

Normal run directories contain the original contract, source record, actual
region plan, seed/sampler metadata, latent/image hashes, generated images,
raw/refined counts, object masks, color measurements and decisions. The UI
layout diagram uses the same boxes as the controller.

Local results are operational diagnostics. Positive hue measurements are not
silently promoted to calibrated full-contract evidence. Achromatic surfaces,
negation coverage, instance identity and extraction recall still need stronger
validation. CLI output distinguishes an operational pass from requirements
that need review. Failed runs exit nonzero and preserve their attempts.

Reproduce research comparisons with `scripts/evaluate_regions.py` and inspect
saved images with `scripts/verify_local.py`. Every attempted strategy,
including failed attention-only and uncentered variants, remains in its
fingerprinted run directory. Read the
[constraint-control assessment](../reports/constraint_reduction_2026-10-15.md)
for the measured comparison and its limits.
