# Local constraint-adapter training

## Evidence and architecture

The optional stack trains SDXL attention adapters with PyTorch, Hugging Face
PEFT, NF4 QLoRA and local MLflow. The first pilot trained on 90 images from
three sources within the 6 GB GPU budget. **Visual hallucination reduction
has not been established.** See the [measured report](../reports/ml_training_2026-10-15.md).
Training completion and lower denoising loss cannot qualify an adapter.

The [expanded corpus report](../reports/datasets_2026-10-17.md) records the new
700-image download and held-out I-HallA prefix. The separate
[expanded study protocol](plans/EXPANDED_TRAINING_PROTOCOL_2026-10-17.md) freezes
the 256 px optimization feasibility study before fitting. Use
`output/jupyter-notebook/expanded_constraint_training.ipynb` with the local kernel;
its final execution outputs and the study journal establish whether it completed.

| Module | Responsibility |
|---|---|
| `src/training/prepare.py`, `data.py` | Bounded acquisition, provenance, licenses, image hashes and grouped splits |
| `cache.py` | Frozen text embeddings and deterministic VAE latents saved to disk |
| `model.py`, `objective.py` | Frozen NF4 backbone, FP32 adapters and Min-SNR denoising loss |
| `train.py`, `export.py` | Optimization, complete checkpoints and Diffusers export metadata |
| `tracking.py` | Explicit local MLflow lifecycle, metrics and artifacts |
| `evaluation.py`, `scripts/evaluate_adapter.py` | Paired generation and checked, durable resume |
| `scripts/review_adapter.py`, `registry.py` | Abstaining diagnostics and experimental activation gates |

Text encoders and VAE run sequentially and leave GPU memory before UNet
training. Frozen embeddings are cached on disk, following the
[Diffusers SDXL training guide](https://huggingface.co/docs/diffusers/en/training/sdxl).
Full-image letterboxing preserves objects and spatial labels: no random crop
or horizontal flip. Token truncation is rejected. VAE encoding/decoding uses
FP32 and rejects nonfinite output; evaluation also rejects blank images.

## Data contract

The pilot contains CLEVR symbolic scenes, VSR validated spatial statements,
and COCO instance counts. Records contain source/revision, image SHA256,
attribution/license metadata, caption, constraints and split. Natural photos
are filtered to image license ID 4 (CC BY 2.0), with original URLs retained.
Preserve the supplied notices; the filter does not establish every downstream right.

Hash groups are isolated across the whole manifest. One image cannot cross
train/validation/test, even with another caption or source. Only positive
records enter the generative cache; optimization selects only the train split.

New validation also compares decoded RGB pixels, catching exact duplicates with
different encoding/metadata. This does not detect every near-duplicate. Expansion
reserves prior pilot image assignments; a new seed cannot recycle held-out images
into training. Test images are excluded from the training cache entirely.
False VSR statements remain
evaluation material. Test images never enter training or checkpoint selection.

## Run locally (PowerShell, repository root)

All model work requires the common process-safe device lease before CUDA. The
pipeline, training, preflight, evaluation and local verifier commands acquire it;
low-level model loaders require an existing lease. Separate checkouts/workers of
the same OS user share the conservative device gate. Unrelated applications do
not participate. A competing operation receives an explicit busy error; a failed
resource cleanup keeps that process unavailable until restart.

For the new notebook, set `HF_HOME` to the actual existing cache location. This
machine uses `E:\hf_cache`. The runner disables dotenv loading and never prints
credentials; therefore a cache path configured only inside `.env` must be supplied
explicitly. The initial probe with no cache location failed and is retained as
`reports/gpu_preflight_cache_configuration_failure_2026-10-17.json`. The corrected
512 px NF4 backward probe passed at 2.460 GB peak reserved memory.

```powershell
$env:HF_HOME = 'E:\hf_cache' # replace with your cached-model location
.\venv\Scripts\python.exe scripts/run_research_notebook.py --notebook output/jupyter-notebook/expanded_constraint_training.ipynb
```

The expanded notebook uses an immutable study directory. Identical completed
reruns verify result/source/checkpoint hashes and reuse evidence. Changed inputs,
incomplete attempts and corrupt evidence require deliberate inspection/recovery;
they do not silently trigger a new fitting run or reuse unrelated weights.

Install the pinned application environment first. Training requires a cached
backbone and never downloads one implicitly. Dataset acquisition uses ordinary
HTTPS with TLS verification; Hugging Face offline flags apply to model loading.

```powershell
& .\venv\Scripts\python.exe -m pip install -r requirements-ml.txt
& .\venv\Scripts\python.exe -m pip check
$env:HF_HOME='E:\hf_cache'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:MLFLOW_DISABLE_AGENT_HINT='1'

# Real forward/backward/optimizer probe on synthetic inputs; no quality claim.
& .\venv\Scripts\python.exe -m src.training.preflight --output reports/ml_preflight_local.json
& .\venv\Scripts\python.exe -m src.training.prepare --per-source 36 --seed 42
& .\venv\Scripts\python.exe -m src.training.train --config configs/training_pilot.yaml
```

Run one GPU workload at a time. Defaults: batch 1, accumulation 2, rank/alpha 4,
learning rate 1e-5, 48 optimizer steps, Min-SNR gamma 5, 512px. Guards require
3 GB free GPU memory and cap measured reservation at 5.5 GB. The pilot reserved
2.363 GB; the strict deterministic backward probe reserved 2.483 GB. These
are PyTorch process measurements, excluding display/driver use. Plain SDXL
LoRA is blocked on this 6 GB profile; QLoRA is the validated path.

For another acquisition size/seed, use a new output directory and config.
Acquisition is bounded to 10–100 positive images per source. More data alone
is not a quality gate: validate labels and keep an independent grouped test set.

## Resume and reproducibility

Checkpoints include optimizer/scaler, adapter weights, sampler order/cursor,
noise generator, CPU/CUDA RNG states, and model/data/code/profile fingerprints.
Current runs preserve executed source and dataset records. To extend a run,
copy its config, increase `max_steps`, then use a completed checkpoint:

```powershell
# Create training_extended.yaml first; RUN_ID is the actual run directory.
& .\venv\Scripts\python.exe -m src.training.train --config configs/training_extended.yaml --resume runs/training/RUN_ID/checkpoint-000048
```

The original 48-step pilot predates strict determinism; its executed source
is retained. Current code intentionally refuses to resume it under changed
fingerprints. A separate real CUDA test compared 2 steps + resume to step 3
with an uninterrupted 3-step run: all 1,120 adapter tensors matched exactly.
This proves that bounded control on the tested stack. It does not promise
cross-platform equality; see [PyTorch's reproducibility limits](https://docs.pytorch.org/docs/main/notes/randomness.html).

## Evaluation and experimental activation

```powershell
& .\venv\Scripts\python.exe scripts/evaluate_adapter.py --checkpoint runs/training/RUN_ID/checkpoint-000048 --resume
& .\venv\Scripts\python.exe scripts/review_adapter.py --evaluation runs/training/RUN_ID/EVALUATION_DIRECTORY
```

The evaluator uses real Diffusers SDXL sampling and the application LoRA hook,
NF4 baseline versus exported candidate, identical seeds, 25 steps and CFG 7.
It saves latents/images, hashes, settings, packages, source snapshots and progress.
Resume verifies experiment identity, hashes, tensor shape, and finite values.
Changed fingerprints get separate directories. Interrupted/failed states never
become generated successes. Four prompts × two seeds is a diagnostic pilot.

Vision review shuffles arm order and rejects malformed answers. Unavailable
judges yield unknown requirements and null metrics. Scores remain uncalibrated.
Qualification requires complete independent full-contract evidence, zero
unverified requirements, at least 50 image groups, three seeds, and a strictly
positive lower bound on the paired CSR improvement's 95% confidence interval.

Trained adapters require a matching base manifest and weight hash. Experimental
loading requires `kind: trained` and `allow_unqualified: true`. This candidate
belongs to `stabilityai/stable-diffusion-xl-base-1.0` with 25+ steps/CFG near 7;
it cannot be attached to the four-step Turbo backend. It remains unqualified
and is not enabled in the default application.

MLflow uses `runs/mlflow.db` and `runs/mlflow_artifacts/` locally. `FINISHED`
means optimization completed; `quality_status=unverified` remains explicit.
Failures use `FAILED`, interruptions use `KILLED`. There is no remote autologging,
model publication or automatic promotion. `mlflow-skinny` provides the tracking
client; a web UI is not a guaranteed installed component.

The local tracking integration test requires `requirements-ml.txt`; it is
explicitly skipped when that optional package is absent. The full measured
regression includes it with MLflow installed.
