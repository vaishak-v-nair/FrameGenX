# FrameGenX

Contract-driven text-to-image generation with first-class verification.

**Generate → Verify → Repair → Accept.** The default `hard_only` decision
checks object counts and spatial relations. CLIP alignment and vision QA for
attributes are recorded as diagnostic evidence; they do not gate acceptance.

> **New here? Start with [TESTING.md](TESTING.md)** — a zero-assumptions,
> copy-paste walkthrough of running and testing everything.

Optional ML: [local constraint-adapter training](docs/15_CONSTRAINT_TRAINING.md)
uses PyTorch, PEFT QLoRA and MLflow. The
[three-source pilot report](reports/ml_training_2026-10-15.md) records actual
training and its limits. Hallucination reduction is not yet established;
the trained candidate remains experimental. The optional
[instance-control profile](docs/16_CONSTRAINT_CONTROL.md) adds separate object
denoising paths, local segmented-color checks and bounded repair. Its
[measured comparison](reports/constraint_reduction_2026-10-15.md) distinguishes
controller outcomes from trained-adapter quality.

The [I-HallA / VSM research profile](docs/17_RESEARCH_EXPERIMENTS.md) adds a
Jupyter-driven CPU ML/DL experiment and read-only research evidence in Benchmarks.
Its Cards checkpoints remain experimental and inactive in SDXL generation.
The [measured Cards study](reports/ihalla_vsm_research_2026-10-17.md) recorded
0/32 valid outputs in both arms; no hallucination reduction was demonstrated.

## How it works (plain English)

Most image generators draw once and hand you whatever came out. FrameGenX
treats your prompt as a **contract** — "three red apples on a table" means
*exactly three apples, red, and a table* — and records which parts of that
contract its current verifiers can establish:

1. It reads your prompt and writes down a checklist (3× apple [red], 1× table).
2. It retrieves similar real photo captions as inspectable evidence.
3. It plans WHERE each object goes (layout boxes) and composes the final prompt.
4. It draws the image (SDXL-Turbo, 4 steps, on a 6GB laptop GPU).
5. It checks its own work: an object detector counts the apples, CLIP scores
   alignment with the original prompt, and an external vision model attempts
   attribute questions such as "what color is the apple?". If that model is
   unavailable, the VQA score and composite reward are reported unavailable.
6. If a hard check fails, it writes itself a corrective prompt ("exactly 3 apples,
   each clearly separated") and tries again — up to 3 repairs.

Every step leaves evidence on disk in `runs/<id>/` — the checklist, the layout
mask, each attempt's image, each attempt's scores, and the accept/reject
decision. Nothing is taken on faith.

## Architecture — config-driven stage graph

```
config.yaml: pipeline.stages: [intent, retrieval, plan, adapt, generate, verify, reflect, refine]

 prompt ──> INTENT ──> RETRIEVAL ──┐  (once)
                                   ▼
        ┌────────── repair loop (max 3) ──────────────────────────┐
        │  PLAN ──> ADAPT ──> GENERATE ──> VERIFY ──> REFLECT     │
        │  masks    LoRA      SDXL-Turbo   DINO count  hard pass? │
        │  prompt   inject    512px, 4stp   CLIP score  corrective │
        │  compose  (once)                 vision QA   edits ──────┘
        └─────────────────────────────────────────────────────────┘
                                   ▼
                     accepted image + full artifact trail (runs/<id>/)
```

| Stage | Module | What it does |
|---|---|---|
| intent | `src/intent/` | LLM extraction with deterministic backstops → ContractV2 (entities, counts, colors) |
| retrieval | `src/retrieval/` | local hybrid caption index (COCO val2017) → inspection evidence |
| plan | `src/scene/` | layout masks + original prompt with corrective edits |
| adapt | `src/generation/lora.py` | plug-and-play LoRA injection (pre-offload hook) |
| generate | `src/generation/` | SDXL-Turbo, 512px, 4 steps, cfg 0; planned boxes steer regional attention; ControlNet is opt-in |
| verify | `src/verification/` | GroundingDINO count check + spatial geometry + CLIP score + external vision QA |
| reflect | `src/reflection/` | hard count/spatial gate by default; optional reward gate; corrective prompt edits |

Every stage is a config toggle — the ablation study is literally
`benchmark.py --config baseline|rag|rag_lora|full`.

## Honest numbers

The historical 15-prompt snapshot in [results.md](results.md) predates
explicit benchmark provenance in the run database and cannot validate a
current comparison. Run `benchmark.py` with recorded seeds and regenerate
the table using `scripts/make_results.py`. Its acceptance column is the
object-count/spatial pass rate, not satisfaction of every contract constraint;
attribute VQA remains diagnostic. The table also records CLIP norm, VQA,
reward, attempts, and latency when available.

**Paper-reported values** (IJRASET, DOI 10.22214/ijraset.2025.76265: FID 7.2,
CLIP 0.63, VQA 0.68, 81% human preference) come from a different evaluation
scale and protocol. They are reported for provenance, **not** reproduced by
this repo's benchmark. Where the two disagree, results.md is what this code
actually does.

Known scale caveats: 1k-caption retrieval index (architecture supports larger
stores), 15-prompt benchmark subset (full 30: rerun `benchmark.py` per config
with `--limit 0`), single seed, reward threshold τ calibrated by a small pilot
(`scripts/tau_pilot.py`).

## Run it

```powershell
# one-time setup (PowerShell, from the repo root)
python -m venv venv                                          # needs Python ≥3.13
# If `python` isn't on PATH, use the full path, e.g.: C:\Python313\python.exe -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt   # CUDA torch — see requirements.txt header
venv\Scripts\python.exe -m spacy download en_core_web_sm     # optional legacy spatial parser utility
venv\Scripts\python.exe scripts\build_index.py --n 1000      # retrieval index (downloads COCO captions)
cd frontend && npm install && npm run build && cd ..         # web UI (needs Node.js)

# demo (one click): starts API + web UI + opens browser
run_demo.bat

# generate (CLI)
venv\Scripts\python.exe -m src.cli generate "three red apples on a table"
```

For cached offline execution, set `HF_HOME` to the cache root and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. The generator resolves the cached
model marker to its snapshot directory; it does not require unrelated README
or alternate-precision weights to be downloaded. Missing required model files
still fail loading. Intent extraction and vision QA use configured providers;
these cache settings only control Hugging Face model loading.

## Run in Docker (GPU)

Docker is the **reproducible environment** path — identical pinned deps on
any machine. For daily driving, prefer native (`run_demo.bat`): native CUDA
runs ~2x faster (no WSL2 GPU-paravirtualization tax).

Requires Docker Desktop with WSL2 backend and an NVIDIA GPU. The container
matches the tested environment exactly: Python 3.14 + torch 2.13.0+cu130,
SDXL-Turbo at 512px with sequential CPU offload + attention slicing.

```powershell
docker compose up --build          # API + UI on :8000, GPU reserved
# UI: http://localhost:8000/app  |  Docs: http://localhost:8000/docs
docker compose logs -f             # watch the pipeline stages
```

Notes:
- Models load from the host HF cache (`HF_HOME`, default `E:\hf_cache`) via a
  bind mount — no downloads inside the container.
- Run artifacts land in `./runs` on the host; secrets come from `.env` at
  runtime and are never baked into the image.
- `PYTORCH_CUDA_ALLOC_CONF` is overridden to the classic caching allocator
  inside the container only: expandable segments use CUDA VMM APIs that are
  unstable through WSL2 GPU paravirtualization (see docker-compose.yml).
- Earlier RTX 3050 Laptop (6GB) measurements were roughly **5.5 min per
  accepted run** in Docker with refinement. Current timings need a fresh
  measurement. Refinement is now disabled by default because the displayed
  draft is the verified artifact and `refined_final.png` is not reverified.
  Native execution (`run_demo.bat` / venv) avoids the measured WSL2 GPU
  paravirtualization overhead.

```powershell
# API + frontend, started by hand
venv\Scripts\python.exe -m src.api          # FastAPI :8000
cd frontend && npm run dev                   # React UI (glass-box view)

# benchmark (overnight; resumable; one process per config)
powercfg /change standby-timeout-ac 0
scripts\run_ablation.bat

# tests
venv\Scripts\python.exe -m pytest tests/ --ignore=tests/pressure_test.py -q --basetemp .pytest_tmp_unit
```

## Verification-first philosophy

Success metric is **constraint satisfaction rate**, not aesthetics. The system
never claims hallucination *elimination* — only measured reduction, evidenced
by the ablation table. See `AGENTS.md` for the design rules.
