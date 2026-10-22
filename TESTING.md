# How to Run and Test FrameGenX (step by step, zero assumptions)

For optional PyTorch/PEFT QLoRA training, dataset checks, checkpoint resume and
paired evaluation, use [Local constraint-adapter training](docs/15_CONSTRAINT_TRAINING.md).
Measured results and remaining gaps are in the
[training assessment](reports/ml_training_2026-10-15.md).

For the CPU I-HallA/VSM notebook, pinned assets, research API/UI and scoped
evaluation, use [Research experiments](docs/17_RESEARCH_EXPERIMENTS.md).

For the explicit SDXL-base regional controller and local count/color checks,
use [Instance control and local verification](docs/16_CONSTRAINT_CONTROL.md).
It includes an offline CLI example with a supplied contract and records
unknown evidence separately from passed checks.

Everything below is typed into **PowerShell**, inside the project folder.
Open PowerShell and run this first, every session:

```powershell
cd E:\BrosKi\framegenx
```

First time on a fresh machine? Do the "one-time setup" block in
[README.md](README.md) first (create venv, install requirements, spaCy
model, build the frontend).

> **Only one GPU job at a time.** If a benchmark is running (check with the
> Part 5 status command), stick to Part 1, Part 4 and Part 5. Parts 2-3
> generate images and will fight the benchmark for the 6GB GPU — both lose.
>
> Run broad CPU tests and training separately on the 16GB host. Concurrent tests
> overlapped an observed research memory-guard failure.

---

## Part 1 — CPU regression tests (run separately from training)

Install `requirements-research.txt` as well to include the optional notebook and
research runtime checks; unavailable optional notebook dependencies are reported
as skipped tests.

```powershell
venv\Scripts\python.exe -m pytest tests/ --ignore=tests/pressure_test.py -q --basetemp .pytest_tmp_unit
```

**Expected output (last line):** a passing test count — the suite covers reward
math, prompt composition, corrective edits (+ repair_policy fragments),
DB migration, extractor backstops (uncountable stoplist, plural rule),
retrieval fusion gates, planner layout rules, and the diagnostic plugins.

What it proves: the behaviors covered by these tests pass in the current local
environment. Browser journeys, image quality, GPU workloads, external providers
and hosted behavior require separate evidence.

---

## Part 2 — Generate ONE image and watch it verify itself (GPU, ~2-4 min)

```powershell
venv\Scripts\python.exe -m src.cli generate "three red apples on a table"
```

**What you'll see, in order:**
1. `[intent] extracting contract...` — the prompt becomes a checklist (3 apples + 1 table)
2. `[retrieval] 3 hits: [...]` — real COCO captions ground the prompt
3. `[generate] attempt 0` — SDXL-Turbo draws it (~30-60s)
4. `[verify] hard=... clip_norm=... vqa=... R=...` — DINO counts apples, CLIP scores alignment, and an external vision provider supplies VQA when available
5. Either `[OK] accepted` for the hard checks, or `[REPAIR 1/3]` with a
   recorded corrective edit and another generation attempt

**Then look at the evidence.** The last line prints a run folder. Open it:

```powershell
explorer runs
```

Newest folder contains:
| File | What it is |
|---|---|
| `generated_0.png` (and `_1`, `_2`...) | the image per attempt |
| `contract_v2.json` | the checklist extracted from your prompt |
| `global_mask_0.png` | planned boxes for multi-object scenes; the default generator does not use ControlNet |
| `verification_0.json` | the scores (counts found, CLIP, VQA, reward) |
| `decision_0.json` | accept/retry + the corrective prompt it wrote |
| `reward_history.json` | reward per attempt — the self-correction trail |

---

## Part 3 — The web app (ONE CLICK)

Two ways to run the stack:

**Native (daily driver, fastest):** double-click **`run_demo.bat`** — native
CUDA, no virtualization tax. What happens:

1. ONE black window opens — that's the server (API + web page in one
   process). Its log scrolls there.
2. The browser opens **by itself** as soon as the server is ready
   (usually 5–15 s — no need to count or refresh).
3. When you're done: **close the black window** — that stops everything.

Double-clicking it again while it's already running doesn't start a
second server — it just re-opens the page. (Node.js is only needed the
first time, if `frontend\dist` doesn't exist yet.)

What you'll see on the page — the "glass box":
- a stage timeline lighting up as the pipeline moves (intent → retrieval →
  plan → adapt → generate → verify → reflect)
- **Step 1** panel: the contract (the checklist extracted from your prompt)
- **Step 2** panel: the real COCO captions it retrieved
- one card **per attempt**: layout mask, generated image, live scores
  (object counts green/red, CLIP, VQA, reward when available), and — if
  rejected — the exact correction it wrote itself for the next attempt
- a final **Verdict** card: first draft vs final image side by side, plus
  the reward-per-attempt bars

No GPU / benchmark still running? Click **"load demo run (2 cats)"** on the
page — it replays the saved self-repair run from disk without generating
anything.

Timing: a fresh prompt takes **~3 min warm natively, ~5.5 min in Docker**
(WSL2 GPU-paravirtualization floor ~44s/step; sequential offload is the
measured-fastest safe mode — see docs/architecture.md). A repeated prompt
returns instantly from cache (check "fresh run" to force a live
generation).

**Docker (reproducible environment):** `docker compose up --build` serves
the same app on :8000 with the GPU reserved — see the README "Run in
Docker" section. Use it when you need the exact pinned environment; use
native when you want speed.

**Optional diagnostic plugins** (off by default): flip
`verification.plugins.scene_comprehension` / `easyocr` to `true` in
config.yaml to add glass-box evidence on the final attempt. They are soft
signals — they never reject a draft.

Quick backend-only check (no frontend needed): open http://127.0.0.1:8000/docs
in a browser — FastAPI's built-in playground.

---

## Part 4 — The proof artifacts (no GPU, just look)

Local evidence you can inspect without a new GPU run:

1. **Recorded hard-check repair:** `runs\4512bf99-e87e-4614-984c-c6d6c95f8b5d\`
   - Attempt 0 recorded 3 cats and 2 walls against a contract for 2 cats
     and 1 wall, then rejected. Attempt 1 recorded 2 cats and 1 wall and
     passed the hard checks. The decisions and detector counts are in the
     corresponding JSON artifacts. These are detector observations, not a
     human confirmation that every visual constraint was satisfied.

---

## Part 5 — Benchmark status + results (safe anytime)

How many attributable, scored benchmark rows exist per configuration:
```powershell
venv\Scripts\python.exe -c "from collections import Counter; from src.core.db import RunTrackerDB; d=RunTrackerDB('runs/framegenx.db'); print(Counter(row[0] for row in d.all_benchmark_rows())); d.close()"
```
Infrastructure errors and interactive runs are excluded. `Counter()` means
no current benchmark rows meet the provenance filter.

After a benchmark session:
```powershell
type ablation.log      # if you captured a log; look for ABLATION COMPLETE or INCOMPLETE
type results.md        # historical snapshot until you regenerate after a controlled run
powercfg /change standby-timeout-ac 30    # turn sleep back on if you disabled it
```

If a run died midway: just re-run it — it resumes, nothing is lost:
```powershell
scripts\run_ablation.bat
```

---

## If something breaks

| Symptom | Fix |
|---|---|
| `CUDA not available` | check `venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` — if `False`, torch is CPU-only: reinstall per the requirements.txt header. If `True` normally but suddenly `False`, reboot (driver hiccup). |
| `Can't find model 'en_core_web_sm'` | `venv\Scripts\python.exe -m spacy download en_core_web_sm` |
| Model download errors | `$env:HF_HOME` controls the model cache location (this machine: `E:\hf_cache`, set as a user env var). Optional elsewhere — default is `~\.cache\huggingface`; just ensure that drive has ~25GB free. |
| Out of memory (GPU) | one generation at a time; the engine auto-drops ControlNet on OOM. Close extra browser tabs — 6GB VRAM is tight. |
| Out of memory (RAM) / python vanishes mid-run | 16GB RAM is tight too: close other apps, reboot if the desktop itself glitched, rerun. |
| Benchmark stuck | `type ablation.log` for the last line; re-running the .bat resumes |
