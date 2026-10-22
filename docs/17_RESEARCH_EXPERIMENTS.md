# I-HallA and VSM research experiments

This optional profile connects reproducible CPU research to FrameGenX's existing
FastAPI and React application. It does not qualify a generator. Read the
[planning and execution task board](../reports/planning/EXECUTION_TASKS.md) and
[reviewed protocol](plans/RESEARCH_TRAINING_EXECUTION_PLAN.md) first.

## What runs

| Component | Work performed | Evidence scope |
| --- | --- | --- |
| Classical ML | StandardScaler and logistic regression on quadrant template features; four frozen synthetic corruption families | Held-out synthetic-corruption diagnostic, uncalibrated for general images |
| PyTorch DL | Compact learned-variance RGB DDPM; equal-budget rho=0 baseline and rho=.1 VSM candidate | VSM Cards CPU feasibility experiment |
| Evaluation | All 40 RGB templates, all four quadrants, all 32 paired sampling seeds; rate intervals and diversity/distance diagnostics | Template validity; full-contract CSR remains null |
| MLflow | Local SQLite run IDs, lifecycle, loss terms and model artifacts | Execution evidence; a FINISHED run is not qualification |
| I-HallA | Bounded inspection of public caption/history-QA workbook metadata | Evaluation unavailable without image assets and permitted VQA; no benchmark training |
| Full stack | Read-only research API and Benchmarks panel | Displays saved experiment status and measured scope |

**2026-10-17 continuation:** a separate acquisition downloaded a held-out
40-image I-HallA prefix plus all five workbooks and resolved captions by their
named science/history worksheets. See the [dataset report](../reports/datasets_2026-10-17.md).
The original Cards run above did not evaluate these images; no I-HallA score or
generator-training permission is inferred from downloading them. Its frozen
negative result remains unchanged as newer source improves metadata handling.

VSM needs a learned variance output and a sampler that consumes it. Its six-channel
RGB Cards checkpoint is incompatible with the current four-latent-channel SDXL
LoRA pipeline. General activation requires a compatible design, resource tests,
disjoint full-contract evaluation and qualification. The existing PEFT/QLoRA
workflow is described separately in [constraint training](15_CONSTRAINT_TRAINING.md).

The paper's inverse-square-root noise schedule is implemented independently; the
pinned reference source's constant-rho variant differs. The bounded tanh variance
mapping is a declared local adaptation. This is not a paper-scale reproduction.

## Run the saved notebook

From `E:\BrosKi\framegenx`, use the repository environment. No provider API key,
CUDA initialization or external model download is required by these cells.

```powershell
# Optional setup in an environment that already has requirements.txt installed.
.\venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\venv\Scripts\python.exe -m ipykernel install --prefix runs/research_jupyter --name framegenx-research --display-name 'FrameGenX CPU Research'

# Execute all cells and save their actual outputs, including execution errors.
.\venv\Scripts\python.exe scripts/run_research_notebook.py
```

The notebook is [ihalla_vsm_training.ipynb](../output/jupyter-notebook/ihalla_vsm_training.ipynb).
It calls tested modules in `src/research/`; training logic is not hidden in cells.
`scripts/create_research_notebook.py` is a maintenance scaffold that clears cell
outputs. Do not run it over an executed notebook you intend to preserve.

The same experiment can be called without Jupyter:

```powershell
.\venv\Scripts\python.exe -m src.research.experiment --config configs/research_cards.json
```

An intact completed run with exactly matching config/data/source/runtime identity is reused
only after artifact hash checks. A deliberate new paired trial uses `--new-run`.
Interrupted attempts restart both arms; model-only checkpoints do not contain an
optimizer/RNG resume state. Keep the original failed attempts as evidence.

Each new experiment copies its hashed research/tracking source files into its
`source/` folder before fitting. Its protocol includes Python, platform and the
versions of the ML, image-decoding, SafeTensors and tracking packages.

The initial protocol is 500 steps/arm, batch 8, four CPU threads, seed 1729 and
sampling seeds 10000–10031. Run training and broad tests/builds **sequentially** on
the 16GB laptop. The memory guard rejects process-tree RSS above 8 GB or available
host RAM below 750 MB; wall limits are 30 minutes/arm and 90 minutes overall.
Measurements sample RSS and physical RAM, not commit charge or GPU memory.

## Assets and provenance

The current local assets are already acquired. On a fresh checkout download only
the two archives from the immutable
[VSM dataset revision](https://huggingface.co/datasets/mbhosale/VSM/tree/4ee0878f6dc96b4a258e344cffb3182024d5a6e1/Cards),
whose dataset card declares CC BY4.0. Place them in
`runs/research_data/vsm/4ee0878f6dc96b4a258e344cffb3182024d5a6e1/`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| card_imgs.tar | 719636480 | `050395809171aba0370539a333d85f5ee79d3c1648ab8b12b3ee75ced3327c91` |
| templates.tar | 133120 | `3fda191bf90f90a20d3a581110c5a5238c606af5fe91c66fe96f7b96dd2fe5e0` |

Preparation validates both hashes, imports at most 4096 grids, rejects unsafe
archive members, ignores scripts and requires all 40 templates. It groups semantic
grid identities before splitting and keeps synthetic variants in their parent's
split. This first archive subset is not representative of the complete dataset.

The color-aware checker uses NCC≥.95, ambiguity margin≥.01 and RGB RMSE≤.10.
The RMSE constant was present in source before the first fit; the original plan's
narrative omitted it. It has not been tuned using generated samples. Source hashes
preserve this implementation detail along with the frozen config.

Data attribution: Bhosale et al., *Score-Control for Hallucination Reduction in
Diffusion Models*,2026. Source modification: bounded import, grouped subset,
independent RGB checker and CPU model/objective implementation. Third-party source
files under `reports/research_sources/` are reference material; no downloaded
installer, evaluator, API agent or training script was executed.

[I-HallA](https://github.com/kaist-cvml/I-HallA-v1.0) is a factual-image/QA benchmark.
Its MIT code license does not establish training rights to every external image.
The two inspected metadata workbooks and their public URLs/checksums are recorded
in `reports/research_sources/ihalla_vsm_2026-10-16/ihalla/metadata_access.json`.
Missing workbooks produce an unavailable status. No I-HallA image is used as a
positive diffusion training example, and no VQA score is fabricated. Dataset rights,
reviewed labels, a disjoint training partition and evaluation permission are
required before further training or external evaluation.

Primary method references: [VSM repository](https://github.com/bhosalems/VSM),
[VSM paper](https://arxiv.org/abs/2606.00377),
[I-HallA paper](https://arxiv.org/abs/2409.12784).

## Inspect through FrameGenX

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app/`, then **Benchmarks** → **ML / DL research
evidence**. The panel's **Refresh research** action reads
`GET /api/research/experiments?limit=5`. It does not submit training jobs.
Rebuild after frontend changes with `npm run build` in `frontend/`.
These are local application instructions, not approval for public exposure.

Reports/checkpoints/samples live under `runs/research_cards/<UUID>/` and local
MLflow uses `runs/mlflow_research.db`. The API validates bounded report identities,
report checksums, denominators and experimental scope, returning only allowlisted metrics
and statuses. Client-supplied file paths cannot select arbitrary reports.
Running, failed, interrupted and corrupt evidence stay visibly distinct.
Statuses describe saved evidence, not a live process monitor. An abrupt OS kill
can leave a running journal; the UI states this liveness limit explicitly. A new
attempt restarts both arms rather than treating that journal as completed evidence.

## Testing and limits

After training exits:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHON_DOTENV_DISABLED='1'
.\venv\Scripts\python.exe -m pytest tests/research -q -p scripts.research_test_guard --basetemp .pytest_tmp_research
```

For a completed run, audit every saved artifact and the actual application mount:

```powershell
.\venv\Scripts\python.exe scripts/verify_research_run.py --run-id 51529996-23e9-436d-91ba-2107057739e8
```

Use the UUID printed by your notebook for a newly created run. This command checks
source/runtime identity, checkpoints, planned sample hashes, local MLflow lifecycle,
the mounted API and the built frontend's entrypoint/data contract. Its saved result
is `reports/research_execution_2026-10-16/integration_verification.json`. This full
artifact audit is separate from the API's inexpensive report checks. Serving the
entrypoint and parsing its data do not verify rendered browser behavior.

The opt-in guard denies external networking and local service ports, allowing
ephemeral loopback for Windows event-loop/test internals. It is a Python test
guard, not OS-level isolation. Actual commands, results and untested boundaries are
recorded in `reports/research_execution_2026-10-16/`.

The current experiment does not measure full-contract CSR, training-seed
uncertainty, general object/color binding, human calibration, browser accessibility,
GPU compatibility or hosted multi-user safety. ChatGPT Dots remains pending exact
service identification. A completed experiment must still report these limits.

Generated datasets, checkpoints, kernels, environments and caches are ignored by
Git. Ignore rules do not remove previously tracked/staged artifacts; review that
existing index separately before making any cleanup commit.
