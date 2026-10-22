# I-HallA / VSM research execution — 2026-10-17

**The local research workflow is implemented and executed. No hallucination
reduction was demonstrated. The trained Cards models remain experimental and
inactive in SDXL generation.**

The [approved task board](planning/EXECUTION_TASKS.md) separates planning,
execution, verification and acceptance for 43 tasks. The original 30 stack work
packages remain staged. This report covers the bounded research work package,
with explicit partial/blocked states for I-HallA, SDXL activation and browser QA.

## Executed experiment

Final run: `51529996-23e9-436d-91ba-2107057739e8`.
Signature: `9e27de2317adeb2d4ac112bcd9fc9d205eebc589ebda148458c19daadd13ac70`.

- Public VSM Cards data: 4096 imported grids; 40 RGB templates. Grouped verified
  split: 2908 training, 600 validation, 588 test grids. Archive/source hashes and
  attribution are recorded; the dataset card declares CC BY 4.0.
- Classical ML: StandardScaler/logistic regression with four frozen synthetic
  corruption families. Fitting uses the training split only.
- PyTorch DL: independent compact learned-variance RGB DDPM, width 16, 100 diffusion
  steps, 500 optimizer steps/arm, batch 8, seed 1729. Compare rho=0 against rho=.1
  under identical initialization, data selection, noise and sampling seeds.
- Evaluation: final checkpoints only; seeds 10000–10031, all 40 RGB templates and
  all four quadrants. No output filtering or post-test threshold tuning.
- All 10 fitting/tracking source files are saved in the final run before fitting.
  Python/platform/numerical/decoder/tracking versions participate in run identity.

This is a small CPU feasibility study, not the authors' full-scale reproduction,
SDXL fine-tuning or a generic image-verification benchmark. The six-channel RGB
checkpoint cannot be loaded as a four-latent-channel SDXL LoRA adapter.

## Actual results

| Measurement | Baseline | VSM candidate |
| --- | ---: | ---: |
| Completed optimizer steps | 500 | 500 |
| Planned/generated samples | 32/32 | 32/32 |
| Valid / invalid / unknown / error | 0 / 32 / 0 / 0 | 0 / 32 / 0 / 0 |
| Cards template validity | 0% | 0% |
| Valid-rate Wilson 95% interval | [0%, 10.72%] | [0%, 10.72%] |
| Distinct sample PNG hashes | 32 | 32 |
| Full-contract CSR | Unmeasured | Unmeasured |

The paired observed validity difference is 0. The empirical bootstrap interval
[0,0] is degenerate at this observed floor; it does not establish population
equality or cover training-seed uncertainty. Fixed-seed sample inspection found
noisy, unusable card outputs. The quality gate remains unmet.

The classical diagnostic scored balanced accuracy 1.0 and ROC AUC 1.0 on 2940
examples derived from 588 held-out grids. These are template-based features and
named synthetic negatives. This score does not establish natural-image accuracy,
independent verification, object/color binding or hallucination reduction.

Experiment wall time was 1058.51s, using four CPU threads. Sampled peak process RSS
was 1.331GB; minimum observed available host RAM was 3.123 GB. These are local CPU
observations, not committed-memory/GPU peaks or a cross-machine latency benchmark.

## Verification performed

| Gate | Actual evidence |
| --- | --- |
| CPU regression suite | **334 passed**,0 failures,20 dependency deprecation warnings; `tests/pressure_test.py` excluded |
| Frontend state tests | **9 passed**; includes invalid, duplicate and wrongly qualified research payloads |
| Frontend build / lint | Vite production build and oxlint passed |
| Installed dependency compatibility | uv checked 283 packages; compatible in the existing environment |
| Python syntax / configuration | AST parsing of18 Python files and TOML parsing passed; Ruff is not installed, so no Python lint claim |
| Fresh notebook execution | All 7 code cells executed, no error outputs, frozen cell-source hash matched |
| Fresh-kernel notebook reexecution | All 7 cells executed; intact completed run reused; no new experiment or retraining |
| Same-seed repeat | All 26 parameter tensors/arm and all 64 sample PNGs reproduced exactly; not independent seed replication |
| Saved artifacts | All 86 final-run artifacts passed checksum audit; both checkpoint heads verified as `[6,32,3,3]` |
| Actual MLflow records | Classical, baseline and candidate runs all FINISHED; qualification remains unverified |
| Actual application integration | Mounted FastAPI read/limit/mutation rejection checks passed; built HTML/JS served; actual API payload accepted by frontend parser |
| Service/device isolation | No external calls or CUDA initialization during the CPU-only API audit; no GPU generation requested |

The final full CPU command was:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHON_DOTENV_DISABLED='1'
.\venv\Scripts\python.exe -m pytest tests -q --ignore=tests/pressure_test.py -p scripts.research_test_guard --basetemp .pytest_tmp_verified_regression
```

The network guard denies external networking and application service ports;
ephemeral loopback remains available for Windows event-loop/test internals.
This is a Python-level test guard, not OS-level network isolation.

## Corrected failures and defects

- Retained an actual initial MemoryError during candidate fitting; broad CPU
  tests overlapped that attempt. Subsequent fitting/testing ran separately.
- Corrected report-publication ordering that could temporarily show corruption.
- Bounded the same report snapshot used for hashing and parsing.
- Rejected existing archive-output directory links, tested with a real Windows
  junction in an isolated directory.
- Cleared stale notebook successes before reruns, saved cells atomically, and
  corrected lifecycle metadata updates. The initial three recovery-test failures
  and their successful fix/rerun remain recorded.
- Added complete fitting-source preservation and relevant runtime identities.
- Marked the legacy current-state audit historical and narrowed testing claims.

See [findings](research_execution_2026-10-16/FINDINGS.md) and the retained failure
logs for reproduction details and the distinction between source findings and
runtime observations. The final 334-test run followed these fixes.

## Exploratory diagnosis of noisy outputs

A separate, explicitly post hoc diagnostic used 8 validation grids and six fixed
timesteps. At timestep 99, alpha-bar was 2.43e-7. Baseline/candidate epsilon MSE was
.02328/.02245; unbounded x0 reconstruction RMSE was 309.61/304.04 in normalized
[-1,1] image units. Approximately 99.7% of predicted channels exceeded that range
before clipping.

This shows inadequate high-noise denoising precision and large reconstruction
amplification. It is consistent with the failed samples, without proving a
particular remedy or a causal VSM benefit. No fitting, sampler, threshold,
benchmark or qualification decision changed. Review model capacity, training
duration and stable prediction parameterization on validation data before a new
locked study; retain this negative result.

## Remaining gates

- **I-HallA:** public caption/history-QA metadata were inspected and held out.
  Image assets, permitted VQA, training-data rights and reviewed disjoint training
  data remain unconfigured. No I-HallA fine-tuning or evaluation score is claimed.
- **SDXL:** compatible variance head/objective/sampler/export, common device
  ownership and an independent full-contract study remain required. These Cards
  weights are inactive. The existing PEFT/QLoRA path is a separate workflow.
- **Browser/accessibility:** earlier automatic approval review rejected browser
  automation. Rendered journeys and WCAG checks remain unverified.
- **Broader QA:** no new GPU workload, clean install, container/hosted deployment,
  remote CI, cross-platform matrix or human calibration is certified here.
- **ChatGPT Dots:** exact service identification remains pending; no invocation
  of an unidentified service is claimed.

The [pinned VSM source](https://github.com/bhosalems/VSM/blob/0972d3e97392a4ddca5c032abd935a3592b0ca41/README.md)
describes pretrained variance-head-only fine-tuning. Review that approach against
SDXL's interfaces and the 6 GB budget before implementing a compatible adapter.
Adding datasets or decreasing denoising loss alone does not satisfy the gates.

## Run and inspect

- [Executed notebook](../output/jupyter-notebook/ihalla_vsm_training.ipynb)
- [Run/setup guide](../docs/17_RESEARCH_EXPERIMENTS.md)
- [Machine-readable execution summary](research_execution_2026-10-16/execution_summary.json)
- [Final raw report](../runs/research_cards/51529996-23e9-436d-91ba-2107057739e8/report.json)
- [Artifact/API audit](research_execution_2026-10-16/integration_verification.json)
- [Full regression log](research_execution_2026-10-16/final_advanced_regression_tests.log)

The existing app exposes `GET /api/research/experiments`; **Benchmarks → ML / DL
research evidence** displays the saved records. This connects research evidence
to the full stack and does not activate the experimental generator.

Generated research data, checkpoints, kernels, environments and caches are now
ignored by Git. The final inventory records 2065 tracked changes and 1009 visible
untracked paths, including substantial pre-existing work. No index cleanup,
unstaging, reset, commit or push was performed. IDE warning resolution was not
visually verified; see [inventory](research_execution_2026-10-16/git_inventory.json).
