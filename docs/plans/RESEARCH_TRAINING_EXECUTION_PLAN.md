# I-HallA / VSM research and execution plan

Approved direction: user selected A on 2026-10-16, accepting the reviewed staged roadmap and its recommendations. This addendum implements the requested task breakdown before coding. It adds explicit research, Jupyter, training, full-stack integration and advanced testing work; it does not certify accuracy in advance.

## Outcome and boundaries

Produce reproducible ML and DL experiments with real licensed data, record the results, expose trustworthy research results through FrameGenX, and run meaningful regression/integration/security/resource tests. The metric for general image generation remains full-contract CSR with independent labels; benchmark factual QA and domain validity are separate measures. No trained candidate is enabled in normal generation without compatible architecture and qualification evidence.

The notebook is an orchestration and inspection surface. Reusable Python modules own data validation, objectives, training, checkpointing, evaluation and report validation. This follows AGENTS.md and the FrameGenX guidance skill: modules, tests, local execution and saved intermediates. Jupyter experiment templates are used for clean top-to-bottom execution.

## Source audit, verified 2026-10-16

| Source | Pinned source | Actual role and constraint |
|---|---|---|
| [I-HallA](https://github.com/kaist-cvml/I-HallA-v1.0) | `e750fd025b1473819234d07ee2d467bb05b3cd9c` | Factual-image QA benchmark/evaluation agents, not a diffusion training implementation. Code license MIT; that does not automatically establish rights to all externally hosted textbook/generated benchmark images. GPT-4o evaluation in the reference requires API credentials/calls. |
| [I-HallA paper](https://arxiv.org/abs/2409.12784) | v1 reviewed | Scientific/historical factuality differs from generic exact-count/color-binding accuracy. Preserve benchmark families as held-out evaluation; never use its known hallucinated images as positive diffusion examples. |
| [VSM](https://github.com/bhosalems/VSM) | `0972d3e97392a4ddca5c032abd935a3592b0ca41` | Learned-variance diffusion training plus variance-derived score-curvature penalty. Source tree has MIT upstream subdirectory licenses but no root license attribution for all new changes; inspect as reference, implement paper mathematics independently rather than transplant unlicensed code. |
| [VSM paper](https://arxiv.org/abs/2606.00377) | v1 reviewed | Compare learned-variance baseline `rho=0` with VSM `rho>0`, with identical architecture/budget/data/noise. The practical diagonal penalty is not a full Jacobian and results do not imply broad SDXL improvement. |
| [VSM data](https://huggingface.co/datasets/mbhosale/VSM) | Resolve and record immutable dataset revision before download | Dataset card declares CC BY 4.0 research data: Cards 91,390, ChessImages 196,062. README approximations differ; count actual imported records. Cards archive ~720 MB, templates ~133 KB; full chess training archive ~4.3 GB. Start with bounded Cards data; do not download all archives. |

Source files were downloaded read-only under `reports/research_sources/ihalla_vsm_2026-10-16/`, with URLs, revisions and SHA256 hashes. No source installer, agent, archive extractor or training command from those repositories has been executed. Upstream `tar.extractall` and legacy Conda environments are not adopted blindly.

### Compatibility finding

Current `src/training/objective.py` uses epsilon/v-prediction denoising loss for SDXL. VSM requires a learned variance output/head and a sampler/training objective that consumes it. A four-channel SDXL noise-prediction LoRA checkpoint cannot silently become a learned-variance checkpoint; extra variance channels cannot be treated as an interchangeable style adapter. First establish a small learned-variance Cards diffusion experiment. Then require an explicit SDXL variance-head/sampler design, resource probe, checkpoint/export compatibility and disjoint contract study before general-generator activation. If this gate fails, keep the research experiment/report connected and the production generator unchanged.

I-HallA training-data use is gated on dataset-level rights, reviewed factual labels and a training partition genuinely disjoint from its evaluation benchmark. Until then, its preparation/evaluation path can be integrated and marked unavailable when assets/provider permission are absent; no fabricated score, paid call or claimed I-HallA fine-tuning.

## Separate planning and execution tasks

Each task has its own planning output, execution work, dependency and exit evidence. `planned` is not `implemented`; `trained` is not `qualified`. Original C1-C13/D1-D4/X1-X5/E1-E8 remain in the [30-task ledger](../../reports/planning/IMPLEMENTATION_TASKS.md); this addendum does not silently mark those six-month roadmap items complete.

| ID | Planning task / deliverable | Execution task | Depends on | Exit evidence |
|---|---|---|---|---|
| R01 | Record A approval, approved plan hash, accepted challenges/taste choices and new request | Update approval record and phase review logs with honest outstanding runtime gates | User A | Approval receipt; plan scope clear, no runtime clearance invented |
| R02 | Audit code/data/model licenses, immutable revisions, sizes and label meaning | Acquire permitted bounded VSM assets safely; build source/data inventory; inspect public I-HallA metadata without training on its evaluation set | R01 | Hash manifest, attribution, safe extraction limits, no evaluation leakage; explicit missing rights/assets |
| R03 | Freeze benchmark roles, R07 evaluation definitions and statistical/compute budget before any fit | Make immutable train/validation/test family splits, label/source hashes, CPU resource preflight and locked baseline/evaluation protocol | R02 | Disjoint splits and immutable config/seeds/thresholds/selection/denominators; insufficient power labeled feasibility |
| R04 | Define classical ML validity baseline and negative-data provenance | Train a small scikit-learn model on independently identified card/template features and recorded synthetic corruptions | R03 | Saved model/config, held-out metrics, corruption definitions, calibrated/uncalibrated scope and no general-image claim |
| R05 | Define learned-variance DDPM and rho ablation | Train PyTorch CPU baseline and VSM candidate with equal data, initial state, batches, noise, steps and sampler | R03; GPU execution additionally blocked on base E1/common-device exclusion tests | Checkpoints, objective terms, exact seeds, timing, CPU memory, failures and MLflow IDs; no CUDA initialization or automatic activation |
| R06 | Specify notebook cells, kernel and rerun/resume behavior | Create and execute a Jupyter experiment notebook calling R02-R05/R07 modules | R02-R05 interfaces | Top-to-bottom executed notebook, compact actual outputs, no credentials/hidden state; interrupted cells fail truthfully |
| R07 | Define independent sampling/validity/quality report protocol in R03 before training | Execute the frozen equal-seed baseline/candidate sampling, color-aware checks and uncertainty/diversity/memorization report | R03 protocol, then R04-R05 artifacts | Every planned sample recorded; failed/invalid/unknown counts included, no filtering noise away; feasibility is not reduction proof |
| R08 | Map I-HallA sheets/images/QA and provider permissions to an evaluation adapter | Import licensed permitted local metadata/assets through validated schemas; export offline evaluation requests; evaluate only when configured | R02 | Source/QA/image identities, missing assets/permissions explicit; dataset license/split gate before training |
| R09 | Design versioned experimental research report and safe full-stack reader | Add bounded read-only API and Benchmarks UI showing actual dataset/method/training/evaluation status and evidence scope | R07/R08 reports | API/UI state agree; missing/corrupt/partial data visible, no arbitrary file access or checkpoint auto-loading |
| R10 | Review SDXL learned-variance/LoRA compatibility and downstream qualification criteria | Implement a generator-compatible adapter only if output-head, sampler, resource, export and evaluation gates pass | R05/R07 plus C1/C2/E4/E5 | Compatible checkpoint proven; locked generic contract study supports promotion. Otherwise explicit incompatibility, not renamed toy weights |
| R11 | Map advanced software tests to data/model/notebook/API/UI/error/security/recovery boundaries | Run targeted unit/property/fault tests, API integration, notebook reexecution, frontend build/lint/state tests and approved browser journeys | R02-R09 | Actual commands/logs and failures; GPU/human/hosted/browser gaps distinct; preserve existing tests |
| R12 | Define reproducible handoff, limitations and next-study decision | Save result summary, source/config/data/checkpoint hashes, user run commands and honest completed/blocked task states | R11 | Reviewable artifacts; no hallucination-elimination or unsupported improvement claim |
| R13 | Identify what the user means by ChatGPT Dots and available connector | Use the identified service only after capability/permission is known; keep local review/task artifacts usable meanwhile | User clarification | Exact service named and actual invocation recorded; never claim an unavailable tool ran |

## Original roadmap: planning and execution separation

For **every** original task C1-C13, D1-D4, X1-X5 and E1-E8, the execution ledger carries separate design/specification, implementation, verification and acceptance states. All completed review work is specification evidence. Actual coding/test acceptance starts per work package after dependencies, with no mass-install step. Proposed task file lists are design surfaces, not files claimed already created.

Execution sequence for the current requested slice: R01 -> R02 -> R03 (including frozen R07 definitions) -> R04/R05 CPU experiments -> R06/R07 execution -> R08/R09 -> R10 compatibility decision -> R11 -> R12. R13 clarification is independent. Any future CUDA training, probe or sampling requires base E1/common-device-gate implementation and actual cross-entrypoint exclusion tests first; a notebook-only lock or free-memory snapshot cannot satisfy it. The initial CPU-only research config rejects CUDA devices and does not initialize CUDA. The broad P0-P5 roadmap remains staged; new optional providers/hosting are not installed merely to display a research result.

## Frozen first feasibility experiment

- Domain: released VSM Cards only, using RGB templates and original image identities. Require a complete manifest of all 40 A-through-10/suit templates, 64x64 RGB dimensions and hashes. Match each of four quadrants against all 40 templates, with color and ambiguous-match handling; missing/corrupt templates fail preparation. Record immutable dataset revision and archive hashes. Keep source images unmodified; resize only in a declared model input transform.
- Data budget: at most 4,096 imported card grids for the first trial, source/family-group split before fitting. Exclude exact/semantic-grid duplicates across splits. Keep a separate pristine validation set plus held-out corruptions for diagnostics. Source “valid” labels are checked in all four quadrants against the complete 40-template inventory in color; ambiguity stays unknown.
- ML: deterministic logistic regression over template/residual features; C=1, max_iter=1000, fixed seed 1729. Freeze synthetic corruption families (pixel noise, missing quadrant, symbol/pixel shuffle and color-channel change) in R03, generate variants from split-specific source families and record their label scope. No threshold fitting on test data. Report per-type held-out errors; this is an experimental domain-validity diagnostic, not a qualified FrameGenX verifier.
- DL: independent Cards research config (not TrainingConfig/LoRA): 128x128 RGB inputs, compact U-Net width16, six output channels split epsilon3/variance3, 100-step cosine beta schedule (offset .008, beta capped .999), uniform timestep sampling, fixed seed1729, batch8, AdamW lr=.001 and <=500 equal steps/arm. LearnedRange maps `(tanh(v)+1)/2` between clipped posterior log variance and log(beta); the bounded mapping is a declared local stability adaptation. Objective is mean epsilon MSE plus mean KL/decoder NLL in bits/dimension (weight1), with epsilon detached in the variational term; VSM adds `rho/sqrt(1-alpha_bar_t) * mean(sigmoid(2*log_variance))`, equivalent to the paper's diagonal `1/(1+precision^2)` penalty. Baseline rho=0 must equal MSE+VLB exactly. Sampler consumes learned variance in ancestral reverse steps, with no noise at t=0 and x0 clipped to [-1,1]. The schedule follows the paper; the pinned source's constant-rho variant is recorded as a discrepancy rather than silently equated. Freeze actual step count and code/config hashes before fit; <=30 minutes/arm, combined <=90-minute notebook deadline. Checkpoints have a separate Cards schema and are never SDXL adapters.
- Resource gate: initial experiment device is CPU, at most4 torch threads, one run lock, bounded data/model allocations and observed process-tree memory. No CUDA import/probe/use path is requested by the notebook. CUDA variants remain disabled until E1 covers every API/CLI/benchmark/train/verifier entrypoint, with parent/child lifecycle and interruption cleanup tested. GPU free-memory inspection alone is insufficient. Respect 6 GB VRAM/16 GB RAM; never force-kill unrelated processes or run concurrent heavy generation.
- Sampling/checkpoint selection: use final fixed-step checkpoints only (no best-test selection), seeds10000 through10031 paired across arms, 100 ancestral steps, same declared transform and no sample filtering. Missing/corrupt/blank outputs remain in denominators. One paired training run and32 paired samples are feasibility evidence only.
- Evaluation frozen in R03: require128x128 RGB sample, each64x64 quadrant compared to all40 templates; color-aware normalized correlation threshold .95, ambiguity margin .01, no post-test tuning. Report valid/invalid/unknown/error and all denominators; blank/unreadable is a failed sample, not skipped noise. Paired sample-bootstrap 95% interval with2000 resamples, seed2718; Wilson interval for each rate; single training-seed uncertainty explicitly remains unmeasured. Include distinct sample hashes and nearest-training pixel distance to diagnose memorization/collapse without calling either full fidelity metrics. Generic FrameGenX CSR stays null/unverified. Keep paper results separate from local measurements.
- Tracking: existing local MLflow only, no remote payload export or provider keys required. No hosted GPU spending or paid VQA call is authorized implicitly by a notebook.

## Integration and test architecture

```text
pinned permitted assets -> validated source manifest -> immutable grouped splits
  |-> classical ML baseline -> model+scope+held-out diagnostics
  `-> equal-budget learned-variance baseline / VSM -> checkpoints -> paired samples
           | Jupyter calls tested Python modules; local MLflow records actual runs
           v
     color-aware domain evaluation -> versioned experimental report
I-HallA metadata/assets -> held-out QA adapter -> explicit unavailable / permitted evaluation
           v
validated allowlisted report reader -> FastAPI read-only research endpoint
           v
React Benchmarks research panel -> task/status/scope/errors + evidence references

compatibility gate -> SDXL head/sampler/export probe -> independent generic CSR study
                    (incompatible or unqualified stays inactive)
```

Advanced testing covers: hostile archives/paths/size budgets, split leakage/hash tampering, invalid tensors/nonfinite objective/gradient behavior, deterministic initialization and resume identity, missing/duplicate/blank samples, wrong-model reports, offline/no-key behavior, API bounded serialization/not-found/corrupt states, request overlap and resource contention, frontend error/empty/partial rendering and notebook top-to-bottom reexecution. Existing regression tests and clean frontend build/lint run after integration. Actual browser automation still requires the approved environment; the earlier rejection is not bypassed. Human calibration and hosted tenant/recovery testing remain distinct gates for the broader roadmap.

## Current status before coding

Planning/source inspection and approval/task breakdown were recorded before coding. The initial engineering review identified40-template inventory, exact objective definition, evaluation-before-training and common-GPU prerequisites. This revision corrects them and deliberately selects CPU for the bounded first experiment, leaving CUDA disabled until E1. A subsequent independent amendment check returned PASS for specification only on all four corrections. Execution may now begin under these gates; actual outcomes are tracked separately. ChatGPT Dots clarification is pending; plugin search did not identify an exact matching tool. Status updates and actual evidence will be recorded in `reports/research_execution_2026-10-16/` and the task ledger.

