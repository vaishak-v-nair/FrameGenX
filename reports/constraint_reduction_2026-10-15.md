# Constraint control and verification — 2026-10-15

**Measured development improvement:** fixed four-case, two-seed local
count/position checks improved from 1/8 to 4/8; chromatic binding diagnostics
improved from 1/6 to 3/6. These are uncalibrated operational checks, not
full-contract CSR. Duplicate vehicles, malformed regions and verifier errors
remain. The trained adapter is disabled and is not qualified by this work.

## What changed

The earlier QLoRA pilot optimized denoising loss on 90 training images; that
loss did not establish exact counts or per-object color binding. Its weights
remain experimental. This follow-up implements explicit inference controls
and independent local measurements; trained adapters are disabled throughout
the controller comparison.

- The compiler creates an individual noun/attribute condition and region for
  each counted foreground instance. Repeated nouns no longer share one union
  of token regions. Unsupported foreground relations, explicit placement,
  unknown foreground counts and duplicate instance IDs fail eligibility.
- The NF4 SDXL backend combines region-specific noise predictions and advances
  the Euler scheduler once per timestep. Early steps use masked neutral
  backgrounds and reversible region centering; later steps use the original
  global prompt. The 512px/25-step/CFG-7 profile processes at most two regions
  per batch and releases models before verification.
- GroundingDINO candidate boxes are segmented with pinned SAM2-tiny weights.
  Eroded-mask HSV measurements expose color contradictions. Weak or mixed
  colors remain unknown. The explicit profile blocks unknown required
  chromatic colors and unavailable instance verification.
- Duplicate part/whole detector boxes are suppressed only when their
  high-quality instance masks agree. Expected counts never select survivors.
  Raw counts, retained/suppressed masks, measurements and hashes are saved.
- The application accepts a validated supplied contract, preserves the
  original requirements, bypasses intent-provider dependence and isolates
  its cache identity. Repairs use different recorded seeds. CLI messages
  distinguish operational checks from full-contract evidence.
- The final verification artifact now agrees with the acceptance decision;
  it previously retained the verifier's initial `accepted: false` snapshot
  even when the pipeline later accepted the attempt. Detailed QA observations
  survive the final write.

## Research and implementation scope

The regional sampling approach adapts
[MultiDiffusion](https://multidiffusion.github.io/), which combines multiple
denoising paths under spatial constraints. Mask centering draws on
[SemanticDraw](https://github.com/ironjr/semantic-draw) and its
[CVPR 2025 supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Lee_SemanticDraw_Towards_Real-Time_CVPR_2025_supplemental.pdf).
These methods are adapted to the existing NF4 SDXL/Euler stack; published
benchmark results are not reproduced here. The official SemanticDraw setup
lists 8GB minimum VRAM. Only this project's separately measured implementation
is assessed against the 6GB laptop constraint.

Local segmentation uses the official Apache-2.0
[SAM2.1-tiny checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny),
revision `de431c4043854a71d8101e17995dfe596bf101a5`, 155,908,064 bytes,
SHA256 `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
It is cached once, then loaded locally with a hash check. No external vision
provider is needed by this profile.

## Development attempts retained

| Run directory under `runs/constraint_control` | Method | Observed limitation |
|---|---|---|
| `e172e12bdea1c6fc` | Separate regional attention only | Apple count collapsed to one; vehicle binding still failed |
| `732bfe8d5df51948` | Uncentered region-path fusion | Three apples and separate red-car/blue-bus forms appeared; row/shape quality was poor |
| `60d48074671f5a06` | Tighter boxes, separate object prompts | Vehicle/animal separation improved; clipped apple and pale bottle remained |
| `d71bb1cfb189b353` | Regional phase followed by global refinement | Clipped apple and pale bottle remained |
| `7bf7ecd0bd068780` | Added reversible mask centering | Bottle count changed from one to two, both locally measured green; apple image still failed (four detected regions) |

Every run retains original output pixels, code snapshot, configuration,
seeds, model/package identities, intermediate latents and SHA256 journals.
No failed attempt is removed or used as a positive training target.

The centered two-case development run reserved **2.582GB** in PyTorch and
took **297.90s**, including one exact native-Diffusers parity check and model
startup. The independent local diagnostics supported 0/2 baseline versus
1/2 candidate count checks and chromatic bindings; this is a development
result with two cases and one seed, not qualification evidence.

An actual detector defect was also reproduced on the saved seed-43 baseline:
GroundingDINO returned seven part/whole boxes for four visible apples. SAM2
mask agreement reduced the diagnostic count to four while preserving the
seven raw detections. See
[regression evidence](instance_count_mask_regression_2026-10-15.json).
This is an inspected regression, not an independently calibrated accuracy rate.

## Fixed comparison

Run: [`1650c5c998968391`](../runs/constraint_control/1650c5c998968391/result.json).
Four known development contracts, seeds 42 and 43, eight baseline/candidate
pairs. The model weights, package versions, seeds, 512px resolution, 25 Euler
steps, CFG 7 and zero negative conditioning match. Baseline PNGs and latents
are copied unchanged from the previous native Diffusers evaluation and
validated by SHA256. Candidate generation disables the trained adapter.
The uncontrolled sampler matched an actual saved baseline latent exactly
before candidate sampling.

The controller was fixed before this complete comparison: 12 regional steps,
12 bootstrap/centering steps, then 13 original-prompt global steps. Earlier
development attempts above informed it; this is not a held-out benchmark.

| Local operational measurement | Baseline | Regional controller |
|---|---:|---:|
| Images with available diagnostics | 8/8 | 8/8 |
| Count-check passes | 1/8 | 4/8 |
| Count and requested left/right passes | 1/8 | 4/8 |
| Supported chromatic binding, where requested | 1/6 | 3/6 |
| Joint count, geometry and applicable chromatic checks | 1/8 | 4/8 |
| Full-contract CSR | Unmeasured | Unmeasured |

Under these local checks, three pairs improved, none regressed, and five
were unchanged. This describes the detector/segmentation/hue outcomes only.
It does not establish statistical significance or overall image quality.
Color denominators exclude the two animal cases, which request no colors;
missing, ambiguous or mismatched color evidence never counts as supported.

| Case / seed | Baseline detected counts | Candidate detected counts | Candidate operational result |
|---|---|---|---|
| Apples / 42 | 6 | 3 | Pass; baseline detector counts reflections |
| Apples / 43 | 4 | 4 | Fail; extra partial apple form persists |
| Vehicle binding / 42 | car 2, bus 2 | car 2, bus 1 | Fail; duplicate car |
| Vehicle binding / 43 | car 0, bus 1 | car 5, bus 1 | Fail; repeated cars |
| Cat/dog / 42 | cat 1, dog 0 | cat 1, dog 1 | Pass; correct left/right |
| Cat/dog / 43 | cat 3, dog 1 | cat 2, dog 0 | Fail; stylized dog misclassified |
| Bottles / 42 | 2 | 2 | Pass; both locally measured green |
| Bottles / 43 | 1 | 2 | Pass; both locally measured green |

### Visual inspection and remaining verifier errors

All sixteen native PNGs were inspected. In the actual pixels, the baseline
apple-42 image has three physical apples and three reflections. Their masks
do not overlap, so mask deduplication correctly leaves the six detector
regions separate; it cannot establish which are reflections. That apparent
check improvement is not evidence that a true physical-object count error
was repaired. The candidate animal-43 image visibly resembles one cat and
one pug-like dog, despite the detector assigning both to the cat class.

The clearest visual corrections are the previously missing dog in animal-42
and the missing green bottle in bottle-43. Both animal candidates show a
cat on the left and a dog-like figure on the right. Vehicle candidates show
better color separation but still fail exact counts. Strong framing, striped
surfaces, clipped objects and inconsistent rendering styles remain visible.
Full visual coherence and background requirements are not certified.

Inspect the [paired gallery](constraint_comparison_2026-10-15.html),
[hash-checked diagnostic summary](constraint_comparison_2026-10-15.json),
[raw verifier report](../runs/constraint_control/1650c5c998968391/local_verification_960b80c5941f.json)
and [qualitative image review](constraint_visual_review_2026-10-15.json).
The qualitative review is unblinded assistant inspection, not independent
human reference labels.

### Runtime and tracking

The complete comparison took **708.26s** for generation, including startup,
one baseline parity generation, eight candidate generations and decoding;
it reserved **2.582GB** and allocated **2.485GB** in PyTorch at peak. The
sequential local verification took **117.34s**, reserved **1.952GB**, and
completed all sixteen images without unavailable diagnostics. Windows process
working-set sampling observed a **5.100GB process peak** during generation;
this is not a full-system RAM measurement. See
[memory snapshot](constraint_generation_memory_2026-10-15.json).

Local MLflow experiment: `framegenx-constraint-control`, run
`7ce64b6419cd4836be5e2094a1706464`. It records bounded diagnostic counts,
case/seed denominators, runtime, memory, input hashes and source journals.
Its scope is local operational diagnostics; `quality_status` remains
`unverified`, promotion `unqualified`, and full-contract CSR `unmeasured`.
The earlier trained-adapter run is unchanged.
The [MLflow read-back check](control_mlflow_check_2026-10-15.json) confirms
the finished lifecycle, scope tags and every saved summary metric.

Reproduce the comparison and derived report:

```powershell
& .\venv\Scripts\python.exe scripts/evaluate_regions.py --baseline runs/training/76fada99-67db-4fbe-a508-c0ef96e6042f/evaluation_557b015ef7a8_74fc501e --method fusion --bootstrap-steps 12 --regional-steps 12 --resume
& .\venv\Scripts\python.exe scripts/verify_local.py --evaluation runs/constraint_control/1650c5c998968391 --resume
& .\venv\Scripts\python.exe scripts/summarize_control_comparison.py --evaluation runs/constraint_control/1650c5c998968391 --verification runs/constraint_control/1650c5c998968391/local_verification_960b80c5941f.json --output-prefix reports/constraint_comparison_2026-10-15
```

Use the offline environment settings in the
[profile guide](../docs/16_CONSTRAINT_CONTROL.md) first. `--track` on the
summary command creates a new local MLflow assessment run. The generation
runner's completed `--resume` currently rechecks sampler parity and rewrites
its timing journal, so verification must be rerun afterward. Existing report
hashes describe the original completed journal and are preserved in MLflow.

## Actual application execution

Executed the documented CLI against the real pipeline, with supplied
`configs/examples/green_bottles_contract.json`, seed 43 and
`configs/constraint_control.yaml`. Run:
[`21cb7dfb-d9fb-4981-b956-aee644768372`](../runs/21cb7dfb-d9fb-4981-b956-aee644768372/generation_meta_0.json).

- CLI exited **0** and accepted the operational checks on the first attempt.
- The actual detector/refiner found **two bottles**; both object masks were
  measured **green**. The external VQA provider was disabled.
- The generated PNG and latent tensor match the research candidate exactly.
  This checks the production dispatch, prompt encoding, sampler and decode,
  rather than just a test fixture or isolated research implementation.
- `verification_0.json`, `decision_0.json` and the final pipeline acceptance
  agree. The supplied negation remains in the saved contract and evidence
  ledger.
- Generation took **136.09s**, total pipeline **143.5s**, with **2.563GB** peak
  GPU reservation. The ledger remains `unverified` for requirements beyond
  the operational checks. The CLI says: operational checks passed; remaining
  contract requirements need review.

See the [application smoke evidence](control_cli_smoke_2026-10-15.json).
The real verifier and reflection stages also replayed the unchanged failed
vehicle-43 PNG. They detected **five cars instead of one** and one bus,
reported a count mismatch plus unresolved car binding, and returned
`accept: false`. This was a verification replay of an existing image,
not another generation. See the
[negative-case evidence](control_rejection_2026-10-15.json) and
[saved rejected decision](../runs/83113d43-559a-4fd5-9dd1-bbe3b6e1cb83/decision_0.json).

This is actual offline CLI execution; browser UI and Docker behavior are
outside this follow-up's executed scope. No additional fine-tuning was run
in this follow-up, and no controller result is attributed to the trained LoRA.

## Regression verification

- Unit and workflow suite: **288 passed**, 20 dependency deprecation warnings,
  127.58s. Command: `venv\Scripts\python.exe -m pytest tests -q
  --ignore=tests/pressure_test.py --basetemp=.pytest_tmp_constraint_release_20261015`.
- `pip check`: no broken requirements; source compilation and selected diff
  whitespace checks passed.
- The uncontrolled custom Euler/CFG sampler matched saved native Diffusers
  latents exactly on the actual GPU. A tiny real Diffusers model also checks
  sampler parity and repeatability in the automated suite.
- Tests cover immutable contracts, route isolation, scheduler advancement,
  count deduplication, color abstention, unavailable verifier rejection,
  deterministic repair and supplied-contract cache isolation.

An initial repository-root pytest invocation collected two unrelated manual
scripts named `test_pipeline.py` and failed collection. The documented `tests/`
suite command above passed; that failed invocation is retained in the logs.

## Limits and qualification

Local detector, mask-quality and hue thresholds remain uncalibrated.
Achromatic surfaces, negation coverage, arbitrary instance identity,
extraction recall and general visual coherence are not established by these
diagnostics. Full-contract CSR stays null. No trained adapter is promoted.
The fixed comparison uses four known development cases and two seeds;
it cannot establish generalization or statistical significance. The
[profile guide](../docs/16_CONSTRAINT_CONTROL.md) documents supported inputs
and exact local execution commands.

[Final integrity checks](constraint_final_checks_2026-10-15.json) validate
all sixteen image/latent pairs, sixteen qualitative review identities,
experiment source snapshots, gallery images and local documentation links.
The [evidence manifest](constraint_evidence_manifest_2026-10-15.json)
records final file hashes.
