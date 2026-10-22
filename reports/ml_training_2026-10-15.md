# Constraint-adapter training assessment — 2026-10-15

**Follow-up:** the [constraint-control assessment](constraint_reduction_2026-10-15.md)
records subsequent inference-controller and verifier changes. Those runs
disable the trained adapter, so they do not revise the adapter conclusions below.

## Decision

**Actual local training completed. Hallucination reduction remains unverified.**
The SDXL-base candidate is experimental and unqualified for default use.
Evidence establishes bounded optimization, Diffusers export/loading, and
deterministic optimizer continuation. It does not establish production
readiness or full contract satisfaction. This continues the
[QA assessment](qa_2026-10-13.md) and
[image-generation research review](research_image_generation_2026-10-13.md).

## Executed training

Run: `76fada99-67db-4fbe-a508-c0ef96e6042f`.
MLflow: `4a7a11a333f04d868d153e28b5673b02`, local `runs/mlflow.db`.
[Raw result](../runs/training/76fada99-67db-4fbe-a508-c0ef96e6042f/result.json).

| Measurement | Observed result |
|---|---|
| Backbone | SDXL base 1.0, frozen NF4 double-quantized linear weights |
| Adapters | PEFT rank/alpha 4; 5,806,080 trainable parameters |
| Compute / resolution | BF16; full-image letterboxing to 512 × 512 |
| Optimization | AdamW, lr 1e-5, Min-SNR gamma 5, batch 1 × accumulation 2 |
| Work performed | 48 optimizer steps; 96 presentations; all 90 training images seen |
| Source presentations | CLEVR 32, COCO 33, VSR 31 |
| Changed adapter tensors | 1,120 |
| Training / total seconds | 507.79 / 647.76 |
| Peak allocated / reserved GPU memory | 2.289 / 2.363 GB (decimal, PyTorch process only) |
| Checkpoints | Steps 16, 32, 48; adapter-only SafeTensors plus training state |
| Quality / promotion | `unverified` / `unqualified` |

Environment: Windows, Python 3.14.6, torch 2.13.0+cu130, Diffusers 0.39.0,
Transformers 5.13.0, PEFT 0.19.1, bitsandbytes 0.50.2, MLflow-skinny 3.17.1.
Optional dependencies were installed against the pinned application runtime.

Backbone snapshot: `462165984030d82259a11f4367a4eed129e94a7b`.
UNet SHA256: `83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139`.
Final repaired adapter SHA256:
`aaaeb9b68609b11ffc032c1a57c765b983b339af22c7e396c24bfd72b289953b`.
Executed pilot source fingerprint:
`11f3be322f5743e258ce6941bfa444e398132b25691b42f1de06cb76f71e40d9`.
Its source snapshot remains in the run; current source includes subsequent fixes.

## Dataset audit

[Manifest](../data/training/constraint_pilot/records.jsonl), SHA256:
`5377fd0a2c0856160a00d51f46d1345ea54d0c5101e8289b5136a5a4bd5d1619`.
121 records: 108 distinct positive image groups plus 13 false VSR statements.

| Source | Positive train / validation / test | Provenance and supervision |
|---|---|---|
| CLEVR | 30 / 3 / 3 | Symbolic counts/colors/shapes/relations; [500-image mirror](https://huggingface.co/datasets/maujim/CLEVR_sample_500) of [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), revision `463520cb0e4d16f2f61ae0607c81eacdd009e9f4`; mirror's CC BY 4.0 metadata retained |
| VSR | 30 / 3 / 3 | [Official validated spatial statements](https://github.com/cambridgeltl/visual-spatial-reasoning), revision `b27a0af0ee1462d2b6b92c8c83e869d9254a241a`; positive validator consensus; annotation/image licenses recorded separately |
| COCO | 30 / 3 / 3 | [Official instance annotations](https://cocodataset.org/#download); counts 1–4, noncrowd, visible mask area filter; original photo attribution retained |

Natural photos passed the CC BY 2.0 image-license filter. Acquisition retained
TLS verification, using a valid S3 path endpoint when the COCO vanity host's
certificate failed. No failed generated image was treated as a positive target.
False VSR captions are excluded from generative learning.

Image hashes do not cross splits. Nine validation groups were reserved; eight
were used for fixed-noise diagnostics (CLEVR 3, COCO 3, VSR 2). Nine test groups
were excluded from learning and checkpoint selection. Hash isolation cannot
prove absence from SDXL pretraining or detect every near-duplicate. COCO
annotation exhaustiveness and VSR label errors remain possible. The CLEVR
mirror is small and is not the full official release.

## Loss evidence and its limit

| Paired validation diagnostic (8 images) | Value |
|---|---|
| Baseline / candidate mean noise MSE | 0.0748820691 / 0.0739780304 |
| Relative decrease | 1.21% |
| Mean / median paired delta | -0.000904039 / -0.000874851 |
| Bootstrap mean-delta 95% CI | [-0.001311354, -0.000531500] |

This is a small objective-space diagnostic with fixed noise and timesteps.
Validation images were monitored during training. It is not an independent
final generalization estimate. A noise-MSE confidence interval is not a CSR
confidence interval. No hallucination-reduction percentage is claimed.

## Reliability defects repaired

1. **Incomplete export metadata:** real Diffusers loading raised
   `KeyError: rank_pattern`. Export now preserves the complete PEFT config,
   including rank/alpha semantics. Metadata migration repaired all three
   checkpoints, retaining originals and proving all 1,120 tensors unchanged.
   [Repair audit](ml_export_repair_2026-10-14.json),
   [original failure](ml_evaluation_export_failure_2026-10-14.log).
2. **Offline file loading:** Diffusers could not infer the local filename.
   The application hook now supplies directory, explicit `weight_name` and
   `local_files_only=True`. The real comparison loaded and activated the export.
3. **Warning-only determinism:** the original pilot warned about nondeterministic
   memory-efficient attention. Current code uses strict checks and disables that
   backend. A [real 512px backward probe](ml_strict_preflight_2026-10-14.json)
   passed with 2.483 GB reserved. The older pilot is not relabeled deterministic.
4. **Optimizer continuation:** the [real CUDA comparison](ml_resume_smoke_2026-10-14.json)
   matched all 1,120 tensors exactly (maximum difference 0.0), with 2.481 GB
   GPU reservation and 4.762 GB peak working set. It compares 2 steps + resume
   to step 3 against an uninterrupted 3-step run on the same stack. Subsequent
   source-snapshot persistence is covered by a focused unit test.
5. **Nonfinite qualification:** NaN, infinite, reversed and malformed CSR
   confidence intervals cannot qualify a candidate.
6. **Provider error leakage:** router failure logs retain safe routing/status
   information without echoing credential-bearing SDK error bodies.
7. **Interrupted evaluation:** each latent/image is journaled durably; resume
   verifies model, adapter, settings, code, package versions, hashes, shape and
   finite values. Original partial evidence stays in its separate directory.
8. **Missing judge mistaken for zero quality:** unavailable answers now produce
   null metrics, with unknown requirements retained in the denominator.

## Paired visual comparison

Four prompts × seeds 42/43 × baseline/candidate = 16 images, 25 steps, CFG 7,
512px. The evaluator uses actual `StableDiffusionXLPipeline` sampling, default
zero empty-negative conditioning, and the application LoRA hook. Baseline is
the same NF4 backbone with adapters disabled. It does not compare production
fp16 Turbo or the complete plan/verify/repair loop.

**Completed:** all 16 images decoded, finite and nonblank, with recorded
latent/image hashes and the exported candidate activated. Peak reserved memory
was 2.366 GB (allocated 2.217 GB); elapsed time was 735.03 seconds. CPU
regression checks overlapped this run, so its duration is not an isolated
latency benchmark. [Generation evidence](ml_paired_generation_2026-10-15.json),
[execution log](ml_evaluation_2026-10-15.log),
[image gallery](ml_adapter_gallery_2026-10-15.html).

All sixteen PNGs were inspected directly. These are nonblind assistant
observations, not calibrated independent human labels:

| Prompt / seed | Baseline observation | Candidate observation |
|---|---|---|
| Three red apples / 42 | Three red apples in a row; neighboring apples touch | Three red apples remain; clear separation still needs interpretation |
| Three red apples / 43 | **Four** apples | **Four** apples; count defect persists |
| Red car left of blue bus / 42 | Blue and red bus-like vehicles; requested red car absent | Blue bus-like vehicle left of red bus-like vehicle; red car still absent |
| Red car left of blue bus / 43 | One red bus; requested car and blue bus absent | One red bus; same class/color binding failure |
| One cat left of one dog / 42 | One cat, no visible dog | One cat, no visible dog |
| One cat left of one dog / 43 | Four animal drawings in repeated compartments | Four animal drawings remain; exact-one-of-each composition not established |
| Two green bottles, no red / 42 | Two green bottles; no red bottle visible | Two green bottles; no red bottle visible |
| Two green bottles, no red / 43 | **One** green bottle | **One** green bottle; count defect persists |

The candidate changes details but does not repair these clear paired failures.
This pilot provides no demonstrated hallucination-reduction result.
[Inspection notes](ml_visual_review_2026-10-15.json) bind each observation to
the corresponding image hashes. Four prompts/two seeds are below qualification
requirements; independent labels remain unavailable. Full-contract CSR stays null.

The earlier exploratory manual Euler loop in `evaluation_b37d64245406` used
encoded empty-prompt negatives rather than default zero conditioning. Its
images are excluded from this comparison. They showed persistent count,
class, and attribute-binding errors, not a positive reduction result.

### Independent vision dependency

The earlier blind review of exploratory images obtained no usable answers:
72 requirements unknown and zero fully answered pairs. Its corrected
`vision_review.json` has null metrics. The October 14
[dependency probe](ml_vision_dependency_probe_2026-10-14.json) found Ollama
connection failure, NVIDIA timeout and Gemini HTTP 503. A new
[single-image probe on October 15](ml_vision_dependency_probe_2026-10-15.json)
reproduced all three failures in 42.29 seconds. The native-pipeline images
therefore have no independent vision review; no full batch was repeated to
manufacture a quality score.

## Research decisions and next experiment

**Implemented:** quantized frozen weights plus adapters, sequential frozen-model
caching, FP32 VAE, accumulation/checkpointing and Min-SNR. The
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
supports training adapters over quantized weights; the actual SDXL CUDA results
establish this project's feasibility. The
[SDXL guide](https://huggingface.co/docs/diffusers/en/training/sdxl) supports
caching and Min-SNR and warns of overfitting.

**Backbone choice:** naive diffusion-objective tuning can degrade distilled
few-step models, as described by [PSO](https://arxiv.org/abs/2410.03190).
The pilot therefore trains SDXL base. PSO is not implemented or benchmarked here;
this adapter is not silently attached to Turbo or its four-step sampler.

**Inference from failures and label inventory:** natural-image color binding
and negation need stronger, correct supervision. The next controlled experiment
should add rights-cleared instance-bound color labels and contract-labeled
preferred/rejected pairs, retain fresh grouped tests, and compare CSR by
failure category across three or more seeds.
[Diffusion-DPO](https://arxiv.org/abs/2311.12908) offers a preference objective;
its reported preference/alignment gains do not establish FrameGenX CSR. It
requires a separate 6 GB feasibility check and constraint-specific pairs.

[VAW's license](https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/LICENSE.md)
restricts its materials to noncommercial research and excludes commercial-product
development; it was not added. [OVAD](https://github.com/OVAD-Benchmark/ovad-benchmark-code)
is an attribute benchmark: keep its benchmark labels for evaluation if reporting
OVAD results. Neither dataset is claimed trained or validated. Source/image
license obligations remain separate from repository-code licenses.

## Verification scope

The [final local regression](ml_regression_2026-10-15.log) passed **234 tests**
with 15 dependency deprecation warnings (SQLAlchemy loader strategy and
Torch JIT on Python 3.14). The earlier focused training/evaluation run passed
43 tests. `pip check` found no broken requirements and Python compilation
passed. The optional MLflow integration is explicitly skipped when its optional
dependency is absent; it was installed and exercised in this environment.
Its final focused [tracking test](ml_tracking_gate_2026-10-15.log) also passed.
[Fresh artifact/tracking audit](ml_evidence_manifest_2026-10-15.json) rechecked
the dataset images and adapter hash and confirmed MLflow `FINISHED` with
quality `unverified`. A [host-memory sample](ml_evaluation_process_probe_2026-10-15.json)
during generation observed a 5.021 GB process peak working set and 3.759 GB
free physical memory on the 16 GB host; this is not a complete memory trace.
The complete visual comparison is recorded above. The actual
[completed-run resume check](ml_evaluation_resume_2026-10-15.log) exited 0 after
validating all sixteen latent/image pairs and reused the existing result
without model generation. Unit coverage additionally rejects changed or
corrupted partial progress; an interrupted partial run was not separately
replayed on CUDA in this continuation. All code snapshots matched their saved
hashes, and the static gallery contained sixteen unique image links with no
missing local targets. This link check is not a browser/accessibility pass.
Browser automation remains unavailable because the earlier approval policy
rejected the local frontend surface. Docker's Linux daemon was unavailable.
This continuation claims no browser, WCAG, container, cross-platform or remote
CI pass. No broad cleanup, commit, push, publication or default activation was
performed in the existing dirty workspace.

Reproduction: [Local constraint-adapter training](../docs/15_CONSTRAINT_TRAINING.md).
