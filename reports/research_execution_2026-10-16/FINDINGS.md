# Research integration findings

This work package implements the reviewed CPU research slice, R01–R12, with the
explicit partial/blocked states in the task board. It does not complete the
separate 30-task stack roadmap or certify the entire application.

## Corrected reliability and integrity issues

| Finding | Evidence and user impact | Correction | Verification |
| --- | --- | --- | --- |
| Report publication could briefly appear corrupt | The report is written before its manifest; a read in between tried to interpret an incomplete publication as completed evidence | Use the running/failed journal until a manifest exists; a completed journal without its manifest remains an evidence error | `test_report_publication_waits_for_manifest_without_false_corruption` |
| Report hashing preceded bounded parsing | Source inspection showed an unbounded second file read before the size-checked parser; malformed local artifacts could impose needless work | Read at most1MB plus one sentinel byte, then hash and parse the same bytes | Oversized, nonfinite, tampered and semantically invalid report tests |
| Archive destination could be an existing directory link | Member path checks did not protect a pre-existing output-directory junction/symlink; the regression creates a real Windows junction within its isolated test directory | Reject linked destination directories before creating or writing output files | `test_extraction_cannot_write_through_a_destination_directory_link`, passed on Windows |
| Interrupted reruns could retain old notebook outputs | A previously executed notebook retains later cell outputs unless they are cleared before restarting | Clear execution counts/outputs first, save each completed cell atomically, and save errors/interruption state | Notebook recovery and old-success-removal tests |
| New notebook lifecycle metadata was not persisting | The full regression recorded 3 failures because `nbformat` converts assigned dictionaries to `NotebookNode` copies; updates to the original dictionary left the saved status at running | Update the stored metadata node directly | Preserved failure log; all 3 recovery cases passed on a fresh rerun |
| Completed-run identity omitted relevant packages/source content | The first implementation compared torch/numpy versions and source hashes but did not preserve all fitting source or identify SciPy, sklearn, image decoding and tracking versions | Include Python/platform/numerical/decoder/tracking identity, copy hashed source before fitting, reject a changed source snapshot | Runtime-identity and source-change tests; final run artifact verification |
| Legacy documentation described absent features that now exist | `CURRENT_STATE_AUDIT.md` still said no tests, benchmark, SQLite or repair loop existed | Mark the undated document historical and link current authorities; narrow testing-guide claims | Source/document inspection |

These are local research/application integrity findings. The directory-link and
large-artifact cases require control over local filesystem inputs; they are not
presented as demonstrated remote exploits.

## Observed training results and failures

- The first attempt, `fc58ed37-0b74-45d3-9c90-35a9c9736eb4`, stopped at the memory
  guard during candidate fitting while broad CPU tests overlapped it. Baseline
  completed 500 steps; candidate last logged 200. The run remains failed, with no
  completed comparison. Its executed error notebook and log are retained.
- The serial trial, `d523c2eb-5f80-4a1d-ac5b-11bdefa80adb`, completed 500 steps/arm
  and all 32 samples/arm. Both arms had 0 valid, 32 invalid, 0 unknown and 0 errors.
  Its single-pair Cards-validity difference is 0; it demonstrates no reduction.
  Each rate's Wilson95% interval is [0, .10718]. The empirical paired bootstrap's
  [0,0] interval does not establish population equality or cover training-seed
  uncertainty. Sample inspection found noisy, unusable card outputs.
- That trial's synthetic-corruption classifier achieved balanced accuracy 1.0 and
  ROC AUC 1.0 on 2940 examples: 588 source grids plus 2352 named synthetic corruptions.
  Template-derived labels/features and easy synthetic negatives do not establish
  independent visual verification, natural-image accuracy or full-contract CSR.
- Integrity, source-preservation and lifecycle amendments were made while the
  serial trial used its already imported source. Its recorded fingerprint is
  retained as historical evidence. A new complete final trial uses the frozen
  amended source; it is not relabeled as an independent training-seed replication.
- Final trial `51529996-23e9-436d-91ba-2107057739e8` also completed 500 steps and 32
  samples per arm, with 0 valid images in both arms. It took 1058.51 seconds inside
  the experiment, observed peak process RSS 1.331 GB and minimum host availability
  3.123 GB. All 26 parameter tensors/arm and all 64 PNGs matched the prior same-seed
  trial exactly. All 86 saved artifacts passed the audit; the actual mounted API
  and frontend data contract checks passed without external calls or CUDA
  initialization. These are execution/reproducibility results, not image quality.

The final run/results and verification commands are recorded in the execution
summary beside this file. No failed image is removed from a denominator, used as
a positive training target, or converted into a qualified SDXL adapter.

## Exploratory denoising diagnosis

An explicitly post hoc diagnostic used the first 8 validation grids, identical
noise seeds across arms and timesteps 0, 10, 25, 50, 75, 99. It did not fit a model or
change the sampler, frozen benchmark or qualification decision.

At timestep 99, alpha-bar was 2.43e-7. Baseline/candidate epsilon MSE was
.02328/.02245; the corresponding unbounded x0 reconstruction RMSE was
309.61/304.04 in normalized [-1, 1] image units. About99.7% of predicted channels
were outside that range before clipping. The identity-noise diagnostic had MSE
2.22e-7 at this terminal timestep; it is not a useful image generator.

These observations show insufficient denoising precision at the highest-noise
stage and large amplification during x0 reconstruction. They are consistent with
the failed/noisy samples, without proving which training or architecture change
will solve them. Investigate model capacity, training duration and stable noise
parameterization on validation data in the next study. No improvement claim is
derived from the candidate's slightly smaller diagnostic error.

## Open gates

1. **General hallucination reduction:** unverified. This small unconditional Cards
   experiment is not a useful image generator and cannot establish object/color
   binding improvements for FrameGenX.
2. **SDXL activation:** blocked. RGB epsilon3/variance3 checkpoints are incompatible
   with the current four-latent-channel SDXL LoRA path. A compatible variance head,
   objective, sampler/export and resource design need a separate reviewed study.
3. **I-HallA:** public caption/history-QA metadata are inspected and held out.
   Image assets, permitted VQA, reviewed factual labels, training rights and a
   disjoint training partition are not established; no I-HallA training or score
   is claimed.
4. **Browser/accessibility:** rendered journeys and WCAG checks remain unverified.
   Earlier automatic approval review rejected browser automation. No alternative
   browser/CDP route was used to bypass that rejection.
5. **Broader runtime:** no clean install, container/hosted deployment, remote CI,
   cross-platform matrix, human calibration or common GPU gate is certified by
   these local CPU/API checks. The full-stack roadmap tracks those separately.
6. **Initial Git inventory:**2064 tracked changes include1863 run artifacts and 56 bytecode
   files. New ignores prevent further research/environment/cache noise; existing
   tracked/staged files remain untouched and still count toward the IDE limit.
7. **ChatGPT Dots:** exact service identification remains pending. No invocation
   of an unidentified service is claimed.

## Next study decision

Do not promote this Cards model or increase a hallucination-reduction claim from
its decreasing training loss. Keep the locked negative result. Before a longer
study, review model receptive field/capacity, training duration and fidelity on a
validation-only pilot. Predeclare a new protocol and compute budget before
fitting, retain the old study, then run disjoint paired evaluation. For the actual
FrameGenX generator, prioritize the approved common-device, compatible-model and
human-calibrated full-contract evidence gates. More libraries or a larger dataset
alone do not satisfy those gates.

The [pinned VSM README](https://github.com/bhosalems/VSM/blob/0972d3e97392a4ddca5c032abd935a3592b0ca41/README.md)
describes pretrained variance-head-only fine-tuning. Review that route against
SDXL's feature/output/sampler interfaces and the 6 GB budget before implementing a
generator adapter. No compatible SDXL export or resource result is claimed here.

## Ancillary tooling

The requested gstack maintenance completed to 1.91.117.0. Its setup reported ready;
the full log is `reports/gstack_update_2026-10-17.log`. Customized skill copies
were backed up by setup under
`C:/Users/vaish/.gstack/backups/skills/20261017T101931/`. This is separate from
project runtime verification.
