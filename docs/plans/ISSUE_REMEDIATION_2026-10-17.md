# Issue remediation and dataset execution

The owner requested fixes, necessary training downloads and important UI approval
choices on 2026-10-17. This work continues the approved stack roadmap; it does not
retroactively qualify the failed Cards study or erase its results.

## Tasks and order before implementation

| Task | Work | Depends on | Acceptance evidence |
| --- | --- | --- | --- |
| F01 | Inventory present failures, source entrypoints and dataset assets | current reports | bounded issue register and storage inventory |
| F02 | Add one process-safe hardware lease before model probing/loading | F01 | independent processes/threads exclude each other; exception cleanup; guarded entrypoints |
| F03 | Remove model construction CUDA probes and release resident models before surrendering a lease | F02 | constructors remain CUDA-free; run cleanup is guaranteed |
| F04 | Make API contention a visible typed outcome rather than competing inference | F02 | simultaneous API/CLI/training requests cannot initialize a second device owner |
| F05 | Expand labeled training data with provenance and immutable prior split reservations | F01 | image hashes, licenses, annotations, no old validation/test image enters training |
| F06 | Acquire the remaining public I-HallA metadata and inspect image availability | F01 | exact source URLs/hashes; training versus evaluation use explicit |
| F07 | Diagnose and correct any confirmed acquisition/validation defects | F05/F06 | malformed, interrupted, unsafe and split-conflict tests |
| F08 | Freeze a separate next experiment protocol before any new fitting | F02/F05 | compatible objective/backbone, bounded compute, validation-only tuning, all outcomes retained |
| F09 | Prepare human label-review UI decision | F01 | reviewable layouts and explicit owner choice below |
| F10 | Implement approved UI choice and truthful evidence feedback | owner UI choice | state and keyboard tests; rendered review only with permitted browser access |
| F11 | Run targeted and broad regression, integration and resource checks | fixes | retain failures as well as final results |
| F12 | Update task ledger and reproducible handoff with actual limits | F11 | no success label inferred from downloads, loss or operational completion |

The larger C/E/D/X roadmap retains its independent pending gates. F02-F04 only
close the portions of E1 for which actual entrypoint and process evidence exists.
No new GPU fitting starts while the device ownership gate is incomplete.

## UI decision ready for owner review: calibration review

This adds human review of verifier evidence. It does not change the already
approved inline contract editor or turn a human label into automatic model
qualification. Existing Studio, History and Benchmarks remain available.

### A. Review inside Benchmarks (recommended)

Add a **Human review** section to Benchmarks. Select an existing local run and
attempt. Show the final image, the exact extracted requirements and the machine
verdict beside it. Each requirement has **Pass / Fail / Cannot judge** plus an
optional note. Show image identity, reviewer label and an explicit **Save review**
action. Saved reviews are versioned local records, separate from machine evidence.
The summary states which labels are missing and never converts an unreviewed item
to pass. Changing the attempt restores its own labels; unsaved edits warn before
navigation. Keyboard focus follows the requirement list; status announces saves.

```
Benchmarks / Human review     Run [select]  Attempt [select]
+----------------------+--------------------------------------+
| final image          | Requirement             Machine      |
| zoom / original      | 2 green bottles         Unverified   |
| image SHA            | [Pass] [Fail] [Cannot judge]          |
|                      | Note [                              ] |
+----------------------+--------------------------------------+
Reviewer [local label]    0 / 3 reviewed       [Save review]
```

Benefits: keeps evaluation in its existing workspace, small navigation change.
Cost: Benchmarks becomes denser; review starts from an existing run.

### B. Dedicated Review tab

Use the same image, requirement labels and save semantics in a new top-level
**Review** tab with an explicit queue of unreviewed attempts and progress within
that queue. This suits repeated annotation sessions but adds a navigation section
and queue state. Existing benchmark charts link into the queue.

### C. Keep annotation in files for now

Provide versioned JSON review templates and validation through a local command.
The app displays imported labels read-only. Less interface work, more manual
effort and higher risk of editing the wrong image identity.

All three choices preserve machine evidence and keep held-out labels out of
generator training. The owner chooses the interface location; silent approval is
not inferred.

## Constraints on new data and experiments

- Reuse valid archives already on disk. Inspect storage before downloading.
- Positive generator targets must be factual/annotation-backed examples.
- Preserve previous train/validation/test reservations across new datasets.
- I-HallA generated weird images are evaluation/negative evidence, never positive
  diffusion targets. Dataset availability is distinct from training permission.
- Keep source revisions, notices, byte counts and SHA256 in acquisition records.
- Paid VQA, remote publication and hosted accounts require explicit scope.
- Preserve the v1 Cards negative result; a new parameterization is a new study.
