# Research extension engineering review — 2026-10-16

Source reviewer: existing independent Codex engineering agent. Static review only, no commands, training or implementation. Initial result FIXES NEEDED, four issues:

1. Four quadrants require all40 templates, not four templates. Require complete identities/dimensions/hashes and unknown for ambiguity.
2. Current SDXL LoRA TrainingConfig/objective is incompatible with a learned-variance Cards trainer. Freeze separate six-channel objective, variance mapping, detached mean VLB, units/schedule/gradient/sampler/checkpoint contract.
3. Freeze R07 evaluation definitions before R04/R05 fitting: exact seeds, thresholds, checkpoints, denominators and uncertainty.
4. A notebook-only GPU lock does not exclude existing entrypoints; E1 must precede CUDA experiments and sampling.

Amendments: complete40-template inventory; separate128RGB six-output DDPM specification with the paper's time-dependent schedule and reference-source discrepancy; fixed final-checkpoint/sample-seed/threshold/CI protocol in R03; initial experiment explicitly CPU-only, rejecting CUDA and requiring E1 for any future GPU route.

Final amendment check: **PASS — specification only**. All four corrections consistent; no remaining contradiction found in that bounded check. Cards feasibility training is not SDXL fine-tuning. I-HallA remains evaluation until separate rights/training-family separation gates pass. Runtime, accuracy and compatibility remain unverified.
