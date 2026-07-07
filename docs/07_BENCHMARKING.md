# Benchmarking Framework

## 1. Benchmark Methodology
To scientifically validate the FrameGenX architecture, we must move away from subjective aesthetic evaluations (e.g., ELO ratings, "vibes") and rely purely on measurable constraint satisfaction. The benchmark measures how well a system adheres to complex instructions, not how beautiful the output is.

## 2. Dataset Design
The `benchmark_dataset.json` consists of 100 meticulously crafted prompts, evenly distributed across 5 distinct failure-mode categories.

1. **Object Count (Numeracy)**: Prompts requesting exact numbers of objects (e.g., "Exactly five apples on a table").
2. **Spatial Relations**: Prompts requiring strict positional logic (e.g., "A dog sitting on top of a television").
3. **Attribute Binding (Prompt Bleed)**: Prompts designed to confuse text encoders (e.g., "A green cube, a red sphere, and a blue pyramid").
4. **Text Rendering**: Prompts requiring specific spelling (e.g., "A neon sign that says 'OPEN 24/7'").
5. **Multi-Object Complexity**: Crowded scenes testing the model's capacity to render many distinct, interacting entities without merging them.

## 3. Metrics
- **CSR (Constraint Satisfaction Rate)**: The primary metric. The percentage of generations that pass the QA Verification layer without any critical failures.
- **TTC (Time To Completion)**: The average compute time required to produce a verified, successful image.
- **Average Repair Loops**: The average number of times the system had to autonomously correct and re-generate an image before it passed verification.

## 4. Statistical Evaluation
- **Baseline**: The raw SDXL pipeline (without FrameGenX's scene planning or verification loops) is run against the benchmark dataset. Its CSR serves as the control.
- **Experiment**: The full FrameGenX pipeline is run against the same dataset.
- **Significance**: A two-sample T-test or Chi-square test is applied to determine if the improvement in CSR is statistically significant (p < 0.05).

## 5. Reproducibility Requirements
- All benchmark prompts, expected JSON contracts, and random seeds must be open-sourced.
- The verification models (GroundingDINO, Florence-2 weights) must be explicitly versioned.
- Hardware specifications (e.g., RTX 3050 6GB) must be documented to prove that high CSR can be achieved in constrained environments through architecture, rather than brute compute.
