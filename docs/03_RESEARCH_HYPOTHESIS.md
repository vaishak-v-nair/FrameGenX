# Research Hypothesis

## 1. Primary Hypothesis
**Contract-driven generation, coupled with constraint satisfaction scoring and autonomous repair loops, significantly increases logical accuracy compared to traditional prompt-driven monolithic models.**

By treating the diffusion model as a "dumb rendering engine" constrained by explicitly calculated structural hints (e.g., ControlNet masks, bounding boxes), we hypothesize that spatial and numeracy errors can be mitigated at the architectural level.

## 2. Core Research Contributions
If FrameGenX disappeared tomorrow, the following research ideas should disappear with it:
1. **Contract-Driven Generation**: The paradigm shift from passing natural language to a diffusion model, to passing a strictly typed, versioned JSON contract that explicitly defines all entities and spatial relationships.
2. **Constraint Satisfaction Scoring**: A deterministic method of scoring an image based purely on how many explicit constraints it satisfies, using independent Vision-Language Models, completely ignoring aesthetic "vibes."
3. **Autonomous Repair Loops**: The concept of a visual compiler that catches its own errors. When an image fails the Constraint Satisfaction Score, the system autonomously modifies its own spatial masks and regenerates the image until it mathematically passes.

## 3. Smallest Possible MVP
To validate the research contribution, we must isolate the smallest measurable problem.
**The MVP is: Count Consistency.**
Instead of solving "Image Correctness" entirely, the MVP aims to prove that FrameGenX can reliably generate exactly *N* distinct objects (e.g., exactly 7 apples) with a statistically significant higher success rate than baseline SDXL or Flux, purely through contract-driven enforcement and verification.

## 4. Validation Criteria
To validate these hypotheses, FrameGenX must be evaluated against a reproducible baseline.
- The input must be a strictly defined dataset of complex prompts designed to trigger known failure modes (specifically numeracy).
- The output must be evaluated purely on **constraint satisfaction**, not aesthetic quality.

## 5. Failure Conditions
The hypotheses will be considered invalidated if:
1. The FrameGenX pipeline shows no statistically significant improvement in Constraint Satisfaction Rates (CSR) over the baseline monolithic model when using modern architectures like Flux.1 or SD3.5.
2. The independent verification layer proves too unreliable or biased to accurately score the generated outputs, rendering the automated benchmarking meaningless.
