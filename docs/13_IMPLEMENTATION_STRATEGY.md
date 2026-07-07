# Implementation Strategy

To transform FrameGenX from a research prototype into a robust open-source platform, we must adhere to a strict implementation philosophy. The following priorities dictate all engineering decisions:

## 1. How Do We Build It? (The MVP First)
Many AI projects fail because they try to solve everything at once. We will not try to solve general "Image Correctness."

**The MVP focuses exclusively on: Count Consistency.**
The first implementation priority is proving that the FrameGenX pipeline can reliably generate exactly *N* distinct objects (e.g., "Exactly 7 apples") with a statistically significant higher Constraint Satisfaction Rate than a baseline SDXL model.

## 2. Priority Order
1. **The Contract Layer**: Implement the rigid JSON schema parser.
2. **The Verification Layer**: Implement the autonomous scoring mechanism using GroundingDINO to count objects and verify bounding boxes. This must exist *before* the generator is refined, so we can measure baseline failures.
3. **The Autonomous Repair Loop**: Implement the logic that detects a failed count (e.g., 6 apples instead of 7) and mathematically forces the Scene Planner to adjust its masks and regenerate.
4. **The Latent Drafter**: Implement the diffusion backbone (SDXL/Flux) as a strictly obedient rendering engine.

## 3. Tradeoffs
- **Speed vs. Accuracy**: We trade latency for accuracy. Running multiple LLMs, VLMs, and Diffusion models in a loop is slow. The system assumes a 6GB VRAM constraint, meaning aggressive memory swapping is prioritized over fast generation times.
- **Aesthetics vs. Logic**: If forced to choose, the system will output a less aesthetically pleasing image that mathematically satisfies the contract over a beautiful image that hallucinates a constraint.

## 4. Scientific Validity & Benchmarking
Every architectural choice must be justified mathematically. If a new module is introduced, we must isolate its effect on the Constraint Satisfaction Rate (CSR). Before any code is merged, the pipeline must be run against the `benchmark_dataset.json` to ensure no regression in numeracy or spatial accuracy.
