# Product Positioning

## 1. Who FrameGenX Serves
FrameGenX is **not** built for casual artists or prompt engineers looking for aesthetic inspiration. 

It is built specifically for:
- **Systems Engineers & Synthetic Data Scientists**: Who require massive datasets of precisely constrained images (e.g., "Exactly 3 cars, with occlusion, matching these exact bounding boxes") and lose money on manual QA when models hallucinate.
- **Autonomous Agents**: LLMs that require a reliable, verifiable "hand" to draw what they conceive without human supervision.

## 2. The 10-Minute Value Proposition
Why would a developer star this repo and clone it?
**Within 10 minutes of cloning, a developer gets a verifiable visual generation API that guarantees the output matches their logical intent.**

They do not get a blank canvas to type prompts into. They get a Python library where they can submit a JSON array of specific objects, and the system guarantees (via internal retry loops and mathematical verification) that the returned image contains exactly those objects. It solves the immediate pain point of babysitting diffusion models.

## 3. Adoption Barriers
- **Compute Cost**: Running an LLM, a Diffusion Model, and multiple VLMs sequentially is resource-intensive.
- **Latency**: Generating a verified image might take minutes locally if it requires multiple autonomous repair loops.
- **Complexity of Use**: Users must learn to trust the JSON contract rather than endlessly tweaking natural language prompts.
