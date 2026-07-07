# Project Thesis

## What is FrameGenX?
FrameGenX is an open-source, API-first visual generation platform that treats image synthesis as a deterministic engineering process rather than a probabilistic art form. It decouples the visual generation pipeline into intent extraction, spatial planning, constrained rendering, and mathematical verification.

## Why does it exist? (The Financial/Productivity Cost)
Currently, automated pipelines, autonomous agents, and synthetic data engineers lose massive amounts of money and productivity because text-to-image systems are not verifiable. 
When a game studio attempts to automate storyboard generation, or a data scientist generates synthetic datasets for autonomous driving, they cannot afford a 20% hallucination rate where a "red stop sign" bleeds into a "red car," or "three pedestrians" becomes "four." 
Because monolithic diffusion models silently return aesthetically pleasing but logically broken images, these industries are forced to insert costly "human-in-the-loop" QA steps. FrameGenX exists to remove the human QA step by making the generation logically verifiable.

## Who needs it?
- **Synthetic Data Engineers**: Who require absolute certainty regarding object counts, attributes, and bounding boxes for training ML models.
- **Autonomous Agents (Agentic Workflows)**: LLMs that need a "hand" to draw what they think, but require a guarantee that the generated image matches their logical intent before publishing.
- **Enterprise Automation Studios**: Production pipelines that demand zero-shot consistency over "vibes" and aesthetics.

## What problem does it solve?
It solves the problem of **Prompt-Driven Ambiguity** by replacing it with **Contract-Driven Generation**. By parsing natural language into a strict JSON contract *before* generation, and verifying the image against that contract *after* generation, FrameGenX solves the lack of deterministic reliability in visual generation.

## What does success look like?
Success is defined by the **Constraint Satisfaction Rate (CSR)**. Success looks like an autonomous agent generating 1,000 complex scenes (e.g., "Exactly 4 blue cubes and 2 red spheres") and FrameGenX autonomously repairing its own errors until 99.9% of the final outputs mathematically satisfy the contract, all without human intervention.

## What would prove the idea wrong?
The core thesis is proven wrong if:
1. Scaling up monolithic foundation models (e.g., GPT-5 + DALL-E 4, or Flux v2) inherently solves complex spatial reasoning, exact numeracy, and attribute binding without the need for an external verification loop or explicit contracts.
2. The computational overhead of running the FrameGenX verification and repair loops (LLM + Diffusion + VLM) is so high that it becomes cheaper and faster to just hire human QA testers.
3. The Verification layer (GroundingDINO, Florence-2) is fundamentally incapable of reaching the confidence thresholds required to accurately score the images, leading to infinite false-negative repair loops.
