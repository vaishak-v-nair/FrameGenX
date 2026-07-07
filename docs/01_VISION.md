# FrameGenX: Vision and Strategy

## 1. Mission
To transition generative visual AI from a probabilistic art form into a deterministic engineering discipline. FrameGenX replaces ambiguous "Prompt-Driven Generation" with strictly verifiable "Contract-Driven Generation," enabling autonomous agents and data pipelines to rely on visual outputs with mathematical certainty.

## 2. Long-Term Vision
FrameGenX envisions a future where generating a highly constrained, multi-character scene is as reliable as compiling code. We aim to build the foundational open-source framework for "Agentic Image Generation." In this future, visual generation isn't a final output reviewed by a human; it is a verifiable programmatic step within larger autonomous systems—such as synthetic data pipelines, game asset generation, and algorithmic storyboarding.

## 3. Core Philosophy
1. **Explicit Over Implicit**: Natural language is inherently ambiguous. Relying on massive parameters to "guess" spatial relationships and structural constraints is inefficient and unscalable. Constraints must be parsed into explicit, structured contracts prior to generation.
2. **Contract-Driven Generation**: A generation is only successful if it satisfies the contract. If a constraint is not in the contract, it does not exist. If the generated image violates the contract, the image is rejected.
3. **Verification Over Hope**: We do not trust the generator. We verify its output against the initial contract using independent vision-language models and object detectors.
4. **Assume Failure**: Hallucination reduction is achievable; hallucination elimination is a myth. The system must be designed around the assumption that the generator will fail, and it must possess the mechanisms to catch and correct those failures through autonomous repair loops.

## 4. Strategic Direction
- **Phase 1: Formalization**: Establish the JSON contract schema and benchmark methodology. Focus on the smallest MVP: **Count consistency**. Prove that FrameGenX can reliably generate exactly *N* objects when state-of-the-art models fail.
- **Phase 2: Modular Pipeline**: Construct the uncoupled pipeline (Intent -> Plan -> Draft -> Enforce -> Verify) as a reference implementation.
- **Phase 3: Agentic Correction**: Introduce feedback loops where the verification layer scores the image and guides the planner to adjust constraints autonomously.
- **Phase 4: Beyond Images**: Extend the contract schema across the temporal dimension, enabling character persistence and multi-frame consistency for storyboards and video.
