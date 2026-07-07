# Product Roadmap

## 1. MVP: Count Consistency
The Minimum Viable Product focuses on solving one measurable problem: Numeracy in Diffusion Models.
- **Goal**: Prove that FrameGenX can reliably generate exactly *N* objects when state-of-the-art models fail, utilizing contract-driven generation and autonomous repair loops.
- **Features**: Intent extraction to `contract_v2.json`, 2D Scene Planning, SDXL Generation, GroundingDINO verification.
- **Constraint**: Must run on consumer hardware (6GB VRAM) via aggressive memory swapping.

## 2. Research Version: Spatial and Attribute Accuracy
Once count consistency is solved, the focus shifts to rigorous spatial and attribute benchmarking.
- **Features**: Integration of Florence-2 for attribute isolation (solving prompt bleed), automated benchmark suite generation for spatial relationships (e.g., "Object A behind Object B").
- **Milestone**: Publish a whitepaper demonstrating a statistically significant increase in overall Constraint Satisfaction Rate vs. baseline models.

## 3. Production Version: Speed and Agentic Integration
Focus shifts to speed, reliability, and external integration.
- **Features**: Swappable generation backends (API integrations for Midjourney, DALL-E, or cloud-hosted Flux), parallel processing of the verification layer, robust API wrapper for agentic consumption.
- **Milestone**: A developer can clone the repo and within 10 minutes have a reliable visual API serving an autonomous LLM agent without human QA.

## 4. Enterprise Version: Temporal and Character Consistency
Focus on multi-frame continuity for specific industry verticals.
- **Features**: Temporal Scene Planning (3D camera positioning, persistent character IDs across frames), custom LoRA injection at the Scene Planner level.
- **Milestone**: Achieving zero-shot, multi-frame character continuity for synthetic data generation and autonomous storyboarding.
