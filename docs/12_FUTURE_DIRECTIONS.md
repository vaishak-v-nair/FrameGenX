# Future Directions: What Happens After Images?

A good architecture survives beyond version 1. FrameGenX's true value lies in its extensibility beyond single-frame image generation. By establishing a rigid `contract_v2.json`, the system is inherently designed to scale into complex, multi-frame, and high-volume domains.

## 1. Video Generation
Current video generation models (Sora, Runway Gen-3) suffer from the same probabilistic lack of control as monolithic image models, but amplified across time. FrameGenX's explicit constraints can be extended to video by defining motion vectors and bounding box trajectories within a temporal contract, acting as a rigorous structural skeleton for temporal diffusion models.

## 2. Algorithmic Storyboards & Game Assets
Storyboarding and game asset generation require strict continuity. If an autonomous agent generates a storyboard for a cinematic sequence, FrameGenX will persist the `entity_id` across frames. The Scene Planner will calculate the geometric transformation of the bounding boxes from frame to frame, ensuring character consistency and environmental persistence without relying on human artists to "fix" the generations.

## 3. Synthetic Datasets at Scale
The most immediate commercial application post-MVP. Machine learning engineers require millions of perfectly annotated images (e.g., autonomous driving datasets). FrameGenX will evolve into a mass-production synthetic data pipeline where the JSON contract not only drives the generation but natively serves as the absolute ground-truth bounding box annotation for the resulting dataset.

## 4. Advertising Pipelines & Automation
Enterprise advertising requires strict adherence to brand guidelines (exact hex colors, specific product placement, exact logo rendering). FrameGenX will integrate custom LoRA injection at the entity level within the Scene Planner, guaranteeing that a specific product is rendered exactly as requested, verified against its brand constraints autonomously.

## 5. Agentic Workflows & Autonomous Repair Loops
FrameGenX will serve as the verifiable visual cortex for LLM agents. An agent can autonomously write a narrative, generate the JSON contracts, run the FrameGenX pipeline, and if the Verification Layer reports a failure, the agent will autonomously adjust the contract and retry—completely removing humans from the loop of visual asset creation.
