# Why Not ComfyUI?

## The Elephant in the Room
It is entirely possible to build a pipeline utilizing SDXL, multiple ControlNets, GroundingDINO, Florence-2, and OCR inside ComfyUI. Given this, why does FrameGenX exist? What makes it different from simply sharing a `.json` ComfyUI workflow?

## 1. Human Canvas vs. Autonomous Compiler
**ComfyUI is a tool for humans.** It is designed for artists and tinkerers who want to visually wire nodes together, manually paint masks, and visually inspect the final output to decide if it looks good.

**FrameGenX is a tool for systems.** It is an API-first compiler designed for autonomous agents and data pipelines. FrameGenX assumes there is no human looking at the screen. It automates the node-wiring logic entirely in code, generates the spatial masks mathematically from text, and evaluates the final output autonomously.

## 2. The Autonomous Repair Loop
In ComfyUI, if an image fails verification (e.g., GroundingDINO detects only 2 apples instead of 3), the workflow stops. A human must adjust the prompt, tweak a ControlNet weight, or redraw a mask, and click "Queue Prompt" again.

FrameGenX introduces **Autonomous Repair Loops**. When the Verification Layer scores a failure, the system automatically feeds the discrepancy report back to the Scene Planner. The system mathematically adjusts the bounding boxes, tweaks the enforcement thresholds, and regenerates the image until it passes, completely independent of human input.

## 3. Contract-Driven, Not Graph-Driven
ComfyUI workflows are extremely brittle. Passing data into a complex ComfyUI graph via API often requires reverse-engineering the node IDs and manually injecting prompt strings into specific nodes.

FrameGenX is driven by a strictly typed `contract_v2.json`. The user (or the autonomous agent) only needs to care about defining the logical constraints: `"I need 3 red apples at these coordinates."` FrameGenX dynamically orchestrates the underlying models—loading and unloading ControlNets as needed—without the user needing to understand the underlying computational graph.
