# Known Failure Cases & Risk Assessment

## 1. Known Limitations
- **Constraint Overload**: Generating scenes with more than 5 distinct, highly-constrained entities introduces severe artifacting in the current SDXL/ControlNet backend due to VRAM limitations forcing aggressive, low-resolution masks.
- **Micro-Details**: Small text and microscopic attributes (e.g., "a tiny ring on the character's left pinky finger") often fail verification because the object is too small for GroundingDINO to confidently detect, triggering a false-negative loop.
- **Verifier Bias**: Vision-Language Models (like Florence-2) occasionally exhibit the same biases as the diffusion models they are verifying. If both models hallucinate the same logical flaw, the verifier will incorrectly pass the image.

## 2. Expected Failures
- **Infinite Retry Loops**: If the user requests a physically impossible or contradictory contract (e.g., "A cube inside a solid sphere"), the generation engine will continually fail the verification layer. Without a hard timeout, the system will enter an infinite repair loop.
- **Performance Bottlenecks**: Running LLM Extraction, Diffusion Generation, and VLM Verification locally on a 6GB VRAM GPU is inherently slow. The pipeline trades speed for accuracy. Time-sensitive applications will fail to meet latency requirements without scaling hardware.

## 3. Edge Cases
- **Abstract Concepts**: The JSON contract requires concrete entities and spatial relationships. Abstract prompts ("The feeling of melancholy on a Tuesday") cannot be parsed into bounding boxes, rendering the FrameGenX pipeline ineffective and overly restrictive.
- **Non-Standard Aspect Ratios**: Extreme panoramic or vertical aspect ratios distort the 2D bounding box calculations in the Scene Planner, leading to warped ControlNet inputs.

## 4. Risk Assessment
- **Dependency Risk**: FrameGenX heavily relies on third-party models (spaCy, SDXL, GroundingDINO). Upstream changes to these architectures could necessitate complete rewrites of the Scene Planner and Verification logic.
- **Compute Overhead**: The primary risk to adoption is the immense compute required for the verification loop. If foundation models (like Midjourney v7 or SD4) suddenly solve spatial reasoning intrinsically, the heavy FrameGenX architecture becomes obsolete.
