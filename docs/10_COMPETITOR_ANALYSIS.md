# Competitor Analysis

## 1. Midjourney & DALL-E 3 (Closed APIs)
- **Strengths**: Unmatched aesthetic quality, high prompt adherence for general concepts. DALL-E 3 leverages GPT-4 for excellent intent extraction.
- **Weaknesses**: Closed ecosystems. Zero explicit control over spatial layout (no bounding boxes). They fail complex numeracy tests and are fundamentally non-deterministic.
- **The FrameGenX Advantage**: Determinism and Verifiability. You cannot put Midjourney in a tight automated loop for synthetic data generation because it will silently fail. FrameGenX guarantees the output matches the logic.

## 2. Flux & SDXL (Open Weights)
- **Strengths**: State-of-the-art open-weight models. High prompt adherence and beautiful rendering.
- **Weaknesses**: Still monolithic architectures. Susceptible to prompt bleed in highly complex, crowded scenes.
- **The FrameGenX Advantage**: FrameGenX does not compete with Flux or SDXL; it *wraps* them. We view these models as swappable "Latent Drafter" backends. FrameGenX adds the contract-driven layout and the verification loop that these foundational models lack.

## 3. ComfyUI (Node-Based Workflows)
- **Strengths**: The industry standard for complex visual pipelines. Infinite flexibility. Supports SDXL + ControlNet + Florence + OCR.
- **Weaknesses**: ComfyUI is a tool for *human artists*. It requires manual wiring, manual creation of masks, and human evaluation of the final output. It is essentially a visual programming language for creatives.
- **The FrameGenX Advantage**: See `WHY_NOT_COMFYUI.md` for a deep dive. In short: FrameGenX is a compiler and an agentic loop, not a canvas for humans.
