"""
RefineStage — Post-acceptance texture enhancement via SDXL Img2Img.

Only runs when:
  1. The draft was ACCEPTED by the reflect stage.
  2. config.yaml has `refinement.enabled: true`.
  3. The accepted image file actually exists.

Speed contract (2026-08-24): the stage REUSES the GenerateStage's live
pipe via diffusers AutoPipeline (from_pipe shares all loaded weights —
zero model reload). The previous implementation loaded a second 7GB
SDXL-base model AND ran guidance_scale=7.5, which doubles every step
(2 UNet passes/step x 20 steps = 40 passes) -> 243s measured. Reusing the
turbo pipe at cfg 0 with 6 steps costs ~6 passes and no load: ~60s.

VRAM contract:
  - Reuse path: no new weights; the existing offload hooks apply as-is.
  - Fallback path (no reusable pipe): load configured model under
    enable_model_cpu_offload() — never .to(device) on a 6GB card.
"""
import torch
from pathlib import Path
from src.core.stage import Stage, register_stage
from src.core.context import RunContext
from src.utils import free_memory
from src.generation.model_source import resolve_model_source
from PIL import Image


@register_stage
class RefineStage(Stage):
    name = "refine"

    def __init__(self, config):
        super().__init__(config)
        r_cfg = config.get("refinement", {}) or {}
        self.strength = float(r_cfg.get("strength", 0.35))
        self.num_inference_steps = int(r_cfg.get("steps", 6))
        self.guidance_scale = float(r_cfg.get("guidance", 0.0))
        self.model_id = r_cfg.get("model", "stabilityai/sdxl-turbo")
        self.device = "cpu"
        self.dtype = torch.float32
        self._engine = None

    def bind_engine(self, engine):
        """Pipeline wires the GenerateStage engine in so the accepted pipe
        can be reused instead of loading a second SDXL (243s -> ~60s)."""
        self._engine = engine

    def _build_pipe(self):
        """Reuse the generate engine's plain pipe when possible; else load
        the configured img2img model from disk (legacy path)."""
        from src.core.device import require_device_lease
        require_device_lease()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self._engine is not None and self._engine.pipe is not None \
                and not self._engine._pipe_has_controlnet:
            from diffusers import AutoPipelineForImage2Image
            print("[refine] Reusing generate pipe via AutoPipeline (no reload)")
            return AutoPipelineForImage2Image.from_pipe(self._engine.pipe)

        print(f"[refine] Loading {self.model_id} img2img (no reusable pipe)...")
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            resolve_model_source(self.model_id),
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            use_safetensors=True,
        )
        # CRITICAL: do NOT call .to(device) — that loads the entire model
        # onto GPU at once and OOMs on 6GB. cpu_offload streams submodels.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        return pipe

    def run(self, ctx: RunContext) -> None:
        if not ctx.decision or not ctx.decision.get("accept", False):
            print("[refine] Skipping: draft was not accepted.")
            return

        # Find the accepted image: generated_{attempt}.png
        accepted_path = ctx.run_dir / f"generated_{ctx.attempt}.png"
        if not accepted_path.exists():
            # Fallback: scan for any generated image
            candidates = sorted(ctx.run_dir.glob("generated_*.png"))
            if not candidates:
                print("[refine] No generated image found. Skipping.")
                return
            accepted_path = candidates[-1]

        print(f"[refine] Loading accepted image: {accepted_path.name}")
        init_image = Image.open(accepted_path).convert("RGB")

        # Flush any stray CUDA allocations, then reuse/load the pipe.
        free_memory()

        pipe = self._build_pipe()

        prompt = ctx.final_prompt or ctx.contract.get("prompt", ctx.prompt)
        print("[refine] Enhancing textures...")
        refined_image = pipe(
            prompt=prompt,
            negative_prompt=None if self.guidance_scale <= 0 else
            "blurry, distorted, artifacts, noisy",
            image=init_image,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
        ).images[0]

        refined_path = ctx.run_dir / "refined_final.png"
        refined_image.save(refined_path)
        reused = (self._engine is not None and self._engine.pipe is not None
                  and not self._engine._pipe_has_controlnet)
        # Use save_artifact for JSON metadata (RunContext has no .artifacts dict)
        ctx.save_artifact("refinement_meta.json", {
            "model": "reused_generate_pipe" if reused else self.model_id,
            "reused_generate_pipe": reused,
            "strength": self.strength,
            "steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "input_image": accepted_path.name,
            "output_image": "refined_final.png",
            "verification_status": "unverified_derivative",
        })

        # Cleanup: flush CUDA cache. The reused pipe stays resident for any
        # subsequent run in this process (free_memory only clears GPU).
        free_memory()
        print(f"[refine] Done! Saved to {refined_path}")
