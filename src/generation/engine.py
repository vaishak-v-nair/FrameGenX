import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
from src.utils import free_memory
from src.generation.model_source import resolve_model_source
from src.core.stage import Stage, register_stage
from src.core.context import RunContext
from src.core.device import require_device_lease
import warnings
from typing import Dict, Optional, Callable

# Suppress cosmetic Diffusers/Transformers warnings
warnings.filterwarnings("ignore", message=".*modules in AutoencoderKL that should be kept in float32.*")
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")


class GenerationEngine:
    """
    Unified Generation Engine, SDXL-Turbo backbone.

    ensure_loaded(pre_offload_hook): constructs the pipe, runs the hook
    (LoRA loading MUST happen before cpu offload — loading into an
    already-offloaded pipe is a known diffusers footgun), THEN calls
    enable_model_cpu_offload.
    """
    def __init__(self, config: dict):
        self.config = config
        # Match the default four-step, guidance-zero generation settings.
        self.base_model = self.config.get("models", {}).get("generator", "stabilityai/sdxl-turbo")
        self.controlnet_model = self.config.get("models", {}).get(
            "controlnet", "diffusers/controlnet-canny-sdxl-1.0"
        )

        self.device = "cpu"  # resolve lazily after acquiring the common device lease
        self.dtype = torch.float32
        self.pipe = None
        self._pipe_has_controlnet = False
        self._pre_offload_hook: Optional[Callable] = None
        # "model": whole submodels swap to GPU (fast, needs ~7GB free RAM
        # for the plain pipe). "sequential": layers stream one at a time
        # (lower peak RAM, ~4x slower — 45s/step measured on this box).
        self._offload_mode = self.config.get("generation", {}).get("offload", "sequential")
        self._pipe_offload_mode = None
        # Set after a CUDA OOM with ControlNet: on a 6GB card with a browser
        # holding VRAM, unet+controlnet don't fit — drop layout control for
        # the rest of the process instead of crashing the run.
        self._controlnet_oom = False

    def _resolve_offload(self, with_controlnet: bool) -> str:
        """Plain SDXL-Turbo pipe ~6.7GB RAM -> fast model offload fits the
        10GB VM. The ControlNet pipe adds ~2.5GB -> model offload thrashed
        the VM into swap (45s+/step, 18-min attempts, 2026-08-24), so it
        always streams sequentially. Speed where it's safe, safety where
        it's not."""
        if with_controlnet and self._offload_mode == "model":
            return "sequential"
        return self._offload_mode

    def register_pre_offload_hook(self, hook: Callable):
        """AdaptStage registers LoRA loading here; fires between pipe
        construction and enable_model_cpu_offload."""
        self._pre_offload_hook = hook

    def ensure_loaded(self, with_controlnet: bool = True):
        require_device_lease()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        resolved_mode = self._resolve_offload(with_controlnet)
        if (self.pipe is not None and self._pipe_has_controlnet == with_controlnet
                and self._pipe_offload_mode == resolved_mode):
            return
        if self.pipe is not None:
            self.teardown()

        # RAM budget: the pipe needs ~7GB resident under model offload.
        # Evict the CLIP singleton first — it reloads in seconds from page
        # cache, while a RAM-starved VM swap-thrashes (45s+/step).
        from src.utils import release_clip
        release_clip()

        # Enable TF32 for faster generation on RTX 30-series
        torch.backends.cuda.matmul.allow_tf32 = True

        if with_controlnet:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            controlnet = ControlNetModel.from_pretrained(
                resolve_model_source(self.controlnet_model, "config.json"),
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                resolve_model_source(self.base_model),
                controlnet=controlnet,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
        else:
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                resolve_model_source(self.base_model),
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
        self._pipe_has_controlnet = with_controlnet

        if self._pre_offload_hook is not None:
            self._pre_offload_hook(self.pipe)

        # CRITICAL memory optimizations for 6GB VRAM (after LoRA hook)
        if resolved_mode == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.enable_model_cpu_offload()
            # Attention slicing: compute attention in slices instead of one
            # giant activation. Mandated by AGENTS.md for the 6GB card; also
            # dampens the VRAM spikes that destabilize WSL2 GPU passthrough
            # under load. (Sequential offload already minimizes VRAM; slicing
            # there would add Python overhead to an already-slow path.)
            self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self._pipe_offload_mode = resolved_mode

    def teardown(self):
        """Full destroy — used between benchmark configs (one process per
        config makes this mostly moot, but keeps single-process use safe)."""
        self.pipe = None
        self._pipe_has_controlnet = False
        self._pipe_offload_mode = None
        free_memory()

    def _build_control_image(self, global_mask: Image.Image, mode: str,
                             width: int, height: int) -> Optional[Image.Image]:
        if mode == "none":
            return None
        if global_mask is None:
            global_mask = Image.new("L", (width, height), color=0)
        if mode == "canny":
            # A canny ControlNet expects an EDGE MAP. The legacy "box" mode
            # fed filled white rectangles with no edge extraction (design
            # review D11) — run cv2.Canny so the conditioning matches what
            # the ControlNet was trained on.
            import cv2
            arr = np.array(global_mask)
            edges = cv2.Canny(arr, 100, 200)
            return Image.fromarray(edges).convert("RGB")
        return global_mask.convert("RGB")  # legacy "box" mode

    def generate(self, prompt: str, gen_config: dict, control_hints: dict,
                 masks: Dict[str, Image.Image], output_path: Path, entities: list = None) -> dict:
        if gen_config.get("region_controller") == "fusion":
            from src.generation.controlled import generate_controlled
            regional_plan = control_hints.get("instance_regions", {})
            if not regional_plan.get("enabled"):
                raise ValueError("Requested instance fusion is unsupported for this contract: " +
                                 regional_plan.get("reason", "missing regional plan"))
            if self.pipe is not None:
                self.teardown()
            return generate_controlled(self.base_model, prompt, gen_config,
                regional_plan, output_path, self._pre_offload_hook)
        width = gen_config.get("width", 512)
        height = gen_config.get("height", 512)
        mode = control_hints.get("control_image_mode", "none")
        if self._controlnet_oom:
            mode = "none"
        control_image = self._build_control_image(masks.get("global_mask"), mode, width, height)
        use_controlnet = control_image is not None

        self.ensure_loaded(with_controlnet=use_controlnet)
        assert self.pipe is not None, "Pipeline must be loaded before generation"

        seed = gen_config.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        # SDXL-Turbo: 1-4 steps, cfg 0.0. SDXL-base: 25+ steps, cfg 7.0.
        # The gen_config dict is populated by PlanStage from config.yaml,
        # so the correct defaults depend on which model is configured.
        inference_steps = gen_config.get("steps", 4)
        cfg_scale = gen_config.get("cfg_scale", 0.0)

        # Use Diffusers' prompt encoder: it honors sequential offload hooks.
        # Compel 2.x produced meta embeddings with this 6GB offload mode.
        negative_prompt = "blurry, distorted, ugly, bad anatomy, bad proportions, bad layout, artifacts, noisy, watermark"
        conditioning, neg_conditioning, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=cfg_scale > 1.0,
            negative_prompt=negative_prompt,
        )

        kwargs = dict(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_conditioning,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=inference_steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=generator,
        )
        if use_controlnet:
            kwargs["image"] = control_image
            kwargs["controlnet_conditioning_scale"] = control_hints.get("control_weight", 0.8)

        # OOM fallback is done OUTSIDE the except block: the in-flight
        # exception's traceback pins the old pipe's forward-pass frames, so
        # rebuilding inside the handler doubles RAM and dies on a 16GB box.
        oom_fallback = False
        out = None
        
        # Apply Regional Latent Conditioning
        if entities:
            from src.generation.attention import RegionalAttentionProcessor, find_token_indices
            token_masks = {}
            for entity in entities:
                bbox = entity.get("bounding_box")
                if not bbox:
                    continue
                target_words = []
                if "class_name" in entity:
                    target_words.append(entity["class_name"])
                if "attributes" in entity:
                    target_words.extend(entity["attributes"])
                
                if target_words:
                    indices = find_token_indices(self.pipe.tokenizer, prompt, target_words)
                    for idx in indices:
                        if idx not in token_masks:
                            token_masks[idx] = []
                        token_masks[idx].append(bbox)

            if token_masks:
                self.pipe.unet.set_attn_processor(RegionalAttentionProcessor(token_masks))
            else:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        else:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            
        try:
            out = self.pipe(**kwargs)
        except torch.cuda.OutOfMemoryError:
            if not use_controlnet:
                raise  # nothing left to shed
            oom_fallback = True

        if oom_fallback:
            print("[generate] CUDA OOM -> retrying without ControlNet "
                  "(layout control skipped, verification unaffected)")
            self._controlnet_oom = True
            mode = "none"
            use_controlnet = False
            kwargs.pop("image", None)
            kwargs.pop("controlnet_conditioning_scale", None)
            self.teardown()  # traceback gone -> old pipe actually frees here
            self.ensure_loaded(with_controlnet=False)
            if seed is not None:
                kwargs["generator"] = torch.Generator(
                    device=self.device).manual_seed(int(seed))
            out = self.pipe(**kwargs)

        image = out.images[0]  # type: ignore
        image.save(output_path)
        # Use Diffusers' final VAE decode (including its precision handling).
        # The former callback decoded twice and produced NaNs/black previews
        # with the fp16 SDXL VAE under sequential offload.
        attempt = output_path.stem.split("_")[-1]
        preview_path = output_path.parent / f"step_{attempt}_{inference_steps - 1}.png"
        image.save(preview_path)

        meta = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": gen_config.get("steps"),
            "cfg_scale": gen_config.get("cfg_scale"),
            "seed": seed,
            "preview_source": "final_pipeline_decode",
            "base_model": self.base_model,
            "resolved_base_model": resolve_model_source(self.base_model),
            "controlnet": self.controlnet_model if use_controlnet else None,
            "control_image_mode": mode,
            "oom_fallback": "controlnet_dropped" if self._controlnet_oom else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        free_memory()
        return meta


@register_stage
class GenerateStage(Stage):
    name = "generate"

    def __init__(self, config):
        super().__init__(config)
        self.engine = GenerationEngine(config)

    def run(self, ctx: RunContext) -> None:
        print(f"[generate] attempt {ctx.attempt}: '{ctx.final_prompt[:80]}...'")
        output_path = ctx.run_dir / f"generated_{ctx.attempt}.png"
        meta = self.engine.generate(
            ctx.final_prompt, ctx.gen_config, ctx.control_hints, ctx.masks, output_path,
            entities=ctx.scene_plan.get("entities", []) if ctx.scene_plan else []
        )
        # carry over any adapt-stage records (e.g. lora_load_failed)
        meta.update({k: v for k, v in ctx.generation_meta.items() if k.startswith("lora")})
        ctx.generation_meta = meta
        ctx.save_artifact(f"generation_meta_{ctx.attempt}.json", meta)
        if ctx.is_benchmark and ctx.config_name in ("rag_lora", "full"):
            expected = {
                adapter.get("name", "adapter")
                for adapter in (ctx.config.get("lora") or {}).get("adapters", [])
                if adapter.get("enabled", True)
            }
            activated = {
                item.get("name") for item in meta.get("lora_load_report", [])
                if item.get("activated")
            }
            if not expected or expected != activated:
                raise RuntimeError(
                    f"LoRA ablation invalid: expected active adapters {sorted(expected)}, "
                    f"observed {sorted(activated)}"
                )
