import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from src.utils import free_memory
from typing import Dict, Any

class GenerationEngine:
    """
    Unified Generation Engine (Phase 5).
    Replaces the separate Drafter and Enforcer to minimize VRAM loading/unloading.
    Accepts synthetic masks from the Scene Planner directly.
    """
    def __init__(self, config: dict):
        self.config = config
        self.base_model = self.config.get("models", {}).get("generator", "stabilityai/sdxl-turbo")
        self.controlnet_model = "diffusers/controlnet-canny-sdxl-1.0" # using canny for segmentation bounds
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = None

    def _load_model(self):
        if self.pipe is None:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model, 
                torch_dtype=self.dtype
            )
            
            # Load Base Pipeline with ControlNet
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.base_model,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )
            
            # CRITICAL for 6GB VRAM
            self.pipe.enable_model_cpu_offload()

    def _unload_model(self):
        # Memory optimization: Do not destroy the pipeline. Let enable_model_cpu_offload handle paging.
        # self.pipe = None
        free_memory()

    def generate(self, prompt: str, gen_config: dict, control_hints: dict, masks: Dict[str, Image.Image], output_path: Path) -> dict:
        self._load_model()
        
        width = gen_config.get("width", 512)
        height = gen_config.get("height", 512)
        
        # In MVP, we use the global segmentation mask
        global_mask = masks.get("global_mask")
        
        if global_mask is None:
            # Fallback to empty mask if none provided
            global_mask = Image.new('L', (width, height), color=0)
            
        # For a Canny ControlNet, providing a solid white box on black background works well enough 
        # to force the layout, as it detects the edges of the box.
        # Alternatively, a depth or segmentation controlnet could be used.
        # We will use it as the conditioning image.
        control_image = global_mask.convert("RGB")
        
        # Generate image
        image = self.pipe(
            prompt=prompt + ", high quality, detailed, realistic",
            negative_prompt="blurry, distorted, bad anatomy",
            image=control_image,
            controlnet_conditioning_scale=control_hints.get("control_weight", 0.8),
            num_inference_steps=gen_config.get("steps", 20),
            guidance_scale=gen_config.get("cfg_scale", 7.0),
            width=width,
            height=height
        ).images[0]
        
        image.save(output_path)
        
        meta = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": gen_config.get("steps"),
            "cfg_scale": gen_config.get("cfg_scale"),
            "base_model": self.base_model,
            "controlnet": self.controlnet_model,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self._unload_model()
        return meta
