import random
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, List

class ScenePlanner:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        # 512x512 is much safer for 6GB VRAM than SDXL default 1024x1024
        
    def _generate_synthetic_masks(self, entities: List[Dict[str, Any]], repair_attempt: int = 0) -> Dict[str, Image.Image]:
        """
        Translates entities into 2D geometric bounding boxes and renders segmentation masks.
        Returns a dict mapping entity class_name to a PIL Image mask.
        """
        masks = {}
        
        # Grid-based simple layout for MVP count consistency
        num_objects = len(entities)
        if num_objects == 0:
            return masks
            
        grid_size = int(num_objects ** 0.5) + (1 if (num_objects ** 0.5) % 1 > 0 else 0)
        cell_size = self.resolution // (grid_size if grid_size > 0 else 1)
        
        # Repair logic: If the model struggles, increase the box size by reducing padding
        # and slightly perturb the positions to break out of bad latent traps
        base_padding = 20
        padding = max(2, base_padding - (repair_attempt * 5))
        
        for idx, entity in enumerate(entities):
            class_name = entity.get("class_name", "object")
            
            # Simple grid placement
            row = idx // grid_size
            col = idx % grid_size
            
            xmin = col * cell_size + padding
            ymin = row * cell_size + padding
            xmax = (col + 1) * cell_size - padding
            ymax = (row + 1) * cell_size - padding
            
            if repair_attempt > 0:
                # Add a deterministic jitter based on attempt and index to break bad structures
                jitter = (repair_attempt * 15 * (-1 if idx % 2 == 0 else 1))
                xmin = max(0, xmin + jitter)
                xmax = min(self.resolution - 1, xmax + jitter)
            
            # Ensure within bounds
            xmax = min(xmax, self.resolution - 1)
            ymax = min(ymax, self.resolution - 1)
            
            # Create a blank black image
            mask = Image.new('L', (self.resolution, self.resolution), color=0)
            draw = ImageDraw.Draw(mask)
            
            # Draw solid white bounding box (segmentation mask for ControlNet)
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
            
            # Save the bounding box back to the entity (simulating planning)
            entity["bounding_box"] = {
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
            }
            
            # If multiple of same class, combine masks (ControlNet usually takes one mask per concept or a combined segmentation map)
            # For simplicity in MVP, we just combine them into a single global segmentation mask
            if "global_mask" not in masks:
                masks["global_mask"] = Image.new('L', (self.resolution, self.resolution), color=0)
            
            global_draw = ImageDraw.Draw(masks["global_mask"])
            global_draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
            
        return masks

    def plan(self, contract: Dict[str, Any], repair_attempt: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, Dict[str, Image.Image]]:
        """
        Takes a ContractV2 dict and returns:
        (scene_plan, control_hints, generation_config, final_prompt, masks)
        """
        entities = contract.get("entities", [])
        
        # 1. Generate Masks & update bounding boxes
        masks = self._generate_synthetic_masks(entities, repair_attempt)
        
        # 2. Scene Plan (Metadata)
        scene_plan = {
            "resolution": self.resolution,
            "entities": entities # Now contains bounding_boxes
        }
        
        # 3. Final Prompt Assembly
        # The prompt is now explicitly constructed from the contract entities, not arbitrary NLP
        prompt_parts = []
        for e in entities:
            desc = f"{' '.join(e.get('attributes', []))} {e.get('class_name', '')}".strip()
            prompt_parts.append(desc)
            
        # Deduplicate and join
        unique_parts = list(set(prompt_parts))
        final_prompt = ", ".join(unique_parts) + ", highly detailed, photorealistic"
        
        # 4. Control Hints
        # We explicitly tell the generator to use the segmentation mask
        base_weight = 0.8
        # If repairing, bump up the control net weight to force the structure harder
        control_weight = min(1.0, base_weight + (repair_attempt * 0.1))
        
        control_hints = {
            "use_segmentation": len(entities) > 0,
            "use_depth": False,
            "use_canny": False,
            "control_weight": control_weight
        }
        
        # 5. Generation Config (Optimized for SDXL-Turbo)
        generation_config = {
            "width": self.resolution,
            "height": self.resolution,
            "steps": 4, # SDXL-Turbo requires 1-4 steps
            "cfg_scale": 0.0, # SDXL-Turbo requires 0.0 CFG
            "sampler": "euler_a"
        }

        return scene_plan, control_hints, generation_config, final_prompt, masks
