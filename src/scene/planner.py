from PIL import Image, ImageDraw
from copy import deepcopy
from typing import Dict, Any, Tuple, List, Optional

from src.core.stage import Stage, register_stage
from src.core.context import RunContext


class ScenePlanner:
    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        # 512x512 is much safer for 6GB VRAM than SDXL default 1024x1024
        self._clip_tokenizer = None

    def _generate_synthetic_masks(self, entities: List[Dict[str, Any]], repair_attempt: int = 0,
                                  layout_blueprint: Optional[Dict[str, Any]] = None,
                                  spatial_rels: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Image.Image]:
        """
        Translates entities into 2D geometric bounding boxes and renders segmentation masks.
        Prioritizes retrieved layout blueprints when available. If fallback is needed,
        respects spatial relationships to avoid isolating interacting objects.
        """
        masks = {}
        num_objects = len(entities)
        if num_objects == 0:
            return masks

        spatial_rels = spatial_rels or []
        bp_boxes = layout_blueprint.get("boxes", []) if layout_blueprint else []
        use_blueprint = len(bp_boxes) >= num_objects

        grid_size = int(num_objects ** 0.5) + (1 if (num_objects ** 0.5) % 1 > 0 else 0)
        cell_size = self.resolution // (grid_size if grid_size > 0 else 1)

        base_padding = 20
        padding = max(2, base_padding - (repair_attempt * 5))

        for idx, entity in enumerate(entities):
            if use_blueprint and idx < len(bp_boxes):
                box = bp_boxes[idx]
                xmin = int(box["xmin"] * self.resolution) + padding
                ymin = int(box["ymin"] * self.resolution) + padding
                xmax = int(box["xmax"] * self.resolution) - padding
                ymax = int(box["ymax"] * self.resolution) - padding
                z_index = box.get("z_index", 1)
            else:
                # Relationship-aware fallback
                # If an entity is involved in a relationship, don't isolate it in a small grid.
                # Just give it a larger box or overlap it. For simplicity, if ANY relationships exist,
                # we just use a looser layout (e.g. overlapping center boxes) instead of a strict grid.
                entity_name = str(entity.get("class_name", "")).lower()
                involved_in_rel = any(
                    r.get("subject", "").lower() == entity_name or r.get("object", "").lower() == entity_name
                    for r in spatial_rels
                )

                if involved_in_rel:
                    # Central overlapping box for related entities so they can interact
                    # We add jitter so they aren't exactly identical, but they share the main space
                    xmin = self.resolution // 4 + (idx * 20)
                    ymin = self.resolution // 4 + (idx * 20)
                    xmax = (self.resolution * 3) // 4 + (idx * 20)
                    ymax = (self.resolution * 3) // 4 + (idx * 20)
                    z_index = 1
                else:
                    # Simple grid placement fallback for independent objects
                    row = idx // grid_size
                    col = idx % grid_size
                    xmin = col * cell_size + padding
                    ymin = row * cell_size + padding
                    xmax = (col + 1) * cell_size - padding
                    ymax = (row + 1) * cell_size - padding
                    z_index = 1

            if repair_attempt > 0:
                jitter = (repair_attempt * 15 * (-1 if idx % 2 == 0 else 1))
                xmin = max(0, xmin + jitter)
                xmax = min(self.resolution - 1, xmax + jitter)

            xmax = min(max(xmin + 10, xmax), self.resolution - 1)
            ymax = min(max(ymin + 10, ymax), self.resolution - 1)

            mask = Image.new('L', (self.resolution, self.resolution), color=0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

            entity["bounding_box"] = {
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "z_index": z_index
            }

            if "global_mask" not in masks:
                # Dark sleek background for the frontend visualization
                masks["global_mask"] = Image.new('RGB', (self.resolution, self.resolution), color=(15, 15, 20))

            global_draw = ImageDraw.Draw(masks["global_mask"])

            # Distinct colors for different entities
            color = (120 + (idx * 50) % 135, 150 + (idx * 30) % 105, 255)
            # Draw semi-transparent fill and solid border
            overlay = Image.new('RGBA', masks["global_mask"].size, (0,0,0,0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([xmin, ymin, xmax, ymax], fill=color + (40,), outline=color, width=3)

            # Add text label
            label = str(entity.get("class_name", "Object")).upper()
            overlay_draw.text((xmin + 8, ymin + 8), label, fill=color)

            # Merge overlay
            masks["global_mask"] = Image.alpha_composite(masks["global_mask"].convert('RGBA'), overlay).convert('RGB')

        return masks

    def _count_clip_tokens(self, text: str) -> int:
        """CLIP-token count for the 77-token budget. Falls back to a word
        heuristic when the tokenizer isn't cached yet (offline pre-gate)."""
        try:
            if self._clip_tokenizer is None:
                from transformers import CLIPTokenizerFast
                self._clip_tokenizer = CLIPTokenizerFast.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            return len(self._clip_tokenizer(text).input_ids)
        except Exception:
            return int(len(text.split()) * 1.4) + 2

    def compose_prompt(self, entities: List[Dict[str, Any]], rag_hits: List[str],
                       prompt_edits: List[str], base_prompt: str = "") -> str:
        """
        Keep the user's prompt and corrective edits as the required core.
        Retrieved captions are evidence for inspection, never instructions
        to the image generator.
        """
        edits = [edit.strip() for edit in prompt_edits if edit.strip()]
        base = (base_prompt or "").strip()
        if not base:
            # Legacy fallback: reconstruct from entities when no base
            # prompt is available.
            entity_parts = []
            for e in entities:
                desc = f"{' '.join(e.get('attributes', []))} {e.get('class_name', '')}".strip()
                entity_parts.append(desc)
            seen = set()
            entity_parts = [p for p in entity_parts if not (p in seen or seen.add(p))]
            base = ", ".join(entity_parts)

        return ", ".join(edits + ([base] if base else []))

    def plan(self, contract: Dict[str, Any], repair_attempt: int = 0,
             prompt_edits: List[str] = None, gen_defaults: Dict[str, Any] = None
             ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, Dict[str, Image.Image]]:
        """
        Takes a ContractV2 dict and returns:
        (scene_plan, control_hints, generation_config, final_prompt, masks)
        """
        # Layout boxes are execution hints, not extracted requirements. Keep
        # the contract immutable across plans and repair attempts.
        entities = deepcopy(contract.get("entities", []))
        prompt_edits = prompt_edits or []
        gen_defaults = gen_defaults or {}

        # Layout masks exist to separate MULTIPLE discrete COUNTED objects.
        # Forcing them onto 0/1-entity (scenic) prompts carves the canvas
        # into grid cells and the model paints a collage instead of one
        # composition (verified: panoramic landscape -> 4-panel collage,
        # run 0c972039). Uncounted entities (presence-only plurals) get no
        # box: you cannot lay out "wildflowers" on a grid.
        # Rule: masks only when >= 2 counted entities.
        counted = [e for e in entities if not e.get("uncounted")]
        use_layout = len(counted) >= 2

        # 1. Generate Masks & update bounding boxes (counted entities only)
        layout_bp = contract.get("retrieved_layout") if use_layout else None
        masks = self._generate_synthetic_masks(counted, repair_attempt, layout_blueprint=layout_bp, spatial_rels=contract.get("spatial_relationships", [])) \
            if use_layout else {}

        # 2. Scene Plan (Metadata)
        scene_plan = {
            "resolution": self.resolution,
            "entities": entities  # Now contains bounding_boxes
        }

        # 3. Final Prompt Assembly (user intent + corrections)
        final_prompt = self.compose_prompt(entities, contract.get("rag_hits", []),
                                           prompt_edits,
                                           base_prompt=contract.get("prompt", ""))

        # 4. Control Hints
        base_weight = gen_defaults.get("control_weight", 0.8)
        # If repairing, bump up the control net weight to force the structure harder
        control_weight = min(1.0, base_weight + (repair_attempt * 0.1))
        configured_mode = gen_defaults.get("control_image_mode", "none")
        if configured_mode not in ("none", "canny", "box"):
            raise ValueError(f"Unsupported control_image_mode: {configured_mode}")
        control_mode = configured_mode if use_layout else "none"

        control_hints = {
            "use_segmentation": use_layout,
            "use_depth": False,
            "use_canny": control_mode == "canny",
            "control_weight": control_weight,
            # Synthetic edges erased backgrounds in the local pilot, so the
            # default is none. Explicit experiments may opt into canny/box.
            "control_image_mode": control_mode,
        }
        if gen_defaults.get("region_controller") == "fusion":
            from src.generation.regions import compile_regions
            if configured_mode != "none":
                raise ValueError("Instance fusion does not combine with synthetic ControlNet images")
            regional_plan = compile_regions(contract, self.resolution)
            control_hints["instance_regions"] = regional_plan
            scene_plan["instance_regions"] = regional_plan
            if regional_plan.get("enabled"):
                # Display exactly the regions used by the controller, rather
                # than the legacy grid computed above.
                masks = {"global_mask": Image.new("RGB", (self.resolution, self.resolution), (15, 15, 20))}
                draw = ImageDraw.Draw(masks["global_mask"])
                entity_map = {e.get("id"): e for e in entities if e.get("id")}
                for index, region in enumerate(regional_plan["regions"]):
                    box = [round(v*self.resolution) for v in region["box"]]
                    color = (120+(index*50)%135, 150+(index*30)%105, 255)
                    draw.rectangle(box, outline=color, width=3)
                    draw.text((box[0]+4, box[1]+4), region["class_name"], fill=color)
                    mask = Image.new("L", (self.resolution, self.resolution), 0)
                    ImageDraw.Draw(mask).rectangle(box, fill=255)
                    masks[f"mask_{index}"] = mask
                    entity = entity_map.get(region["entity_id"])
                    if entity is not None:
                        entity["bounding_box"] = dict(zip(("xmin", "ymin", "xmax", "ymax"), box), z_index=1)

        # 5. Generation Config (SDXL-Turbo: 1-4 steps, cfg 0.0)
        generation_config = {
            "width": self.resolution,
            "height": self.resolution,
            "steps": gen_defaults.get("steps", 4),
            "cfg_scale": gen_defaults.get("cfg_scale", 0.0),
            "sampler": "euler_a",
            "seed": gen_defaults.get("seed"),
        }
        if gen_defaults.get("region_controller") == "fusion":
            generation_config.update(region_controller="fusion",
                                     bootstrap_steps=gen_defaults.get("bootstrap_steps", 12),
                                     regional_steps=gen_defaults.get("regional_steps", 12))
            if generation_config["seed"] is not None:
                generation_config["seed"] += repair_attempt * int(gen_defaults.get("repair_seed_stride", 1))

        return scene_plan, control_hints, generation_config, final_prompt, masks


@register_stage
class PlanStage(Stage):
    name = "plan"

    def __init__(self, config):
        super().__init__(config)
        gen_cfg = config.get("generation", {})
        self.planner = ScenePlanner(resolution=gen_cfg.get("resolution", 512))
        self.gen_defaults = gen_cfg

    def run(self, ctx: RunContext) -> None:
        # SRRL memory injection on initial generation
        if ctx.attempt == 0:
            retrieved_warnings = ctx.contract.get("retrieved_warnings", [])
            for w in retrieved_warnings:
                for edit in w.get("edits", []):
                    if edit not in ctx.prompt_edits:
                        ctx.prompt_edits.append(edit)
                        print(f"[plan] SRRL Memory: Injected historical fix -> '{edit}'")

        print(f"[plan] attempt {ctx.attempt}, edits={ctx.prompt_edits}")

        scene_plan, control_hints, gen_config, final_prompt, masks = self.planner.plan(
            ctx.contract, ctx.attempt, prompt_edits=ctx.prompt_edits,
            gen_defaults=self.gen_defaults,
        )
        ctx.scene_plan = scene_plan
        ctx.control_hints = control_hints
        ctx.gen_config = gen_config
        ctx.final_prompt = final_prompt
        ctx.masks = masks
        ctx.save_artifact(f"scene_plan_{ctx.attempt}.json", scene_plan)
        ctx.save_artifact(f"generation_config_{ctx.attempt}.json", gen_config)
        ctx.save_artifact(f"final_prompt_{ctx.attempt}.txt", final_prompt, is_json=False)
        for mask_name, mask_img in masks.items():
            ctx.save_artifact(f"{mask_name}_{ctx.attempt}.png", mask_img, is_json=False)
