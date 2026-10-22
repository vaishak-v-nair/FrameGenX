"""Local, abstaining color diagnostics inside independently segmented objects.

GroundingDINO supplies candidate object boxes; SAM2 supplies pixel masks.
Hue measurements do not establish calibrated full-contract correctness.
Achromatic colors, weak masks and mixed/uncertain colors remain unknown.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from src.training.common import file_sha256


SAM_MODEL = "facebook/sam2.1-hiera-tiny"
SAM_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM_WEIGHTS_SHA256 = "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"
CHROMATIC = {"red", "orange", "yellow", "green", "blue", "purple", "pink"}
ACHROMATIC = {"black", "white", "gray", "grey", "brown"}


def mask_instances(entries: list[dict], masks: list[np.ndarray]) -> tuple[list[int], list[dict]]:
    """Suppress duplicate part/whole boxes only when object pixels agree.

    Never use the requested count to choose survivors. Different classes
    converging to the same object are ambiguous rather than two instances.
    """
    if len(entries) != len(masks):
        raise ValueError("Instance records and masks differ")
    kept, ambiguous = [], []
    ranked = sorted(range(len(entries)), key=lambda i: entries[i]["mask_quality"], reverse=True)
    for index in ranked:
        a = entries[index]
        duplicate = None
        for other in kept:
            b = entries[other]
            if min(a["mask_quality"], b["mask_quality"]) < .8:
                continue
            x1, y1 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
            x2, y2 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
            area_a = (a["box"][2]-a["box"][0])*(a["box"][3]-a["box"][1])
            area_b = (b["box"][2]-b["box"][0])*(b["box"][3]-b["box"][1])
            containment = max(0, x2-x1)*max(0, y2-y1)/max(1, min(area_a, area_b))
            union = (masks[index] | masks[other]).sum()
            overlap = float((masks[index] & masks[other]).sum()/union) if union else 0
            if overlap >= .85 and containment >= .8:
                if a["class_name"] == b["class_name"]:
                    duplicate = other
                    break
                ambiguous.append({"instances": [index, other],
                                  "classes": [a["class_name"], b["class_name"]], "mask_iou": overlap})
        a["kept_for_instance_count"] = duplicate is None
        if duplicate is None:
            kept.append(index)
        else:
            a["duplicate_of"] = duplicate
    return sorted(kept), ambiguous


def masked_color_stats(image: Image.Image, mask: np.ndarray, mask_quality: float) -> dict:
    pixels = np.asarray(image.convert("RGB"))
    mask = np.asarray(mask)
    if mask.shape != pixels.shape[:2] or not np.isin(mask, [False, True]).all():
        raise ValueError("Color measurement requires an aligned binary object mask")
    if not math.isfinite(mask_quality) or not 0 <= mask_quality <= 1:
        raise ValueError("Invalid segmentation quality value")
    # Exclude boundary pixels that commonly mix foreground and background.
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    result = {"observed_color": None, "status": "unknown", "mask_quality": mask_quality,
              "mask_pixels": int(mask.sum()), "interior_pixels": int(interior.sum()),
              "method": "SAM2_mask_then_HSV_hue_diagnostic"}
    if mask_quality < .8 or interior.sum() < 128:
        return {**result, "reason": "weak or too-small segmentation mask"}
    hsv = cv2.cvtColor(pixels.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., i] for i in range(3))
    chromatic = interior & (saturation >= .35) & (value >= .15)
    coverage = float(chromatic.sum() / interior.sum())
    result["chromatic_coverage"] = coverage
    if chromatic.sum() < 128 or coverage < .35:
        return {**result, "reason": "insufficient chromatic object pixels; illumination not calibrated"}
    h = hue[chromatic]
    bins = {"red": (h < 20) | (h >= 340), "orange": (h >= 20) & (h < 45),
            "yellow": (h >= 45) & (h < 70), "green": (h >= 70) & (h < 170),
            "blue": (h >= 190) & (h < 260), "purple": (h >= 260) & (h < 295),
            "pink": (h >= 295) & (h < 340)}
    fractions = {name: float(values.mean()) for name, values in bins.items()}
    ordered = sorted(fractions, key=fractions.get, reverse=True)
    dominant, runner_up = ordered[:2]
    result["hue_fractions"] = fractions
    if fractions[dominant] < .75 or fractions[dominant] - fractions[runner_up] < .35:
        return {**result, "reason": "mixed color or hue boundary ambiguity"}
    return {**result, "status": "observed_diagnostic", "observed_color": dominant}


def summarize_class_color(expected: list[str], observations: list[dict], detected: int) -> dict:
    result = {"expected_colors": expected, "detected_instances": detected,
              "status": "unknown", "scope": "uncalibrated_instance_color_diagnostic"}
    if any(c not in CHROMATIC for c in expected):
        return {**result, "reason": "achromatic/unsupported color requires calibration"}
    if len(expected) != detected or len(observations) != detected:
        return {**result, "reason": "instance count differs; color binding cannot establish the contract"}
    if any(o.get("observed_color") is None for o in observations):
        return {**result, "reason": "one or more object colors are unknown"}
    if len(set(expected)) > 1:
        return {**result, "reason": "same-class multi-color instance identity is not calibrated"}
    observed = [o["observed_color"] for o in observations]
    return {**result, "observed_colors": observed,
            "status": "supported_diagnostic" if Counter(expected) == Counter(observed) else "contradicted_diagnostic"}


class InstanceColorVerifier:
    def __init__(self):
        self.device = "cpu"
        self.model = self.processor = None

    def _load(self):
        import torch
        from src.core.device import require_device_lease
        require_device_lease()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            from huggingface_hub import hf_hub_download
            from transformers import Sam2Processor, Sam2Model
            source = Path(hf_hub_download(SAM_MODEL, "config.json", revision=SAM_REVISION,
                                          local_files_only=True)).parent
            if file_sha256(source / "model.safetensors") != SAM_WEIGHTS_SHA256:
                raise ValueError("Local segmentation weights differ from the pinned verifier")
            self.processor = Sam2Processor.from_pretrained(source, local_files_only=True)
            self.model = Sam2Model.from_pretrained(source, local_files_only=True,
                                                  torch_dtype=torch.float32).eval()
        self.model.to(self.device)

    def verify(self, image: Image.Image, contract: dict, detected_boxes: dict, output: Path) -> dict:
        import torch
        requested = defaultdict(list)
        counted_classes = set()
        for entity in contract.get("entities") or []:
            name = str(entity.get("class_name", "")).strip().lower()
            if not name:
                continue
            if not entity.get("uncounted"):
                counted_classes.add(name)
            colors = [str(a).lower() for a in entity.get("attributes") or []
                      if str(a).lower() in CHROMATIC | ACHROMATIC]
            if len(colors) == 1:
                requested[name].append(colors[0])
        entries = []
        for name in sorted(counted_classes):
            for index, box in enumerate(detected_boxes.get(name, [])):
                coordinates = [box[k] for k in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(v) for v in coordinates):
                    raise ValueError("Nonfinite detector box")
                x1, y1, x2, y2 = coordinates
                clipped = [max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)]
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    raise ValueError("Invalid detector box")
                entries.append({"class_name": name, "detection_index": index, "box": clipped,
                                "detector_score": box.get("score")})
        if len(entries) > 12:
            raise ValueError("Segmentation diagnostic exceeds its 12-instance budget")
        output.mkdir(parents=True, exist_ok=True)
        kept, ambiguous = [], []
        if entries:
            self._load()
            try:
                inputs = self.processor(images=image, input_boxes=[[e["box"] for e in entries]], return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    predictions = self.model(**inputs, multimask_output=False)
                masks = self.processor.post_process_masks(predictions.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
                quality = predictions.iou_scores.detach().cpu().reshape(-1).tolist()
                if masks.shape[:2] != (len(entries), 1) or len(quality) != len(entries):
                    raise RuntimeError("Unexpected segmentation output shape")
                mask_arrays = [masks[i, 0].numpy().astype(bool) for i in range(len(entries))]
                for index, entry in enumerate(entries):
                    mask = mask_arrays[index]
                    stats = masked_color_stats(image, mask, quality[index])
                    filename = f"{index:02d}_{re.sub('[^A-Za-z0-9_-]', '_', entry['class_name'])}.png"
                    Image.fromarray(mask.astype(np.uint8) * 255).save(output / filename)
                    entry.update(stats, mask_file=filename, mask_sha256=file_sha256(output / filename))
                kept, ambiguous = mask_instances(entries, mask_arrays)
            finally:
                self.model.to("cpu")
                torch.cuda.empty_cache()
        unique_entries = [entries[i] for i in kept]
        refined_boxes = {name: [dict(zip(("xmin", "ymin", "xmax", "ymax"), e["box"]), score=e["detector_score"])
                               for e in unique_entries if e["class_name"] == name] for name in counted_classes}
        classes = {name: summarize_class_color(colors,
            [e for e in unique_entries if e["class_name"] == name], len(refined_boxes.get(name, [])))
            for name, colors in requested.items()}
        return {"scope": "local_uncalibrated_masked_color_diagnostics_not_full_CSR",
                "model": SAM_MODEL, "revision": SAM_REVISION, "weights_sha256": SAM_WEIGHTS_SHA256,
                "instances": entries, "classes": classes, "full_contract_CSR": None,
                "refined_boxes": refined_boxes, "ambiguous_instances": ambiguous,
                "mask_count_status": "unknown" if ambiguous or any(e["mask_quality"] < .8 for e in unique_entries) else "observed_diagnostic"}


def refine_count_decision(original: dict, colors: dict) -> dict:
    """Preserve raw detection evidence and recompute operational count checks."""
    from copy import deepcopy
    result = deepcopy(original)
    expected = result["signals"]["expected_counts"]
    result["signals"]["box_deduplicated_counts"] = result["signals"]["detected_counts"]
    refined = {name: len(colors["refined_boxes"].get(name, [])) for name in expected}
    result["signals"]["detected_counts"] = refined
    result["detected_boxes"] = colors["refined_boxes"]
    result["hard_failures"] = [{"type": "count_mismatch", "class_name": name,
        "expected": count, "detected": refined[name]} for name, count in expected.items() if refined[name] != count]
    if colors["mask_count_status"] == "unknown":
        result["hard_failures"].append({"type": "uncertain_instance_identity", "class_name": "objects",
            "expected": "unambiguous instance masks", "detected": "ambiguous or weak object masks"})
    result["accepted"] = not result["hard_failures"]
    result["count_method"] = "grounding_dino_with_SAM2_duplicate_mask_suppression"
    result["instance_count_status"] = colors["mask_count_status"]
    return result
