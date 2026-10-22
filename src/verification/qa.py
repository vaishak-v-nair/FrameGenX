import torch
import re
from PIL import Image
from pathlib import Path
from src.utils import free_memory
from typing import Dict, Any
from src.core.device import require_device_lease


class QAVerifier:
    """
    GroundingDINO open-vocabulary detection: verifies the count of objects
    specified in the ContractV2 schema.

    VRAM policy (design doc): load fp16, infer on GPU, then move to CPU
    residence between uses — never keep resident on GPU alongside SDXL.
    """

    def __init__(self):
        self.device = "cpu"
        # fp32, not fp16: grounding-dino's deformable attention mixes fp32
        # tensors internally and grid_sample crashes under a Half model
        # ("expected scalar type Half but found Float"). Model is tiny
        # (~700MB fp32) and CPU-resident between uses, so VRAM stays safe.
        self.dtype = torch.float32
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = None
        self.model = None

    def _load_models(self):
        require_device_lease()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, torch_dtype=self.dtype
            )
        self.model.to(self.device)

    def _to_cpu(self):
        # CPU residence, not destruction: avoids 30-60s disk reloads per
        # attempt while keeping VRAM free for the SDXL pipe.
        if self.model is not None:
            self.model.to("cpu")
        free_memory()

    def _post_process(self, outputs, inputs, image):
        """transformers 5.x changed the grounded-detection post-process
        signature (threshold/text_labels vs input_ids/box_threshold).
        Try new API first, fall back to the v4.x call."""
        try:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=0.3,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )[0]

    @staticmethod
    def _label_matches_class(label: str, class_name: str) -> bool:
        """Match a detected noun phrase without accepting substrings of words."""
        phrase = str(class_name).strip().lower()
        if not phrase:
            return False
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", str(label).lower()) is not None

    @staticmethod
    def _dedupe_boxes(boxes: list, iou_threshold: float = 0.90) -> list:
        """Suppress only near-identical detections of the same queried class.

        The high threshold avoids merging nearby real objects. It is a
        conservative duplicate fix, not a calibrated counting algorithm.
        """
        ranked = sorted(boxes, key=lambda box: box.get("score", 0), reverse=True)
        kept = []
        for box in ranked:
            if all(QAVerifier._box_iou(box, other) < iou_threshold for other in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _box_iou(a: dict, b: dict) -> float:
        x1, y1 = max(a["xmin"], b["xmin"]), max(a["ymin"], b["ymin"])
        x2, y2 = min(a["xmax"], b["xmax"]), min(a["ymax"], b["ymax"])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a["xmax"] - a["xmin"]) * max(0.0, a["ymax"] - a["ymin"])
        area_b = max(0.0, b["xmax"] - b["xmin"]) * max(0.0, b["ymax"] - b["ymin"])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def verify(self, image_path: Path, contract: Dict[str, Any]) -> dict:
        image = Image.open(image_path).convert("RGB")

        # Parse entities from ContractV2. Uncounted entities (unspecified
        # plurals like "wildflowers") are presence-checked by the VQA
        # existence question, never by DINO counting — counting a meadow
        # produced unwinnable "expected 1, detected 10" failures.
        entities = contract.get("entities", [])
        expected_counts = {}
        for entity in entities:
            if entity.get("uncounted"):
                continue
            class_name = entity.get("class_name", "").lower()
            if class_name:
                expected_counts[class_name] = expected_counts.get(class_name, 0) + 1

        # If no explicit objects requested, automatically pass (or skip object count check)
        if not expected_counts:
            return {"accepted": True, "hard_failures": [], "signals": {}}

        self._load_models()

        # GroundingDINO expects text queries formatted like "cat . dog ."
        text_queries = [f"{cls_name} ." for cls_name in expected_counts.keys()]
        text_query = " ".join(text_queries)

        assert self.processor is not None, "QA processor failed to load"
        assert self.model is not None, "QA model failed to load"

        from src.utils import cast_inputs
        raw = self.processor(images=image, text=text_query, return_tensors="pt")
        inputs = type(raw)(cast_inputs(dict(raw), self.device, self.dtype))
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self._post_process(outputs, inputs, image)

        # 5.x returns "text_labels"; 4.x returned "labels"
        labels = results.get("text_labels") if isinstance(results, dict) else None
        if labels is None:
            labels = results["labels"]

        raw_detected_counts = {cls_name: 0 for cls_name in expected_counts.keys()}
        unboxed_counts = {cls_name: 0 for cls_name in expected_counts.keys()}
        detected_boxes = {cls_name: [] for cls_name in expected_counts.keys()}
        boxes = results.get("boxes", [])
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()

        for i, (label, score) in enumerate(zip(labels, results["scores"])):
            for cls_name in expected_counts.keys():
                if self._label_matches_class(label, cls_name):
                    raw_detected_counts[cls_name] += 1
                    # Collect bounding box for spatial verification
                    if i < len(boxes):
                        box = boxes[i]
                        detected_boxes[cls_name].append({
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                            "score": float(score) if hasattr(score, "item") else float(score),
                        })
                    else:
                        unboxed_counts[cls_name] += 1
                    break  # Assign to the first matching class

        detected_boxes = {class_name: self._dedupe_boxes(class_boxes)
                          for class_name, class_boxes in detected_boxes.items()}
        detected_counts = {
            class_name: len(detected_boxes[class_name]) + unboxed_counts[class_name]
            for class_name in expected_counts
        }

        hard_failures = []
        for cls_name, expected in expected_counts.items():
            detected = detected_counts.get(cls_name, 0)
            if detected != expected:
                hard_failures.append(
                    {
                        "type": "count_mismatch",
                        "class_name": cls_name,
                        "expected": expected,
                        "detected": detected,
                    }
                )

        decision = {
            "accepted": len(hard_failures) == 0,
            "hard_failures": hard_failures,
            "detected_boxes": detected_boxes,
            "signals": {
                "detected_counts": detected_counts,
                "raw_detected_counts": raw_detected_counts,
                "expected_counts": expected_counts,
            },
        }

        self._to_cpu()
        return decision
