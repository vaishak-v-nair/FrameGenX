import torch
from PIL import Image
from pathlib import Path
from src.utils import free_memory
from typing import Dict, Any

class QAVerifier:
    """
    Phase 6 QA Verifier.
    Uses GroundingDINO for open-vocabulary zero-shot object detection to explicitly
    verify the count of objects specified in the ContractV2 schema.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = None
        self.model = None
        
    def _load_models(self):
        if self.model is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def _unload_models(self):
        # Memory optimization: Do not destroy the model to avoid slow disk reloading.
        # self.model = None
        # self.processor = None
        free_memory()

    def verify(self, image_path: Path, contract: Dict[str, Any]) -> dict:
        self._load_models()
        image = Image.open(image_path).convert("RGB")
        
        # Parse entities from ContractV2
        entities = contract.get("entities", [])
        expected_counts = {}
        for entity in entities:
            class_name = entity.get("class_name", "").lower()
            if class_name:
                expected_counts[class_name] = expected_counts.get(class_name, 0) + 1
                
        # If no explicit objects requested, automatically pass (or skip object count check)
        if not expected_counts:
            self._unload_models()
            return {"accepted": True, "hard_failures": [], "signals": {}}
            
        # GroundingDINO expects text queries formatted like "cat . dog ."
        text_queries = [f"{cls_name} ." for cls_name in expected_counts.keys()]
        text_query = " ".join(text_queries)
        
        inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )[0]
        
        detected_counts = {cls_name: 0 for cls_name in expected_counts.keys()}
        
        for label, score in zip(results["labels"], results["scores"]):
            # Grounding DINO labels can sometimes contain extra text, check for substring matching
            for cls_name in expected_counts.keys():
                if cls_name in label.lower():
                    detected_counts[cls_name] += 1
                    break # Assign to the first matching class
                    
        hard_failures = []
        for cls_name, expected in expected_counts.items():
            detected = detected_counts.get(cls_name, 0)
            if detected != expected:
                hard_failures.append({
                    "type": "count_mismatch",
                    "class_name": cls_name,
                    "expected": expected,
                    "detected": detected
                })
                
        decision = {
            "accepted": len(hard_failures) == 0,
            "hard_failures": hard_failures,
            "signals": {
                "detected_counts": detected_counts,
                "expected_counts": expected_counts
            }
        }
        
        self._unload_models()
        return decision
