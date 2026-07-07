import torch
from pathlib import Path
from PIL import Image
from src.utils import free_memory

class AdvancedQAVerifier:
    """
    Verification layer using GroundingDINO for exact object grounding, 
    Florence-2 for scene comprehension, and EasyOCR for text hallucination checks.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.florence_model = None
        self.florence_processor = None
        self.dino_model = None
        self.dino_processor = None
        self.ocr_reader = None

    def _load_florence(self):
        if self.florence_model is None:
            from transformers import AutoProcessor, AutoModelForCausalLM
            model_id = "microsoft/Florence-2-base"
            self.florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)

    def _load_dino(self):
        if self.dino_model is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-base"
            self.dino_processor = AutoProcessor.from_pretrained(model_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def _load_ocr(self):
        if self.ocr_reader is None:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def _unload_models(self):
        self.florence_model = None
        self.florence_processor = None
        self.dino_model = None
        self.dino_processor = None
        self.ocr_reader = None
        free_memory()

    def check_objects(self, image: Image.Image, text_queries: list) -> dict:
        self._load_dino()
        results = {}
        for query in text_queries:
            inputs = self.dino_processor(images=image, text=query, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            detected = self.dino_processor.image_processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=target_sizes
            )[0]
            
            results[query] = {
                "count": len(detected["scores"]),
                "boxes": detected["boxes"].cpu().numpy().tolist()
            }
        return results

    def check_text(self, image: Image.Image) -> list:
        self._load_ocr()
        import numpy as np
        img_np = np.array(image.convert("RGB"))
        results = self.ocr_reader.readtext(img_np)
        return [text for (bbox, text, prob) in results if prob > 0.5]

    def verify(self, image_path: Path, contract: dict) -> dict:
        image = Image.open(image_path).convert("RGB")
        parsed = contract.get("parsed", {})
        
        objects_to_check = parsed.get("objects", [])
        if parsed.get("subject"):
            objects_to_check.append(parsed.get("subject"))
            
        object_results = self.check_objects(image, objects_to_check)
        text_results = self.check_text(image)
        
        self._unload_models()
        
        decision = {
            "accepted": True, # Placeholder logic
            "object_results": object_results,
            "text_results": text_results
        }
        
        return decision
