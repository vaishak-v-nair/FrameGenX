import re
import spacy
from typing import Dict, Any
from datetime import datetime
from src.intent.models import ContractV2, Entity, ParsedIntent, Constraint, RagInfluence, ExecutionPlan
from src.utils import free_memory

class IntentExtractor:
    def __init__(self):
        self.nlp = None
        self.embedder = None
        
        self.number_map = {
            "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, 
            "four": 4, "five": 5, "six": 6, "seven": 7
        }
        
        self.color_set = {
            "red","blue","green","yellow","black","white",
            "orange","purple","pink","brown","gray", "ancient", "mythical", "jagged"
        }
        
        self.knowledge_base = [
            "portrait photography lighting",
            "cinematic depth composition",
            "golden hour lighting",
            "symmetry in art",
            "foreground midground background layering",
            "human pose structure",
            "object counting accuracy",
        ]

    def _load_models(self):
        """Lazy load models."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _unload_models(self):
        self.nlp = None
        self.embedder = None
        free_memory()

    def _extract_count(self, text: str) -> int:
        for word, num in self.number_map.items():
            if f" {word} " in f" {text.lower()} ":
                return num
        digits = re.findall(r'\d+', text)
        return int(digits[0]) if digits else 1

    def extract(self, prompt: str) -> Dict[str, Any]:
        """
        Parses a prompt into a strict ContractV2 schema, combining the original 
        Colab NLP intelligence with the new MVP layout entity extraction.
        """
        self._load_models()
        doc = self.nlp(prompt)
        
        # --- RAG / Context Hits ---
        prompt_embedding = self.embedder.encode(prompt)
        similarities = [
            float(prompt_embedding @ self.embedder.encode(k))
            for k in self.knowledge_base
        ]
        rag_hits = [
            self.knowledge_base[i]
            for i in sorted(range(len(similarities)), key=lambda x: -similarities[x])[:3]
        ]
        
        # --- Legacy Colab Parsing ---
        subject = None
        objects = []
        colors = []

        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ["nsubj", "ROOT"]:
                subject = chunk.text
                break
        if subject is None:
            subject = list(doc.noun_chunks)[0].text if list(doc.noun_chunks) else prompt

        for token in doc:
            if token.pos_ == "NOUN":
                objects.append(token.text)
            if token.pos_ == "ADJ":
                colors.append(token.text)
                
        parsed = ParsedIntent(
            subject=subject,
            objects=list(set(objects)),
            colors=list(set(colors)),
            primary_subject=subject,
            secondary_objects=[obj for obj in set(objects) if obj != subject]
        )
        
        # --- MVP Layout Entities ---
        entities = []
        count_value = 1
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            count = self._extract_count(chunk_text)
            if count > 1:
                count_value = count # Capture main count for constraint
                
            attributes = [token.text.lower() for token in chunk if token.text.lower() in self.color_set]
            class_name = chunk.root.text.lower()
            
            if class_name in ["rain", "background", "foreground", "scene", "lighting"]:
                continue
                
            for _ in range(count):
                entities.append(Entity(class_name=class_name, attributes=attributes))

        # --- Constraints Assembly ---
        constraints = [
            Constraint(
                type="object_count",
                value=count_value,
                target=subject,
                priority="critical"
            )
        ]
        
        # --- Signals & Execution Plan ---
        execution_plan = ExecutionPlan(
            has_human="person" in prompt.lower() or "adventurer" in prompt.lower(),
            has_depth=any("depth" in r for r in rag_hits),
            risk_level="high" if len(rag_hits) > 2 else "medium"
        )
        
        signal_map = {
            "object_count": {"threshold": count_value},
            "class_uniqueness": {"threshold": len(set(objects))},
            "depth_layers": {"threshold": 2},
            "spatial_ratio": {"threshold": 0.2},
            "embedding": {"threshold": 1.0}
        }

        # Assemble contract
        contract = ContractV2(
            created_at=datetime.utcnow().isoformat(),
            prompt=prompt,
            parsed=parsed,
            constraints=constraints,
            rag_hits=rag_hits,
            rag_influence=RagInfluence(
                depth_required=execution_plan.has_depth,
                lighting_complex=any("lighting" in r for r in rag_hits)
            ),
            execution_plan=execution_plan,
            signal_map=signal_map,
            entities=entities
        )
        
        self._unload_models()
        return contract.model_dump()
