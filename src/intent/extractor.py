import os
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any

from src.intent.models import (
    ContractV2,
    Entity,
    ParsedIntent,
    Constraint,
    RagInfluence,
    ExecutionPlan,
)
from src.core.stage import Stage, register_stage
from src.core.context import RunContext, load_config
from src.core.llm_router import LLMRouter

# Backstop stoplist (config-overridable via intent.uncountable_stoplist):
# atmospheric/terrain concepts that GroundingDINO cannot ground as countable
# boxes. Treating them as entities produced unwinnable constraints ("mist x1")
# and nonsense repairs ("exactly 1 mist, each clearly separated") — verified
# in run 0c972039 2026-08-24. Deterministic filter per AGENTS.md rule 5.
DEFAULT_UNCOUNTABLE_STOPLIST = [
    "mist", "fog", "haze", "light", "sunlight", "glow", "shadow", "sky",
    "sunset", "sunrise", "atmosphere", "background", "scenery", "landscape",
    "view", "valley", "summit", "depth", "bokeh", "gradient", "texture",
    "pattern", "tone", "mood", "weather", "cloud", "clouds", "water",
    "reflection", "silhouette", "panorama", "horizon",
]


# Number words that make a plural noun explicitly counted ("three peaks").
_NUMBER_WORDS = {
    "a", "an", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "dozen", "couple", "pair",
}


def _is_uncounted_plural(prompt: str, class_name: str) -> bool:
    """Deterministic plural rule (backstop to LLM rule 6): if the prompt
    references the class in plural form WITHOUT a preceding number word,
    the entity is presence-only. "wildflowers in bloom" -> uncounted;
    "three peaks" -> counted; "a cat" -> counted. The LLM repeatedly
    ignored this rule (run 3500c905: wildflowers -> wildflower x1 -> DINO
    found 10 -> unwinnable), so it is enforced in code per AGENTS.md rule 5.
    """
    name = class_name.lower().strip()
    if not name:
        return False
    forms = {name + "s", name + "es"}
    if name.endswith("y"):
        forms.add(name[:-1] + "ies")
    low = prompt.lower()
    for form in forms:
        for m in re.finditer(rf"\b{re.escape(form)}\b", low):
            prev_tokens = low[:m.start()].split()[-2:]
            if any(t in _NUMBER_WORDS or t.isdigit() for t in prev_tokens):
                return False  # explicitly counted plural
            return True  # bare plural
    return False  # singular or absent -> counted as usual


def _preserve_support_relation(prompt: str, relation: Dict[str, Any]) -> bool:
    """Recognize an explicit support verb when an LLM weakens it to above.

    Only correct an already extracted subject/object pair. This avoids
    inventing object bindings from a broad keyword match.
    """
    if str(relation.get("predicate", "")).lower() != "above":
        return False
    subject = str(relation.get("subject", "")).strip()
    obj = str(relation.get("object", "")).strip()
    if not subject or not obj:
        return False
    pattern = (rf"\b{re.escape(subject)}\b(?P<before>[^.!?;]{{0,55}}?)"
               rf"\b(?:sitting|standing|resting|perched|lying)\s+on"
               rf"(?:\s+top\s+of)?\b(?P<after>[^.!?;]{{0,55}}?)"
               rf"\b{re.escape(obj)}\b")
    match = re.search(pattern, prompt, flags=re.IGNORECASE)
    if not match:
        return False
    return not re.search(r"\b(?:not|never|without)\b",
                         match.group("before") + match.group("after"),
                         flags=re.IGNORECASE)


# Rule-based repair policy (ported from the FrameGenX paper contract,
# FrameGenX/output/contract_v2.json). The reflector consults these mappings
# when synthesizing corrective edits. Actions are PROMPT-SAFE fragments:
# they are spliced into the drawing prompt, so meta-language
# ("increase_object_weight") is forbidden here by construction.
DEFAULT_REPAIR_POLICY = {
    "mappings": [
        {"failure": "object_missing", "action": "prominently featured"},
        {"failure": "duplicate_object", "action": ""},
        {"failure": "text_missing", "action": "with clearly legible text"},
        {"failure": "low_alignment", "action": "sharp focus, highly detailed"},
    ]
}


class IntentExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or load_config()
        intent_cfg = (cfg or {}).get("intent", {}) or {}
        self.uncountable_stoplist = set(
            intent_cfg.get("uncountable_stoplist", DEFAULT_UNCOUNTABLE_STOPLIST)
        )
        self.router = LLMRouter()

    def extract(self, prompt: str) -> Dict[str, Any]:
        system_prompt = '''You are a strict JSON API for parsing image generation prompts into a structured schema.
Output ONLY valid JSON matching the following structure. Do NOT wrap in markdown blocks.
{
  "parsed": {
    "subject": "string (the main subject)",
    "objects": ["list of strings (all nouns)"],
    "colors": ["list of strings (all colors)"],
    "primary_subject": "string",
    "secondary_objects": ["list of strings"]
  },
  "entities": [
    {
      "class_name": "string (noun, singular, e.g. 'car')",
      "attributes": ["list of colors/adjectives"]
    }
  ],
  "constraints": [
    {
      "type": "object_count",
      "value": 1,
      "target": "string (the main subject)",
      "priority": "critical"
    }
  ],
  "spatial_relationships": [
    {
      "subject": "string",
      "predicate": "string (e.g. left_of, right_of, above, below, inside)",
      "object": "string"
    }
  ],
  "negations": ["list of strings (things that should NOT be in the image)"],
  "execution_plan": {
    "has_human": false,
    "has_depth": false,
    "risk_level": "medium"
  },
  "signal_map": {
    "object_count": {"threshold": 1},
    "class_uniqueness": {"threshold": 1},
    "depth_layers": {"threshold": 2},
    "spatial_ratio": {"threshold": 0.2},
    "embedding": {"threshold": 1.0}
  }
}

Rules:
1. "entities" must contain ONLY physically countable, visually detectable
   objects (things an object detector could draw a box around: "apple",
   "car", "cat", "chair", "person"). One entity per instance: "two cars" ->
   two entities with class_name "car".
2. NEVER make entities out of atmosphere, lighting, terrain or abstraction:
   mist, fog, light, sunlight, glow, shadow, sky, sunset, sunrise, valley,
   summit, background, scenery, atmosphere, weather, reflection. Those are
   scene descriptors — capture them in "parsed".objects instead. A prompt
   like "panoramic alpine peaks at sunrise with mist in the valleys" has
   ZERO countable entities unless a discrete object is named.
3. Skip scene descriptors like "background", "photo", "image", "lighting" in
   "entities".
4. "spatial_relationships" may only reference entities that exist in
   "entities" (both subject and object must be countable objects).
5. Landscape-scale nouns (mountain(s), peak(s), hill(s), tree(s), cloud(s),
   wave(s), building(s) in a skyline, etc.) are SCENERY: do NOT create
   entities or count constraints for them unless the prompt states an exact
   small count ("three peaks" -> 3 peak entities; "jagged alpine peaks"
   with no number -> zero peak entities). Object detectors cannot count
   scenery reliably, so such constraints are unwinnable.
6. Plural nouns with NO explicit count ("wildflowers in bloom", "birds in
   the sky") -> emit exactly ONE entity for the class and set
   "uncounted": true on it. They are presence-checked, never counted.
   Singular with "a/an" or an explicit number -> counted entities as usual
   ("a cat" -> 1 cat entity, uncounted omitted/false; "three cars" -> 3
   car entities).
7. Preserve support relations precisely. "A cat sitting on a chair" has
   predicate "on", not "above"; above only establishes vertical ordering.
'''
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this prompt: {prompt}"}
        ]
        
        data = self.router.generate_json(messages=messages, temperature=0.0)

        # Deterministic backstop: drop uncountable/atmospheric entities the
        # LLM emitted anyway, and any relationship referencing them. Keeps
        # verification (GroundingDINO counts) winnable for scenic prompts.
        raw_entities = data.get("entities", [])
        dropped = [e.get("class_name") for e in raw_entities
                   if str(e.get("class_name", "")).lower() in self.uncountable_stoplist]
        if dropped:
            print(f"[intent] dropped uncountable entities: {sorted(set(dropped))}")
        entities = [e for e in raw_entities
                    if str(e.get("class_name", "")).lower() not in self.uncountable_stoplist]
        # Deterministic plural backstop (see _is_uncounted_plural): the LLM
        # cannot be trusted to apply rule 6 consistently.
        for e in entities:
            if _is_uncounted_plural(prompt, str(e.get("class_name", ""))):
                e["uncounted"] = True
        spatial_rels = [
            dict(rel) for rel in data.get("spatial_relationships", [])
            if str(rel.get("subject", "")).lower() not in self.uncountable_stoplist
            and str(rel.get("object", "")).lower() not in self.uncountable_stoplist
        ]
        # Relationships must reference counted entities only: uncounted
        # entities get no DINO boxes, so spatial geometry on them can only
        # produce phantom violations.
        uncounted_names = {str(e.get("class_name", "")).lower()
                           for e in entities if e.get("uncounted")}
        if uncounted_names:
            spatial_rels = [
                rel for rel in spatial_rels
                if str(rel.get("subject", "")).lower() not in uncounted_names
                and str(rel.get("object", "")).lower() not in uncounted_names
            ]

        extraction_corrections = []
        for rel in spatial_rels:
            if _preserve_support_relation(prompt, rel):
                extraction_corrections.append({
                    "subject": rel["subject"], "object": rel["object"],
                    "from": rel["predicate"], "to": "on",
                    "reason": "explicit support verb in prompt",
                })
                rel["predicate"] = "on"

        raw_constraints = []
        for item in data.get("constraints", []):
            constraint = dict(item)
            value = constraint.get("value")
            if constraint.get("type") == "spatial_relation" and isinstance(value, dict):
                for correction in extraction_corrections:
                    if (str(value.get("subject", "")).lower() ==
                            str(correction["subject"]).lower()
                            and str(value.get("object", "")).lower() ==
                            str(correction["object"]).lower()
                            and str(value.get("predicate", "")).lower() == "above"):
                        constraint["value"] = {**value, "predicate": "on"}
                        break
            raw_constraints.append(constraint)
        constraints = [Constraint(**c) for c in raw_constraints]

        for rel in spatial_rels:
            constraints.append(Constraint(
                type="spatial_relation",
                value=rel,
                target=f"{rel.get('subject', '')}->{rel.get('object', '')}",
                priority="high",
            ))
            
        contract = ContractV2(
            created_at=datetime.now(timezone.utc).isoformat(),
            prompt=prompt,
            parsed=ParsedIntent(**data.get("parsed", {})),
            constraints=constraints,
            rag_hits=[],
            rag_influence=RagInfluence(),
            execution_plan=ExecutionPlan(**data.get("execution_plan", {})),
            signal_map=data.get("signal_map", {}),
            entities=[Entity(**e) for e in entities],
        )
        
        result = contract.model_dump()
        result["spatial_relationships"] = spatial_rels
        result["negations"] = data.get("negations", [])
        # Rule-based repair policy (see DEFAULT_REPAIR_POLICY): consumed by
        # src/reflection/reflector.py to synthesize deterministic edits.
        result["repair_policy"] = DEFAULT_REPAIR_POLICY
        if extraction_corrections:
            result["extraction_corrections"] = extraction_corrections

        return result


@register_stage
class IntentStage(Stage):
    name = "intent"

    def __init__(self, config):
        super().__init__(config)
        self.extractor = IntentExtractor(config)

    def run(self, ctx: RunContext) -> None:
        print("[intent] extracting contract via Groq API...")
        ctx.contract = self.extractor.extract(ctx.prompt)
        ctx.save_artifact("contract_v2.json", ctx.contract)
