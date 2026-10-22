"""
Spatial Relationship Extraction — detects explicit spatial relations between
entities from natural language prompts using spaCy dependency parsing.

Relationships extracted:
    left_of, right_of, above, below, beside, on_top_of, in_front_of,
    behind, inside, surrounding, between, near

Each relationship is a (subject, predicate, object) triple that gets
appended to ContractV2 for downstream layout planning and verification.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# Preposition → canonical relation mapping
SPATIAL_PREPOSITIONS = {
    # left/right
    "left": "left_of",
    "right": "right_of",
    # vertical
    "above": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "over": "above",
    "atop": "on_top_of",
    # contact/support
    "on": "on_top_of",
    "upon": "on_top_of",
    # proximity
    "beside": "beside",
    "alongside": "beside",
    "next": "beside",  # "next to"
    "near": "near",
    "by": "near",
    "close": "near",
    # containment
    "in": "inside",
    "inside": "inside",
    "within": "inside",
    # spatial facing
    "behind": "behind",
    "front": "in_front_of",  # "in front of"
    "before": "in_front_of",
    "after": "behind",
    # enclosure
    "around": "surrounding",
    "surrounding": "surrounding",
    # interposition
    "between": "between",
    "among": "between",
}

# Multi-word patterns (checked first, before single-word)
MULTI_WORD_PATTERNS = {
    "in front of": "in_front_of",
    "on top of": "on_top_of",
    "next to": "beside",
    "close to": "near",
    "to the left of": "left_of",
    "to the right of": "right_of",
    "on the left of": "left_of",
    "on the right of": "right_of",
}


@dataclass
class SpatialRelationship:
    """A spatial relationship between two entities."""
    subject: str       # The entity being positioned
    predicate: str     # The canonical spatial relation
    object: str        # The reference entity
    raw_text: str = "" # The original preposition phrase
    confidence: float = 1.0

    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }


def extract_spatial_relations(
    doc,  # spaCy Doc object
    entities: List[str],
) -> List[SpatialRelationship]:
    """
    Extract spatial relationships from a parsed spaCy doc.

    Args:
        doc: A spaCy-parsed Doc object.
        entities: List of entity class_names detected by IntentExtractor.

    Returns:
        List of SpatialRelationship triples.
    """
    relationships = []
    text_lower = doc.text.lower()

    # Normalize entities for matching
    entity_set = {e.lower() for e in entities}

    # Phase 1: Multi-word pattern matching (highest priority)
    for pattern, predicate in MULTI_WORD_PATTERNS.items():
        if pattern in text_lower:
            # Find entities around the pattern
            idx = text_lower.index(pattern)
            before = text_lower[:idx].strip()
            after = text_lower[idx + len(pattern):].strip()

            subject = _find_nearest_entity(before, entity_set, direction="backward")
            obj = _find_nearest_entity(after, entity_set, direction="forward")

            if subject and obj and subject != obj:
                relationships.append(SpatialRelationship(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    raw_text=pattern,
                    confidence=0.9,
                ))

    # Phase 2: Dependency-based extraction
    for token in doc:
        if token.dep_ == "prep" and token.text.lower() in SPATIAL_PREPOSITIONS:
            predicate = SPATIAL_PREPOSITIONS[token.text.lower()]
            head = token.head

            # Find the subject (head's subtree or the head itself)
            subject_candidates = _get_noun_from_token(head, entity_set)

            # Find the object (prep's children, specifically pobj)
            object_candidates = []
            for child in token.children:
                if child.dep_ == "pobj":
                    object_candidates = _get_noun_from_token(child, entity_set)
                    break

            if subject_candidates and object_candidates:
                subj = subject_candidates[0]
                obj = object_candidates[0]
                if subj != obj:
                    # Avoid duplicates from multi-word phase
                    if not any(
                        r.subject == subj and r.predicate == predicate and r.object == obj
                        for r in relationships
                    ):
                        relationships.append(SpatialRelationship(
                            subject=subj,
                            predicate=predicate,
                            object=obj,
                            raw_text=token.text,
                            confidence=0.8,
                        ))

    return relationships


def extract_negations(doc, entities: List[str]) -> List[str]:
    """
    Extract negation constraints from a parsed spaCy doc.

    Detects patterns like "no people", "without background",
    "not including X", "except for Y".

    Args:
        doc: A spaCy-parsed Doc object.
        entities: List of detected entity class_names.

    Returns:
        List of entity class names that are explicitly negated/excluded.
    """
    negated = []
    text_lower = doc.text.lower()

    # Pattern: "no <entity>"
    for token in doc:
        if token.text.lower() in ("no", "without", "excluding"):
            for child in token.children:
                noun = child.text.lower()
                if noun in {e.lower() for e in entities} or _is_likely_noun(child):
                    negated.append(child.lemma_.lower())
            # Also check the next token directly
            if token.i + 1 < len(doc):
                next_token = doc[token.i + 1]
                if _is_likely_noun(next_token):
                    negated.append(next_token.lemma_.lower())

        # Pattern: "not ... <entity>" via dependency
        if token.dep_ == "neg":
            head = token.head
            for child in head.children:
                if child.dep_ in ("dobj", "attr", "pobj") and _is_likely_noun(child):
                    negated.append(child.lemma_.lower())

    return list(set(negated))


def _find_nearest_entity(
    text: str,
    entity_set: set,
    direction: str = "forward",
) -> Optional[str]:
    """Find the nearest entity mention in a text fragment."""
    words = text.split()
    if direction == "backward":
        words = reversed(words)

    for word in words:
        word_clean = word.strip(",.!?;:()[]{}\"'").lower()
        if word_clean in entity_set:
            return word_clean
    return None


def _get_noun_from_token(token, entity_set: set) -> List[str]:
    """Extract entity-matching nouns from a token and its subtree."""
    candidates = []

    # Check the token itself
    lemma = token.lemma_.lower()
    text = token.text.lower()
    if lemma in entity_set:
        candidates.append(lemma)
    elif text in entity_set:
        candidates.append(text)

    # Check compound children (e.g., "red car" → "car")
    for child in token.children:
        if child.dep_ in ("compound", "amod", "nsubj", "dobj", "conj", "appos"):
            child_lemma = child.lemma_.lower()
            child_text = child.text.lower()
            if child_lemma in entity_set:
                candidates.append(child_lemma)
            elif child_text in entity_set:
                candidates.append(child_text)

    return candidates


def _is_likely_noun(token) -> bool:
    """Check if a spaCy token is likely a meaningful noun."""
    return token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2
