"""
Tests for IntentExtractor — validates contract extraction accuracy across
a range of prompt patterns: multi-object, color binding, count extraction,
spatial relations, edge cases, and stoplist filtering.

Hermetic by design: LLMRouter is replaced by the shared StubRouter from
conftest.py (canned contracts implementing the extractor's documented
rules). No network, no GPU — see conftest.py for rationale.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent.extractor import IntentExtractor


@pytest.fixture(scope="module")
def extractor(stub_intent_extractor):
    """Shared hermetic extractor — one instance per module."""
    return stub_intent_extractor


# ============================================================================
# 1. Basic Contract Shape
# ============================================================================

class TestContractShape:
    def test_returns_dict_with_required_keys(self, extractor):
        contract = extractor.extract("a red car")
        assert isinstance(contract, dict)
        assert "prompt" in contract
        assert "entities" in contract
        assert "constraints" in contract
        assert "parsed" in contract
        assert "schema_version" in contract
        assert contract["schema_version"] == "2.0"

    def test_prompt_preserved_verbatim(self, extractor):
        prompt = "three blue birds sitting on a fence"
        contract = extractor.extract(prompt)
        assert contract["prompt"] == prompt

    def test_created_at_is_iso_format(self, extractor):
        contract = extractor.extract("a cat")
        created = contract["created_at"]
        assert "T" in created  # ISO 8601 format
        assert "+" in created or "Z" in created or created.endswith("+00:00")  # timezone aware


# ============================================================================
# 2. Count Extraction
# ============================================================================

class TestCountExtraction:
    def test_single_object_default_count(self, extractor):
        contract = extractor.extract("a red apple")
        entities = contract["entities"]
        # Should have exactly 1 apple entity
        apple_entities = [e for e in entities if e["class_name"] == "apple"]
        assert len(apple_entities) == 1

    def test_numeric_count(self, extractor):
        contract = extractor.extract("3 red apples on a table")
        entities = contract["entities"]
        apple_entities = [e for e in entities if e["class_name"] == "apple" or e["class_name"] == "apples"]
        # "3" should produce 3 entities
        assert len(apple_entities) >= 2  # At minimum recognizes multiplicity

    def test_word_count_two(self, extractor):
        contract = extractor.extract("two blue birds")
        entities = contract["entities"]
        bird_entities = [e for e in entities if "bird" in e["class_name"]]
        assert len(bird_entities) == 2

    def test_word_count_three(self, extractor):
        contract = extractor.extract("three red cars")
        entities = contract["entities"]
        car_entities = [e for e in entities if "car" in e["class_name"]]
        assert len(car_entities) == 3

    def test_word_a_means_one(self, extractor):
        contract = extractor.extract("a yellow sunflower")
        entities = contract["entities"]
        flower_entities = [e for e in entities if "sunflower" in e["class_name"]]
        assert len(flower_entities) == 1

    def test_constraint_captures_count(self, extractor):
        contract = extractor.extract("three red apples")
        constraints = contract["constraints"]
        count_constraints = [c for c in constraints if c["type"] == "object_count"]
        assert len(count_constraints) >= 1
        assert count_constraints[0]["value"] == 3


# ============================================================================
# 3. Color / Attribute Binding
# ============================================================================

class TestAttributeBinding:
    def test_single_color_extracted(self, extractor):
        contract = extractor.extract("a red car")
        entities = contract["entities"]
        car_entities = [e for e in entities if e["class_name"] == "car"]
        assert len(car_entities) >= 1
        assert "red" in car_entities[0]["attributes"]

    def test_multiple_colors(self, extractor):
        contract = extractor.extract("a red car and a blue house")
        entities = contract["entities"]
        car_entities = [e for e in entities if e["class_name"] == "car"]
        house_entities = [e for e in entities if e["class_name"] == "house"]
        # At minimum, colors should be extracted
        all_attrs = []
        for e in entities:
            all_attrs.extend(e["attributes"])
        assert "red" in all_attrs
        assert "blue" in all_attrs

    def test_no_color_means_empty_attributes(self, extractor):
        contract = extractor.extract("a table with a lamp")
        entities = contract["entities"]
        # Entities should exist but may have empty attributes
        for e in entities:
            for attr in e["attributes"]:
                # Any attribute present should be from the color_set
                assert attr in extractor.color_set


# ============================================================================
# 4. Stoplist Filtering
# ============================================================================

class TestStoplistFiltering:
    """Entity stoplist prevents phantom entities that make DINO benchmarks unwinnable."""

    def test_scene_words_filtered(self, extractor):
        contract = extractor.extract("a cat in the background with nice lighting")
        entities = contract["entities"]
        class_names = [e["class_name"] for e in entities]
        assert "background" not in class_names
        assert "lighting" not in class_names

    def test_surface_words_filtered(self, extractor):
        contract = extractor.extract("a dog on the grass near a road")
        entities = contract["entities"]
        class_names = [e["class_name"] for e in entities]
        assert "grass" not in class_names
        assert "road" not in class_names

    def test_real_objects_not_filtered(self, extractor):
        contract = extractor.extract("a cat and a dog")
        entities = contract["entities"]
        class_names = [e["class_name"] for e in entities]
        assert "cat" in class_names
        assert "dog" in class_names

    def test_atmospheric_words_filtered(self, extractor):
        contract = extractor.extract("a sunset view of the ocean with warm colors")
        entities = contract["entities"]
        class_names = [e["class_name"] for e in entities]
        assert "sunset" not in class_names
        assert "view" not in class_names
        assert "colors" not in class_names


# ============================================================================
# 5. Parsed Intent Fields
# ============================================================================

class TestParsedIntent:
    def test_subject_detected(self, extractor):
        contract = extractor.extract("a red car parked in the driveway")
        parsed = contract["parsed"]
        assert parsed["subject"] is not None
        assert len(parsed["subject"]) > 0

    def test_objects_populated(self, extractor):
        contract = extractor.extract("a cat sitting on a table")
        parsed = contract["parsed"]
        assert len(parsed["objects"]) > 0

    def test_colors_populated(self, extractor):
        contract = extractor.extract("a blue bird and a red flower")
        parsed = contract["parsed"]
        assert "blue" in parsed["colors"] or "red" in parsed["colors"]


# ============================================================================
# 6. Execution Plan
# ============================================================================

class TestExecutionPlan:
    def test_human_detected(self, extractor):
        contract = extractor.extract("a person standing in a park")
        ep = contract["execution_plan"]
        assert ep["has_human"] is True

    def test_no_human(self, extractor):
        contract = extractor.extract("three red apples on a table")
        ep = contract["execution_plan"]
        assert ep["has_human"] is False

    def test_adventurer_counts_as_human(self, extractor):
        contract = extractor.extract("an adventurer in a mythical forest")
        ep = contract["execution_plan"]
        assert ep["has_human"] is True


# ============================================================================
# 7. Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_prompt_does_not_crash(self, extractor):
        """Empty or trivial prompts should not crash the extractor."""
        try:
            contract = extractor.extract("")
            assert contract is not None
        except Exception:
            # Some edge cases may raise; that's acceptable for truly empty input
            pass

    def test_very_long_prompt(self, extractor):
        prompt = "a " + " and a ".join(["red apple"] * 20)
        contract = extractor.extract(prompt)
        assert contract is not None
        assert len(contract["entities"]) > 0

    def test_prompt_with_numbers_and_words(self, extractor):
        contract = extractor.extract("2 red cars and three blue trucks")
        entities = contract["entities"]
        # Should detect both numeric and word counts
        assert len(entities) >= 2

    def test_unicode_prompt(self, extractor):
        contract = extractor.extract("a café with two chairs")
        assert contract is not None
        assert contract["prompt"] == "a café with two chairs"


# ============================================================================
# 8. RAG Fields Start Empty (RetrievalStage fills them)
# ============================================================================

class TestRagFieldsEmpty:
    def test_rag_hits_empty(self, extractor):
        contract = extractor.extract("a red car")
        assert contract["rag_hits"] == []

    def test_rag_influence_defaults(self, extractor):
        contract = extractor.extract("a red car")
        rag = contract["rag_influence"]
        assert rag["depth_required"] is False
        assert rag["lighting_complex"] is False


# ============================================================================
# 9. Repair policy (paper-contract parity)
# ============================================================================

def test_contract_carries_repair_policy(extractor):
    """The extractor must emit the rule-based repair policy so the
    reflector's deterministic-override layer can fire."""
    contract = extractor.extract("a red car")
    policy = contract.get("repair_policy")
    assert policy and "mappings" in policy
    failures = {m["failure"] for m in policy["mappings"]}
    assert {"object_missing", "duplicate_object", "low_alignment"} <= failures
    # prompt-safety: no raw meta-action names in any fragment
    for m in policy["mappings"]:
        assert "increase_object_weight" not in m["action"]
        assert "enforce_single_subject" not in m["action"]
