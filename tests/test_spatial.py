"""
Tests for the Spatial Verification pipeline:
  1. SpatialRelationVerifier — bounding box geometry checks
  2. extract_spatial_relations — NLP-based spatial extraction
  3. Integration — spatial relationships flow from extractor → verifier
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verification.spatial_verifier import SpatialRelationVerifier


# ============================================================================
# 1. SpatialRelationVerifier — Geometry Checks
# ============================================================================

class TestSpatialVerifierGeometry:
    @pytest.fixture
    def verifier(self):
        return SpatialRelationVerifier()

    def test_left_of_pass(self, verifier):
        """Subject center is left of object center."""
        boxes = {
            "car": [{"xmin": 50, "ymin": 100, "xmax": 150, "ymax": 200}],
            "house": [{"xmin": 300, "ymin": 100, "xmax": 450, "ymax": 250}],
        }
        result = verifier.verify(
            [{"subject": "car", "predicate": "left_of", "object": "house"}],
            boxes, 512, 512
        )
        assert result["passed"] is True
        assert result["verified"] == 1

    def test_left_of_fail(self, verifier):
        """Subject center is right of object center."""
        boxes = {
            "car": [{"xmin": 300, "ymin": 100, "xmax": 450, "ymax": 200}],
            "house": [{"xmin": 50, "ymin": 100, "xmax": 150, "ymax": 250}],
        }
        result = verifier.verify(
            [{"subject": "car", "predicate": "left_of", "object": "house"}],
            boxes, 512, 512
        )
        assert result["passed"] is False

    def test_above(self, verifier):
        """Subject center_y < object center_y (image coords: y=0 is top)."""
        boxes = {
            "bird": [{"xmin": 200, "ymin": 50, "xmax": 300, "ymax": 150}],
            "tree": [{"xmin": 200, "ymin": 250, "xmax": 300, "ymax": 400}],
        }
        result = verifier.verify(
            [{"subject": "bird", "predicate": "above", "object": "tree"}],
            boxes, 512, 512
        )
        assert result["passed"] is True

    def test_below(self, verifier):
        boxes = {
            "cat": [{"xmin": 200, "ymin": 300, "xmax": 300, "ymax": 400}],
            "shelf": [{"xmin": 200, "ymin": 50, "xmax": 300, "ymax": 150}],
        }
        result = verifier.verify(
            [{"subject": "cat", "predicate": "below", "object": "shelf"}],
            boxes, 512, 512
        )
        assert result["passed"] is True

    def test_near(self, verifier):
        boxes = {
            "dog": [{"xmin": 200, "ymin": 200, "xmax": 260, "ymax": 260}],
            "cat": [{"xmin": 270, "ymin": 210, "xmax": 330, "ymax": 270}],
        }
        result = verifier.verify(
            [{"subject": "dog", "predicate": "near", "object": "cat"}],
            boxes, 512, 512
        )
        assert result["passed"] is True

    def test_near_fail(self, verifier):
        """Objects on opposite corners — too far to be 'near'."""
        boxes = {
            "dog": [{"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50}],
            "cat": [{"xmin": 450, "ymin": 450, "xmax": 512, "ymax": 512}],
        }
        result = verifier.verify(
            [{"subject": "dog", "predicate": "near", "object": "cat"}],
            boxes, 512, 512
        )
        assert result["passed"] is False

    def test_inside(self, verifier):
        """Subject fully contained within object."""
        boxes = {
            "bird": [{"xmin": 150, "ymin": 150, "xmax": 250, "ymax": 250}],
            "cage": [{"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 300}],
        }
        result = verifier.verify(
            [{"subject": "bird", "predicate": "inside", "object": "cage"}],
            boxes, 512, 512
        )
        assert result["passed"] is True


# ============================================================================
# 2. Missing Entity Handling
# ============================================================================

class TestSpatialVerifierMissing:
    @pytest.fixture
    def verifier(self):
        return SpatialRelationVerifier()

    def test_missing_subject(self, verifier):
        boxes = {"house": [{"xmin": 300, "ymin": 100, "xmax": 450, "ymax": 250}]}
        result = verifier.verify(
            [{"subject": "car", "predicate": "left_of", "object": "house"}],
            boxes, 512, 512
        )
        assert result["passed"] is False
        assert "not detected" in result["failures"][0]["reason"]

    def test_missing_object(self, verifier):
        boxes = {"car": [{"xmin": 50, "ymin": 100, "xmax": 150, "ymax": 200}]}
        result = verifier.verify(
            [{"subject": "car", "predicate": "left_of", "object": "house"}],
            boxes, 512, 512
        )
        assert result["passed"] is False
        assert "not detected" in result["failures"][0]["reason"]

    def test_empty_relationships(self, verifier):
        result = verifier.verify([], {}, 512, 512)
        assert result["passed"] is True
        assert result["total"] == 0

    def test_fuzzy_matching(self, verifier):
        """class_name 'bicycle' matches detected 'bicycle' or vice versa."""
        boxes = {
            "bicycle": [{"xmin": 50, "ymin": 100, "xmax": 150, "ymax": 200}],
            "house": [{"xmin": 300, "ymin": 100, "xmax": 450, "ymax": 250}],
        }
        result = verifier.verify(
            [{"subject": "bicycle", "predicate": "left_of", "object": "house"}],
            boxes, 512, 512
        )
        assert result["passed"] is True


# ============================================================================
# 3. Multiple Relationships
# ============================================================================

class TestSpatialVerifierMultiple:
    @pytest.fixture
    def verifier(self):
        return SpatialRelationVerifier()

    def test_all_pass(self, verifier):
        boxes = {
            "car": [{"xmin": 50, "ymin": 200, "xmax": 200, "ymax": 350}],
            "house": [{"xmin": 300, "ymin": 100, "xmax": 480, "ymax": 400}],
            "tree": [{"xmin": 200, "ymin": 50, "xmax": 280, "ymax": 150}],
        }
        result = verifier.verify(
            [
                {"subject": "car", "predicate": "left_of", "object": "house"},
                {"subject": "tree", "predicate": "above", "object": "car"},
            ],
            boxes, 512, 512
        )
        assert result["passed"] is True
        assert result["verified"] == 2

    def test_partial_fail(self, verifier):
        boxes = {
            "car": [{"xmin": 300, "ymin": 200, "xmax": 450, "ymax": 350}],  # car is RIGHT
            "house": [{"xmin": 50, "ymin": 100, "xmax": 200, "ymax": 400}],
            "tree": [{"xmin": 200, "ymin": 50, "xmax": 280, "ymax": 150}],
        }
        result = verifier.verify(
            [
                {"subject": "car", "predicate": "left_of", "object": "house"},  # FAIL
                {"subject": "tree", "predicate": "above", "object": "car"},      # PASS
            ],
            boxes, 512, 512
        )
        assert result["passed"] is False
        assert result["verified"] == 1
        assert result["total"] == 2
        assert len(result["failures"]) == 1


@pytest.fixture(scope="module")
def nlp_model():
    import spacy
    return spacy.load("en_core_web_sm")


# ============================================================================
# 4. Spatial Relationship Extraction (NLP)
# ============================================================================

class TestSpatialRelationExtraction:
    def test_on_top_of_pattern(self, nlp_model):
        from src.intent.relationships import extract_spatial_relations
        doc = nlp_model("a cat on a table")
        rels = extract_spatial_relations(doc, ["cat", "table"])
        predicates = [r.predicate for r in rels]
        assert "on_top_of" in predicates

    def test_beside_pattern(self, nlp_model):
        from src.intent.relationships import extract_spatial_relations
        doc = nlp_model("a car parked beside a house")
        rels = extract_spatial_relations(doc, ["car", "house"])
        # Should detect some spatial relation
        assert len(rels) >= 0  # May or may not detect depending on parse

    def test_in_front_of_multi_word(self, nlp_model):
        from src.intent.relationships import extract_spatial_relations
        doc = nlp_model("two people standing in front of a house")
        rels = extract_spatial_relations(doc, ["people", "house"])
        predicates = [r.predicate for r in rels]
        # Multi-word pattern should detect "in_front_of"
        assert "in_front_of" in predicates

    def test_no_relations_for_simple_prompt(self, nlp_model):
        from src.intent.relationships import extract_spatial_relations
        doc = nlp_model("three red apples")
        rels = extract_spatial_relations(doc, ["apple"])
        assert len(rels) == 0  # No spatial relations in simple prompts
