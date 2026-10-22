"""Evidence accounting regressions for contract-level acceptance claims."""
from src.verification.contract_evidence import build_contract_evidence
from src.verification.qa import QAVerifier
from src.verification.verify_stage import VQAVerifier, color_matches


def _count_decision(**counts):
    return {"signals": {"expected_counts": counts, "detected_counts": counts}}


def test_color_and_negation_cannot_be_certified_by_count_pass():
    contract = {
        "entities": [{"class_name": "cat", "attributes": ["red"]}],
        "constraints": [{"type": "object_count", "target": "cat", "value": 1}],
        "negations": ["dog"],
    }
    evidence = build_contract_evidence(
        contract, _count_decision(cat=1), {},
        {"vqa_details": [{"type": "color", "class_name": "cat",
                          "expected": "red", "answer": "red", "correct": True}]},
    )
    assert evidence["status"] == "unverified"
    assert evidence["counts"] == {"total": 3, "passed": 1,
                                   "failed": 0, "unverified": 2}
    assert evidence["requirements"][1]["observation"] == {"vqa_answers": ["red"]}


def test_count_and_spatial_only_contract_has_covered_checks():
    relation = {"subject": "cat", "predicate": "left_of", "object": "chair"}
    contract = {
        "entities": [{"class_name": "cat"}, {"class_name": "chair"}],
        "spatial_relationships": [relation],
        "constraints": [{"type": "spatial_relation", "target": "cat->chair",
                         "value": relation}],
    }
    evidence = build_contract_evidence(
        contract, _count_decision(cat=1, chair=1),
        {"results": [{"passed": True, "reason": "left of"}]}, {},
    )
    assert evidence["status"] == "passed"
    assert evidence["counts"]["total"] == 3


def test_unchecked_depth_and_unknown_constraints_are_reported():
    contract = {
        "spatial_relationships": [
            {"subject": "cat", "predicate": "behind", "object": "chair"}],
        "constraints": [{"type": "legible_text", "target": "sign", "value": "OPEN"}],
    }
    evidence = build_contract_evidence(
        contract, {}, {"results": [{"passed": True, "reason": "area proxy"}]}, {},
    )
    assert evidence["status"] == "unverified"
    assert evidence["counts"]["unverified"] == 2


def test_archived_skipped_spatial_check_cannot_be_reclassified_as_pass():
    evidence = build_contract_evidence(
        {"spatial_relationships": [
            {"subject": "cat", "predicate": "on", "object": "chair"}]},
        {}, {"results": [{"passed": True,
                           "reason": "unknown predicate 'on': skipped"}]}, {},
    )
    assert evidence["status"] == "unverified"


def test_first_box_cannot_certify_relation_for_multiple_instances():
    relation = {"subject": "cat", "predicate": "left_of", "object": "chair"}
    evidence = build_contract_evidence(
        {"spatial_relationships": [relation]},
        {"detected_boxes": {"cat": [{}, {}], "chair": [{}]}},
        {"results": [{"passed": True, "reason": "first cat is left"}]}, {},
    )
    assert evidence["status"] == "unverified"


def test_empty_extraction_is_not_a_full_contract_pass():
    evidence = build_contract_evidence({}, {}, {}, {})
    assert evidence["status"] == "unverified"
    assert evidence["counts"]["total"] == 0


def test_zero_question_vqa_has_no_synthetic_perfect_score():
    verifier = object.__new__(VQAVerifier)
    result = verifier.score(None, {"entities": []})
    assert result["vqa_status"] == "not_applicable"
    assert result["vqa_score"] is None


def test_vqa_prose_and_partial_numbering_do_not_create_false_scores():
    verifier = object.__new__(VQAVerifier)
    verifier.router = type("Router", (), {
        "generate_vision": lambda self, **kwargs: "1. yes\n2. not red"
    })()
    from PIL import Image
    result = verifier.score(Image.new("RGB", (8, 8)), {
        "entities": [{"class_name": "cat", "attributes": ["red"]}],
    })
    assert result["vqa_status"] == "partial"
    assert result["vqa_score"] is None
    assert [d["correct"] for d in result["vqa_details"]] == [True, None]
    assert not color_matches("red", "not red")


def test_duplicate_detections_suppressed_without_merging_separate_objects():
    boxes = [
        {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100, "score": .90},
        {"xmin": 11, "ymin": 11, "xmax": 101, "ymax": 101, "score": .80},
        {"xmin": 60, "ymin": 10, "xmax": 150, "ymax": 100, "score": .70},
    ]
    kept = QAVerifier._dedupe_boxes(boxes)
    assert [box["score"] for box in kept] == [.90, .70]
