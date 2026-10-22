"""Account for every extracted visual requirement without inventing a CSR result.

This is an evidence ledger, not a calibrated visual judge. The existing
detector and box geometry supply operational count/spatial observations;
VQA answers remain diagnostic until validated against human labels.
"""
from collections import Counter
from typing import Any, Dict


def build_contract_evidence(
    contract: Dict[str, Any],
    count_decision: Dict[str, Any],
    spatial_result: Dict[str, Any],
    vqa_scores: Dict[str, Any],
    local_instances: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    contract = contract or {}
    signals = count_decision.get("signals") or {}
    expected_counts = signals.get("expected_counts") or {}
    detected_counts = signals.get("detected_counts") or {}
    spatial_results = spatial_result.get("results") or []
    vqa_details = vqa_scores.get("vqa_details") or []
    entities = contract.get("entities") or []
    requirements = []
    seen = set()

    def add(kind, target, expected, status="unverified", method=None,
            reason=None, observation=None):
        key = (kind, str(target).lower(), str(expected).lower())
        if key in seen:
            return
        seen.add(key)
        entry = {"kind": kind, "target": target, "expected": expected,
                 "status": status}
        if method:
            entry["method"] = method
        if reason:
            entry["reason"] = reason
        if observation is not None:
            entry["observation"] = observation
        requirements.append(entry)

    declared_counts = Counter(
        str(e.get("class_name", "")).strip().lower()
        for e in entities if isinstance(e, dict) and not e.get("uncounted")
        and str(e.get("class_name", "")).strip()
    )
    for class_name, expected in declared_counts.items():
        if class_name in expected_counts and class_name in detected_counts:
            detected = detected_counts[class_name]
            add("object_count", class_name, expected,
                "unverified" if count_decision.get("instance_count_status") == "unknown" else
                ("passed" if detected == expected else "failed"),
                count_decision.get("count_method", "grounding_dino"), observation={"detected": detected})
        else:
            add("object_count", class_name, expected, reason="count evidence missing")

    uncounted_names = {
        str(e.get("class_name", "")).strip().lower()
        for e in entities if isinstance(e, dict) and e.get("uncounted")
    }
    for class_name in sorted(uncounted_names - {""}):
        answers = [d.get("answer") for d in vqa_details
                   if d.get("type") == "existence"
                   and str(d.get("class_name", "")).lower() == class_name]
        add("object_presence", class_name, True,
            reason="vision QA is diagnostic, not a calibrated presence gate",
            observation={"vqa_answers": answers} if answers else None)

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        class_name = str(entity.get("class_name", "")).strip().lower()
        if not class_name:
            continue
        for attribute in entity.get("attributes") or []:
            attribute = str(attribute).strip().lower()
            if not attribute:
                continue
            answers = [d.get("answer") for d in vqa_details
                       if d.get("type") == "color"
                       and str(d.get("class_name", "")).lower() == class_name
                       and str(d.get("expected", "")).lower() == attribute]
            local = (local_instances or {}).get("classes", {}).get(class_name)
            contradicted = bool(local and attribute in local.get("expected_colors", [])
                                and local.get("status") == "contradicted_diagnostic")
            observation = {"vqa_answers": answers} if answers else {}
            if local:
                observation["local_instances"] = local
            add("attribute", class_name, attribute, "failed" if contradicted else "unverified",
                method="segmented_chromatic_measurement" if local else None,
                reason="attribute binding has no calibrated hard verifier",
                observation=observation or None)
        if entity.get("relative_position") is not None:
            add("relative_position", class_name, entity["relative_position"],
                reason="entity canvas position is not checked")
        if entity.get("bounding_box") is not None:
            add("bounding_box", class_name, entity["bounding_box"],
                reason="declared entity box is not checked")

    relationships = contract.get("spatial_relationships") or []
    for index, relation in enumerate(relationships):
        if not isinstance(relation, dict):
            add("spatial_relation", "unknown", relation,
                reason="malformed relation")
            continue
        subject = str(relation.get("subject", "")).strip().lower()
        obj = str(relation.get("object", "")).strip().lower()
        predicate = str(relation.get("predicate", "")).strip().lower()
        result = spatial_results[index] if index < len(spatial_results) else None
        if result is None:
            add("spatial_relation", f"{subject}->{obj}", predicate,
                reason="spatial evidence missing")
            continue
        reason = str(result.get("reason", ""))
        # Box area is a perspective hint, not evidence of occlusion/depth.
        # An unsupported predicate also cannot be counted as a visual fail.
        detected_boxes = count_decision.get("detected_boxes") or {}
        instance_ambiguous = (len(detected_boxes.get(subject, [])) > 1
                              or len(detected_boxes.get(obj, [])) > 1)
        unverifiable = (predicate in {"in_front_of", "behind", "between"}
                        or instance_ambiguous
                        or "unsupported predicate" in reason
                        or "unknown predicate" in reason
                        or "skipped" in reason
                        or "requires two reference" in reason)
        add("spatial_relation", f"{subject}->{obj}", predicate,
            "unverified" if unverifiable else
            ("passed" if result.get("passed") else "failed"),
            "box_geometry" if not unverifiable else None,
            reason=reason)

    for negation in contract.get("negations") or []:
        add("negation", str(negation), "absent",
            reason="absence has no calibrated verifier")

    for constraint in contract.get("constraints") or []:
        if not isinstance(constraint, dict):
            add("other_constraint", "unknown", constraint,
                reason="malformed constraint")
            continue
        kind = str(constraint.get("type", "other_constraint"))
        target = str(constraint.get("target", ""))
        expected = constraint.get("value")
        if kind == "object_count":
            class_name = target.strip().lower()
            if class_name in declared_counts and expected == declared_counts[class_name]:
                continue
            add("object_count", target, expected,
                reason="explicit count lacks matching entity count evidence")
        elif kind == "spatial_relation" and isinstance(expected, dict):
            if expected in relationships:
                continue
            add(kind, target, expected,
                reason="explicit relation lacks spatial evidence")
        else:
            add(kind, target, expected,
                reason="constraint type has no verifier")

    counts = {name: sum(item["status"] == name for item in requirements)
              for name in ("passed", "failed", "unverified")}
    status = ("failed" if counts["failed"] else
              "unverified" if counts["unverified"] or not requirements else
              "passed")
    return {
        "status": status,
        "scope": "extracted_contract_only",
        "counts": {"total": len(requirements), **counts},
        "requirements": requirements,
    }
