"""
Spatial Relation Verifier — validates spatial relationships between detected
objects using bounding box geometry from GroundingDINO.

Uses the contract's spatial relationships (from intent/relationships.py)
and the verification stage's detected bounding boxes to check:
    left_of(A, B)    → A.center_x < B.center_x
    right_of(A, B)   → A.center_x > B.center_x
    above(A, B)      → A.center_y < B.center_y (image coords: y=0 is top)
    below(A, B)      → A.center_y > B.center_y
    on_top_of(A, B)  → A above B, bottom/top gap <= 15% image height,
                        horizontal overlap > 0.3
    beside(A, B)     → |A.center_x - B.center_x| > min_sep AND overlap_y > 0.3
    in_front_of(A,B) → A.area > B.area (perspective proxy: closer = larger)
    behind(A, B)     → A.area < B.area
    inside(A, B)     → >= 70% of A's box inside B and A.area < B.area
    near(A, B)       → distance(centers) < 0.3 * image_diagonal

Returns a verification result with per-relationship pass/fail and failure details.

VRAM: 0 MB — pure geometry computation on CPU.
"""
from typing import Dict, List, Any, Optional
import math
import re


class SpatialRelationVerifier:
    """Validates spatial relations using bounding box geometry."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        # Configurable tolerances
        self.overlap_threshold = 0.3   # Minimum IoU for containment/adjacency
        self.proximity_threshold = 0.3  # Fraction of diagonal for "near"
        self.separation_margin = 0.05   # Minimum center distance for beside
        self.contact_threshold = 0.15   # Fraction of image height for support contact

    def verify(
        self,
        relationships: List[Dict[str, str]],
        detected_boxes: Dict[str, List[Dict[str, float]]],
        image_width: int = 512,
        image_height: int = 512,
    ) -> Dict[str, Any]:
        """
        Verify spatial relationships against detected bounding boxes.

        Args:
            relationships: List of dicts with keys: subject, predicate, object.
            detected_boxes: Dict mapping class_name → list of bounding boxes
                           (each box: {xmin, ymin, xmax, ymax} in pixel coords).
            image_width: Width of the generated image.
            image_height: Height of the generated image.

        Returns:
            Dict with keys:
                passed: bool — all relationships verified
                total: int — number of relationships checked
                verified: int — number that passed
                failures: list — details of failed relationships
                results: list — per-relationship results
        """
        if not relationships:
            return {
                "passed": True,
                "total": 0,
                "verified": 0,
                "failures": [],
                "results": [],
            }

        results = []
        failures = []
        verified = 0

        for rel in relationships:
            subject = rel.get("subject", "").lower()
            predicate = str(rel.get("predicate", "")).strip().lower()
            obj = rel.get("object", "").lower()

            # Find matching bounding boxes
            subject_boxes = self._find_boxes(subject, detected_boxes)
            object_boxes = self._find_boxes(obj, detected_boxes)

            if not subject_boxes:
                result = {
                    "relationship": rel,
                    "passed": False,
                    "reason": f"subject '{subject}' not detected",
                }
                results.append(result)
                failures.append(result)
                continue

            if not object_boxes:
                result = {
                    "relationship": rel,
                    "passed": False,
                    "reason": f"object '{obj}' not detected",
                }
                results.append(result)
                failures.append(result)
                continue

            # Use the highest-confidence (first) box for each entity
            s_box = subject_boxes[0]
            o_box = object_boxes[0]

            passed, reason = self._check_relation(
                predicate, s_box, o_box, image_width, image_height
            )

            result = {
                "relationship": rel,
                "passed": passed,
                "reason": reason,
                "subject_box": s_box,
                "object_box": o_box,
            }
            results.append(result)
            if passed:
                verified += 1
            else:
                failures.append(result)

        return {
            "passed": len(failures) == 0,
            "total": len(relationships),
            "verified": verified,
            "failures": [
                {
                    "subject": f["relationship"]["subject"],
                    "predicate": f["relationship"]["predicate"],
                    "object": f["relationship"]["object"],
                    "reason": f["reason"],
                }
                for f in failures
            ],
            "results": results,
        }

    def _find_boxes(
        self, entity_name: str, detected_boxes: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """Find bounding boxes for an entity, with fuzzy matching."""
        if not entity_name:
            return []
        # Exact match
        if entity_name in detected_boxes:
            return detected_boxes[entity_name]

        # Allow whole-word descriptive phrases, not incidental substrings.
        for class_name, boxes in detected_boxes.items():
            class_name = class_name.lower()
            if not class_name:
                continue
            if (re.search(rf"(?<!\w){re.escape(entity_name)}(?!\w)", class_name)
                    or re.search(rf"(?<!\w){re.escape(class_name)}(?!\w)", entity_name)):
                return boxes

        return []

    def _check_relation(
        self,
        predicate: str,
        s_box: Dict[str, float],
        o_box: Dict[str, float],
        img_w: int,
        img_h: int,
    ) -> tuple:
        """Check a specific spatial relation. Returns (passed, reason)."""
        s_cx, s_cy = self._center(s_box)
        o_cx, o_cy = self._center(o_box)
        s_area = self._area(s_box)
        o_area = self._area(o_box)
        diag = math.sqrt(img_w ** 2 + img_h ** 2)

        # The intent LLM can emit natural variants despite the canonical
        # predicate instructions. Normalize supported synonyms explicitly.
        predicate = {
            "on": "on_top_of",
            "atop": "on_top_of",
            "standing_on": "on_top_of",
            "placed_on": "on_top_of",
        }.get(predicate, predicate)

        if predicate == "left_of":
            passed = s_cx < o_cx
            reason = f"subject center_x={s_cx:.0f} {'<' if passed else '>='} object center_x={o_cx:.0f}"
            return passed, reason

        elif predicate == "right_of":
            passed = s_cx > o_cx
            reason = f"subject center_x={s_cx:.0f} {'>' if passed else '<='} object center_x={o_cx:.0f}"
            return passed, reason

        elif predicate == "above":
            # Image coords: y=0 is top, so "above" means lower y value
            passed = s_cy < o_cy
            reason = f"subject center_y={s_cy:.0f} {'<' if passed else '>='} object center_y={o_cy:.0f}"
            return passed, reason

        elif predicate == "below":
            passed = s_cy > o_cy
            reason = f"subject center_y={s_cy:.0f} {'>' if passed else '<='} object center_y={o_cy:.0f}"
            return passed, reason

        elif predicate == "on_top_of":
            # The subject's bottom should meet the support object's top.
            # Merely ending above the object's bottom lets full-image boxes
            # certify an unrelated "on" relationship.
            y_gap = abs(s_box["ymax"] - o_box["ymin"]) / img_h
            y_ok = s_cy < o_cy and y_gap <= self.contact_threshold
            x_overlap = self._overlap_x(s_box, o_box)
            passed = y_ok and x_overlap > self.overlap_threshold
            reason = (
                f"support_contact={'pass' if y_ok else 'fail'} "
                f"(gap={y_gap:.2f}, max={self.contact_threshold}), "
                f"x_overlap={x_overlap:.2f} (threshold={self.overlap_threshold})"
            )
            return passed, reason

        elif predicate == "beside":
            x_dist = abs(s_cx - o_cx) / img_w
            y_overlap = self._overlap_y(s_box, o_box)
            passed = x_dist > self.separation_margin and y_overlap > self.overlap_threshold
            reason = (
                f"x_distance={x_dist:.2f} (min={self.separation_margin}), "
                f"y_overlap={y_overlap:.2f} (min={self.overlap_threshold})"
            )
            return passed, reason

        elif predicate == "in_front_of":
            # Perspective proxy: closer objects appear larger
            passed = s_area > o_area * 0.8  # Allow some tolerance
            reason = f"subject_area={s_area:.0f} vs object_area={o_area:.0f}"
            return passed, reason

        elif predicate == "behind":
            passed = s_area < o_area * 1.2
            reason = f"subject_area={s_area:.0f} vs object_area={o_area:.0f}"
            return passed, reason

        elif predicate == "inside":
            intersection = self._intersection(s_box, o_box)
            containment = intersection / s_area if s_area > 0 else 0
            size_ok = s_area < o_area
            passed = containment >= 0.7 and size_ok
            reason = f"containment={containment:.2f} (min=0.70), size_ok={size_ok}"
            return passed, reason

        elif predicate == "near":
            dist = math.sqrt((s_cx - o_cx) ** 2 + (s_cy - o_cy) ** 2)
            threshold = diag * self.proximity_threshold
            passed = dist < threshold
            reason = f"distance={dist:.0f} (threshold={threshold:.0f})"
            return passed, reason

        elif predicate == "surrounding":
            # Subject surrounds object: object is contained within subject and subject is larger
            intersection = self._intersection(s_box, o_box)
            containment = intersection / o_area if o_area > 0 else 0
            size_ok = s_area > o_area
            passed = containment >= 0.7 and size_ok
            reason = f"containment={containment:.2f} (min=0.70), size_ok={size_ok}"
            return passed, reason

        elif predicate == "between":
            # This schema has only one reference object. A between check
            # needs two anchors, so the relation cannot be verified here.
            return False, "between requires two reference objects"

        else:
            return False, f"unsupported predicate '{predicate}'"

    def _center(self, box: Dict[str, float]) -> tuple:
        return (box["xmin"] + box["xmax"]) / 2, (box["ymin"] + box["ymax"]) / 2

    def _area(self, box: Dict[str, float]) -> float:
        return max(0, box["xmax"] - box["xmin"]) * max(0, box["ymax"] - box["ymin"])

    def _intersection(self, a: Dict, b: Dict) -> float:
        """Area of intersection between two bounding boxes."""
        x1 = max(a["xmin"], b["xmin"])
        y1 = max(a["ymin"], b["ymin"])
        x2 = min(a["xmax"], b["xmax"])
        y2 = min(a["ymax"], b["ymax"])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _overlap_x(self, a: Dict, b: Dict) -> float:
        """Fraction of a's width that overlaps with b horizontally."""
        overlap = max(0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
        width_a = a["xmax"] - a["xmin"]
        return overlap / width_a if width_a > 0 else 0

    def _overlap_y(self, a: Dict, b: Dict) -> float:
        """Fraction of a's height that overlaps with b vertically."""
        overlap = max(0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
        height_a = a["ymax"] - a["ymin"]
        return overlap / height_a if height_a > 0 else 0

    def _iou(self, a: Dict, b: Dict) -> float:
        """Intersection over Union of two boxes."""
        intersection = self._intersection(a, b)
        area_a = self._area(a)
        area_b = self._area(b)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0
