import numpy as np
from PIL import Image
import pytest

from src.verification.instance_color import masked_color_stats, summarize_class_color, mask_instances, refine_count_decision


@pytest.mark.parametrize("name,rgb", [("red", (220, 20, 20)), ("blue", (20, 40, 220)),
                                      ("green", (20, 180, 30)), ("yellow", (240, 220, 20))])
def test_color_measured_only_inside_object_mask(name, rgb):
    pixels = np.full((80, 80, 3), (20, 40, 220), dtype=np.uint8)
    pixels[20:60, 20:60] = rgb
    mask = np.zeros((80, 80), dtype=bool); mask[20:60, 20:60] = True
    actual = masked_color_stats(Image.fromarray(pixels), mask, .95)
    assert actual["observed_color"] == name


def test_weak_small_achromatic_and_mixed_objects_abstain():
    mask = np.ones((40, 40), dtype=bool)
    red = Image.new("RGB", (40, 40), (220, 10, 10))
    assert masked_color_stats(red, mask, .5)["observed_color"] is None
    assert masked_color_stats(Image.new("RGB", (40, 40), "white"), mask, .95)["observed_color"] is None
    pixels = np.asarray(red).copy(); pixels[:, 20:] = (10, 10, 220)
    assert masked_color_stats(Image.fromarray(pixels), mask, .95)["observed_color"] is None
    assert masked_color_stats(Image.new("RGB", (8, 8), "red"), np.ones((8, 8), bool), .95)["observed_color"] is None


@pytest.mark.parametrize("quality", [float("nan"), float("inf"), -1, 2])
def test_invalid_quality_rejected(quality):
    with pytest.raises(ValueError):
        masked_color_stats(Image.new("RGB", (40, 40), "red"), np.ones((40, 40), bool), quality)


def test_mask_shape_and_nonbinary_values_cannot_be_accepted():
    image = Image.new("RGB", (40, 40), "red")
    for mask in (np.ones((20, 20)), np.full((40, 40), float("nan")), np.full((40, 40), 255)):
        with pytest.raises(ValueError): masked_color_stats(image, mask, .95)


def test_missing_unknown_and_same_class_multi_color_instances_stay_unknown():
    assert summarize_class_color(["red"], [], 0)["status"] == "unknown"
    assert summarize_class_color(["red"], [{"observed_color": None}], 1)["status"] == "unknown"
    assert summarize_class_color(["red", "blue"], [{"observed_color": "red"}, {"observed_color": "blue"}], 2)["status"] == "unknown"
    assert summarize_class_color(["white"], [{"observed_color": "white"}], 1)["status"] == "unknown"


def test_observed_color_contradictions_are_diagnostics_not_csr():
    result = summarize_class_color(["red"], [{"observed_color": "blue"}], 1)
    assert result["status"] == "contradicted_diagnostic"
    assert result["scope"] == "uncalibrated_instance_color_diagnostic"
    assert summarize_class_color(["red"], [{"observed_color": "red"}], 1)["status"] == "supported_diagnostic"


def test_mask_identity_suppresses_part_whole_duplicates_without_using_expected_count():
    mask = np.zeros((80, 80), bool); mask[10:60, 10:60] = True
    entries = [{"class_name": "apple", "box": [10, 10, 60, 60], "mask_quality": .98},
               {"class_name": "apple", "box": [15, 20, 45, 50], "mask_quality": .95}]
    kept, ambiguous = mask_instances(entries, [mask, mask.copy()])
    assert kept == [0] and not ambiguous
    assert entries[1]["duplicate_of"] == 0


def test_overlapping_box_instances_with_distinct_masks_are_never_merged():
    left, right = np.zeros((80, 80), bool), np.zeros((80, 80), bool)
    left[10:60, 5:35] = True; right[10:60, 35:65] = True
    entries = [{"class_name": "apple", "box": [0, 0, 70, 70], "mask_quality": .98} for _ in range(2)]
    kept, ambiguous = mask_instances(entries, [left, right])
    assert kept == [0, 1] and not ambiguous


def test_one_mask_cannot_satisfy_two_different_object_classes():
    mask = np.ones((40, 40), bool)
    entries = [{"class_name": name, "box": [0, 0, 40, 40], "mask_quality": .98} for name in ("cat", "dog")]
    kept, ambiguous = mask_instances(entries, [mask, mask.copy()])
    assert len(kept) == 2 and len(ambiguous) == 1
    original = {"signals": {"expected_counts": {"cat": 1, "dog": 1}, "detected_counts": {"cat": 1, "dog": 1}}}
    colors = {"refined_boxes": {"cat": [{}], "dog": [{}]}, "mask_count_status": "unknown"}
    assert refine_count_decision(original, colors)["accepted"] is False
