"""Data expansion must never recycle earlier held-out images into fitting."""
import hashlib
from pathlib import Path

import pytest

from src.training.data import assign_splits


def rows(count, source="clevr"):
    return [{"image_sha256": hashlib.sha256(f"image-{i}".encode()).hexdigest(),
             "source": source, "positive": True} for i in range(count)]


def test_expansion_preserves_all_prior_assignments_across_different_seed():
    original = assign_splits(rows(30), seed=42)
    reservations = {r["image_sha256"]: r["split"] for r in original}
    expanded = assign_splits(rows(100), seed=99, reserved_splits=reservations)
    assert {r["image_sha256"]: r["split"] for r in expanded[:30]} == reservations
    assert {r["split"] for r in expanded} == {"train", "validation", "test"}
    assert sum(r["split"] == "train" for r in expanded) == 80


def test_same_image_from_two_sources_and_negative_statement_share_one_split():
    data = rows(20, "vsr") + rows(20, "coco")
    data.append({**data[0], "positive": False})
    expanded = assign_splits(data, reserved_splits={})
    groups = {}
    for row in expanded:
        assert groups.setdefault(row["image_sha256"], row["split"]) == row["split"]


def test_reservations_are_not_overwritten_to_make_a_dataset_look_balanced():
    data = rows(10)
    reserved = {r["image_sha256"]: "train" for r in data}
    with pytest.raises(ValueError, match="three independent splits"):
        assign_splits(data, reserved_splits=reserved)
    assert set(reserved.values()) == {"train"}


@pytest.mark.parametrize("reservation", [{"../image": "test"}, {"a" * 64: "heldout"}])
def test_invalid_reservations_are_rejected(reservation):
    with pytest.raises(ValueError, match="Invalid reserved"):
        assign_splits(rows(20), reserved_splits=reservation)


def test_legacy_seed_assignment_remains_stable():
    original = assign_splits(rows(30), seed=42)
    assert original == assign_splits(rows(30), seed=42)
    assert len([r for r in original if r["split"] == "test"]) == 3
