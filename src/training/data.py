"""Validate training records and keep image groups out of other splits."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from src.training.common import file_sha256


SPLITS = {"train", "validation", "test"}


def pixel_sha256(path: Path) -> str:
    """Ignore encoding/metadata differences when detecting exact image leakage."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


def image_group(record: dict) -> str:
    return record.get("image_pixel_sha256", record["image_sha256"])


def split_reservations(manifests: list[Path]) -> tuple[dict[str, str], list[dict]]:
    """Freeze prior image assignments before expanding a training corpus."""
    reserved, provenance = {}, []
    for path in manifests:
        rows, audit = validate_records(path)
        for row in rows:
            split = row["split"]
            for digest in (row["image_sha256"], pixel_sha256(path.parent / row["image"])):
                if digest in reserved and reserved[digest] != split:
                    raise ValueError("Prior manifests disagree about an image split")
                reserved[digest] = split
        provenance.append({"path": str(path), "dataset_sha256": audit["dataset_sha256"],
                           "image_groups": audit["image_groups"]})
    return reserved, provenance


def assign_splits(records: list[dict], seed: int = 42,
                  reserved_splits: dict[str, str] | None = None) -> list[dict]:
    if reserved_splits is not None and any(
            not re.fullmatch(r"[0-9a-f]{64}", key) or value not in SPLITS
            for key, value in reserved_splits.items()):
        raise ValueError("Invalid reserved image split")
    by_source = defaultdict(set)
    for record in records:
        if record.get("positive") is True:
            by_source[record["source"]].add(image_group(record))
    assigned = dict(reserved_splits or {})
    for source in sorted(by_source):
        groups = sorted(by_source[source], key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode()).hexdigest())
        if len(groups) < 5:
            raise ValueError(f"Source {source} needs at least five independent image groups")
        validation_n = max(1, len(groups) // 10)
        test_n = max(1, len(groups) // 10)
        if reserved_splits is None:
            # Preserve the original pilot's exact deterministic assignment.
            for index, group in enumerate(groups):
                split = ("test" if index < test_n else "validation"
                         if index < test_n + validation_n else "train")
                assigned.setdefault(group, split)
        else:
            remaining = [group for group in groups if group not in assigned]
            for split, wanted in (("test", test_n), ("validation", validation_n)):
                deficit = max(0, wanted - sum(assigned.get(group) == split for group in groups))
                for group in remaining[:deficit]:
                    assigned[group] = split
                remaining = remaining[deficit:]
            for group in remaining:
                assigned[group] = "train"
        if set(assigned[group] for group in groups) != SPLITS:
            raise ValueError(f"Source {source} cannot have three independent splits under its reservations")
    return [{**record, "split": assigned.get(image_group(record), "test")}
            for record in records]


def validate_records(path: Path, verify_images: bool = True) -> tuple[list[dict], dict]:
    root = path.parent.resolve()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError("Training dataset is empty")
    ids, image_splits, pixel_splits, seen_pairs, pixel_cache = set(), {}, {}, set(), {}
    counts = defaultdict(Counter)
    for record in records:
        required = {"id", "source", "caption", "image", "image_sha256", "positive",
                    "split", "license", "provenance", "constraints"}
        if not required.issubset(record):
            raise ValueError(f"Record is missing fields: {sorted(required - record.keys())}")
        if not all(isinstance(record[field], str) and record[field].strip()
                   for field in ("id", "source", "image", "image_sha256")):
            raise ValueError("Record identity and image fields must be nonempty strings")
        if not re.fullmatch(r"[0-9a-f]{64}", record["image_sha256"]):
            raise ValueError("Invalid training image SHA256")
        if record["id"] in ids:
            raise ValueError("Duplicate training record id")
        ids.add(record["id"])
        if record["split"] not in SPLITS or not isinstance(record["positive"], bool):
            raise ValueError("Invalid split or positivity label")
        if not isinstance(record["caption"], str) or not record["caption"].strip():
            raise ValueError("Empty training caption")
        if (not isinstance(record["license"], dict) or not record["license"].get("images")
                or not isinstance(record["provenance"], dict) or not record["provenance"]
                or not isinstance(record["constraints"], list) or not record["constraints"]
                or not all(isinstance(c, dict) and c.get("type") for c in record["constraints"])):
            raise ValueError("Missing license, provenance or constraint labels")
        image_path = (root / record["image"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError("Training image is missing or outside its dataset root")
        if verify_images:
            if file_sha256(image_path) != record["image_sha256"]:
                raise ValueError("Training image hash changed")
            with Image.open(image_path) as image:
                image.verify()
            if record["image_sha256"] not in pixel_cache:
                pixel_cache[record["image_sha256"]] = pixel_sha256(image_path)
            pixel = pixel_cache[record["image_sha256"]]
            if record.get("image_pixel_sha256", pixel) != pixel:
                raise ValueError("Training image pixel hash changed")
            previous_pixel_split = pixel_splits.setdefault(pixel, record["split"])
            if previous_pixel_split != record["split"]:
                raise ValueError("Pixel-identical image leakage between training and held-out splits")
        else:
            pixel = image_group(record)
        prior = image_splits.setdefault(record["image_sha256"], record["split"])
        if prior != record["split"]:
            raise ValueError("Image leakage between training and held-out splits")
        pair = (pixel, record["caption"].strip().lower())
        if pair in seen_pairs:
            raise ValueError("Duplicate image/caption pair")
        seen_pairs.add(pair)
        if record["positive"]:
            counts[record["source"]][record["split"]] += 1
    for source, splits in counts.items():
        if any(splits[split] == 0 for split in SPLITS):
            raise ValueError(f"Source {source} lacks a positive training/validation/test split")
    if not counts:
        raise ValueError("Dataset has no positive training examples")
    return records, {"records": len(records), "positive_by_source_and_split": dict(counts),
                     "negative_examples": sum(not r["positive"] for r in records),
                     "dataset_sha256": file_sha256(path), "image_groups": len(image_splits),
                     "decoded_pixel_groups": len(pixel_splits) if verify_images else None,
                     "image_verification_performed": verify_images}
