"""Safe VSM data preparation and explicitly scoped RGB template diagnostics.

Implements the published data format independently. No downloaded script runs.
"""
from io import BytesIO
import hashlib
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np
from PIL import Image

from src.research.artifacts import atomic_json, json_digest, sha256, workspace_path

EXPECTED_TEMPLATES = {f"{suit}{rank}" for suit in "CDHS" for rank in range(1, 11)}


def extract_pngs(archive_path: Path, destination: Path, *, limit: int, template=False):
    """Read bounded regular PNG members only; reject links and unsafe paths."""
    if archive_path.stat().st_size > 750_000_000:
        raise ValueError("Research archive exceeds its byte budget")
    if destination.is_symlink() or getattr(destination, "is_junction", lambda: False)():
        raise ValueError("Image destination cannot be a directory link")
    destination.mkdir(parents=True, exist_ok=True)
    written, ignored, seen = [], [], set()
    with tarfile.open(archive_path, "r:") as archive:
        for index, member in enumerate(archive):
            if index > 200_000:
                raise ValueError("Archive member budget exceeded")
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or ":" in member.name or member.issym() or member.islnk()):
                raise ValueError("Unsafe research archive member")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Special archive members are unsupported")
            if name.suffix.lower() != ".png":
                ignored.append(str(name))
                continue
            expected_parent = "resized_templates_png" if template else "card_imgs"
            if len(name.parts) != 2 or name.parts[0] != expected_parent:
                raise ValueError("Unexpected PNG archive layout")
            if member.size <= 0 or member.size > 1_000_000 or name.name in seen:
                raise ValueError("Invalid or duplicate PNG member")
            seen.add(name.name)
            raw = archive.extractfile(member).read(1_000_001)
            if len(raw) != member.size:
                raise ValueError("Truncated PNG archive member")
            with Image.open(BytesIO(raw)) as image:
                expected_size = (64, 64) if template else (128, 128)
                if image.format != "PNG" or image.size != expected_size:
                    raise ValueError("Unexpected research image format or dimensions")
                image.verify()
            target = destination / name.name
            if target.is_symlink():
                raise ValueError("Image destination cannot be a symlink")
            if target.exists() and target.read_bytes() != raw:
                raise ValueError("Existing extracted image differs from source")
            if not target.exists():
                target.write_bytes(raw)
            written.append(target)
            if len(written) == limit:
                break
    return written, ignored


class CardTemplates:
    def __init__(self, directory: Path):
        paths = sorted(directory.glob("*.png"))
        if {p.stem for p in paths} != EXPECTED_TEMPLATES or len(paths) != 40:
            raise ValueError("The complete 40-template inventory is required")
        self.names = [p.stem for p in paths]
        values, self.manifest = [], []
        for path in paths:
            with Image.open(path) as image:
                if image.size != (64, 64) or image.mode not in {"RGB", "RGBA", "P"}:
                    raise ValueError("Invalid card template dimensions or channels")
                values.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
            self.manifest.append({"id": path.stem, "sha256": sha256(path), "size": [64, 64]})
        self.values = np.stack(values)
        # Per-channel centering retains color information, plus explicit RGB RMSE.
        flat = self.values.reshape(40, -1, 3)
        centered = (flat - flat.mean(axis=1, keepdims=True)).reshape(40, -1)
        self.normalized = centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(1e-12)

    def inspect(self, image, threshold=.95, margin=.01):
        pixels = np.asarray(image)
        if pixels.shape != (128, 128, 3) or pixels.dtype != np.uint8:
            raise ValueError("Cards diagnostics require 128x128 uint8 RGB pixels")
        if float(pixels.std()) < 1:
            return {"status": "invalid", "reason": "blank", "quadrants": [],
                    "features": [0.] * 12}
        rows, features = [], []
        for y, x in ((0, 0), (0, 64), (64, 0), (64, 64)):
            tile = pixels[y:y+64, x:x+64].astype(np.float32) / 255
            centered = tile.reshape(-1, 3) - tile.reshape(-1, 3).mean(axis=0)
            vector = centered.reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm < 1e-8:
                scores = np.full(40, -1., dtype=np.float32)
            else:
                scores = self.normalized @ (vector / norm)
            order = np.argsort(scores)
            best = int(order[-1])
            score = float(np.clip(scores[best], -1, 1))
            gap = float(score - scores[order[-2]])
            rmse = float(np.sqrt(np.mean((tile - self.values[best])**2)))
            status = "invalid" if score < threshold or rmse > .10 else (
                "unknown" if gap < margin else "valid")
            rows.append({"template": self.names[best], "ncc": score,
                         "margin": gap, "rgb_rmse": rmse, "status": status})
            features.extend([score, gap, rmse])
        status = "invalid" if any(r["status"] == "invalid" for r in rows) else (
            "unknown" if any(r["status"] == "unknown" for r in rows) else "valid")
        return {"status": status, "quadrants": rows, "features": features}


def family_split(family: str) -> str:
    bucket = int(hashlib.sha256(("cards-split-v1:" + family).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def prepare_data(config):
    root = workspace_path(config.data_root)
    for name, expected in (("card_imgs.tar", config.cards_archive_sha256),
                           ("templates.tar", config.templates_archive_sha256)):
        if sha256(root / name) != expected:
            raise ValueError(f"Changed research archive: {name}")
    templates, ignored = extract_pngs(root / "templates.tar", root / "templates", limit=41, template=True)
    matcher = CardTemplates(root / "templates")
    paths, _ = extract_pngs(root / "card_imgs.tar", root / "subset", limit=config.max_records)
    records, seen_pixels = [], set()
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        observation = matcher.inspect(rgb, config.ncc_threshold, config.ambiguity_margin)
        identities = [q["template"] for q in observation["quadrants"]]
        family = "|".join(identities) if observation["status"] == "valid" else "unknown:" + pixel_hash
        if pixel_hash in seen_pixels:
            continue
        seen_pixels.add(pixel_hash)
        records.append({"file": path.name, "sha256": sha256(path), "pixel_sha256": pixel_hash,
                        "family": family, "split": family_split(family),
                        "source_status": observation["status"], "templates": identities})
    split_counts = {split: sum(r["split"] == split and r["source_status"] == "valid" for r in records)
                    for split in ("train", "validation", "test")}
    if min(split_counts.values()) < 2:
        raise ValueError("Insufficient verified source families in each split")
    manifest = {"schema": "framegenx-cards-data-v1", "dataset": "mbhosale/VSM",
                "revision": config.dataset_revision, "license": "CC-BY-4.0",
                "attribution": "Bhosale et al., Score-Control for Hallucination Reduction in Diffusion Models, 2026",
                "data_scope": "first bounded archive subset, not representative full benchmark",
                "archive_hashes": {"cards": config.cards_archive_sha256, "templates": config.templates_archive_sha256},
                "templates": matcher.manifest, "ignored_nonimage_members": ignored,
                "records": records, "verified_split_counts": split_counts,
                "quality_status": "unverified"}
    digest = json_digest(manifest)
    manifest_file = root / ("prepared_" + digest[:16] + ".json")
    atomic_json(manifest_file, manifest)
    return manifest_file


def load_data(manifest_file: Path, config):
    import json
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "framegenx-cards-data-v1" or
            manifest.get("revision") != config.dataset_revision):
        raise ValueError("Incompatible research data manifest")
    root = workspace_path(config.data_root)
    records, families = manifest["records"], {}
    for record in records:
        name = record["file"]
        if not re.fullmatch(r"grid_\d+\.png", name):
            raise ValueError("Invalid source image identity")
        if record["split"] != family_split(record["family"]):
            raise ValueError("Research split identity changed")
        if record["family"] in families and families[record["family"]] != record["split"]:
            raise ValueError("Research family leaked across splits")
        families[record["family"]] = record["split"]
        if sha256(root / "subset" / name) != record["sha256"]:
            raise ValueError("Research source pixels changed")
    matcher = CardTemplates(root / "templates")
    if matcher.manifest != manifest["templates"]:
        raise ValueError("Research templates changed")
    return manifest, matcher
