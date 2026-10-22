"""Acquire a bounded, attributed constraint dataset without executing dataset code.

Only consensus-positive VSR statements supervise generation. False statements
are retained as evaluation labels. COCO counts exclude crowd and tiny objects;
they are annotation-derived counts, not a promise of exhaustive human truth.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import zipfile

import ijson
from PIL import Image
import requests

from src.training.common import atomic_json, file_sha256, project_path
from src.training.data import assign_splits, validate_records, split_reservations, pixel_sha256
from src.core.files import atomic_replace, remove_temporary


CLEVR_REVISION = "463520cb0e4d16f2f61ae0607c81eacdd009e9f4"
CLEVR_REPO = "maujim/CLEVR_sample_500"
# The public COCO bucket's vanity HTTPS hostname currently has a mismatched
# certificate. S3's path endpoint serves the same objects with valid TLS.
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ANNOTATIONS = COCO_BASE + "/annotations/annotations_trainval2017.zip"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def replace_download(temporary: Path, destination: Path):
    """Bound transient Windows scanner/reader sharing violations."""
    atomic_replace(temporary, destination)


def download(url: str, path: Path, max_bytes: int, retries: int = 3) -> Path:
    """Stream into an exact temporary file; never use remote filenames as paths."""
    if path.is_file() and path.stat().st_size:
        if path.stat().st_size > max_bytes:
            raise ValueError("Cached download exceeds size budget")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        partial = None
        try:
            size = 0
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                declared_size = int(response.headers.get("Content-Length", 0))
                if declared_size > max_bytes:
                    raise ValueError("Download exceeds size budget")
                fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
                partial = Path(temporary_name)
                with os.fdopen(fd, "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("Download exceeds size budget")
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if not size:
                raise ValueError("Downloaded file is empty")
            if declared_size and size != declared_size and not response.headers.get("Content-Encoding"):
                raise requests.RequestException("Download length differs from its response header")
            replace_download(partial, path)
            return path
        except (requests.RequestException, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(attempt + 1)
        finally:
            if partial is not None:
                remove_temporary(partial)
    raise RuntimeError("Download did not complete")


def image_record(root: Path, path: Path, **fields) -> dict:
    with Image.open(path) as image:
        image.verify()
    return {**fields, "image": path.relative_to(root).as_posix(),
            "image_sha256": file_sha256(path), "image_pixel_sha256": pixel_sha256(path)}


def clevr_caption(scene: dict) -> tuple[str, list[dict]]:
    """Describe the complete scene, then an unambiguous left/right relation."""
    objects = scene["objects"]
    descriptors = [f"{o['size']} {o['color']} {o['material']} {o['shape']}" for o in objects]
    counts = Counter(descriptors)
    parts = [f"{count} {desc}{'s' if count > 1 else ''}" for desc, count in counts.items()]
    caption = "A rendered scene with " + ", ".join(parts) + "."
    constraints = [{"type": "object_count", "shape": shape,
                    "count": sum(o["shape"] == shape for o in objects)}
                   for shape in sorted({o["shape"] for o in objects})]
    constraints += [{"type": "attribute_binding", "object_index": i, **obj}
                    for i, obj in enumerate(objects)]
    for subject, descriptor in enumerate(descriptors):
        if counts[descriptor] != 1:
            continue
        for other in scene["relationships"]["left"][subject]:
            if counts[descriptors[other]] == 1:
                caption += f" The {descriptors[other]} is left of the {descriptor}."
                constraints.append({"type": "spatial", "subject_index": other,
                                    "object_index": subject, "relation": "left_of"})
                return caption, constraints
    return caption, constraints


def prepare_clevr(root: Path, count: int, rng: random.Random) -> list[dict]:
    base = f"https://huggingface.co/datasets/{CLEVR_REPO}/resolve/{CLEVR_REVISION}"
    for filename in ("LICENSE.txt", "COPYRIGHT.txt"):
        download(f"{base}/{filename}", root / "sources" / ("clevr_" + filename), 100_000)
    metadata = download(f"{base}/scenes/CLEVR_train_scenes.json",
                        root / "sources/clevr_train_scenes.json", 25_000_000)
    scenes = json.loads(metadata.read_text(encoding="utf-8"))["scenes"]
    candidates = [s for s in scenes if 3 <= len(s["objects"]) <= 4]
    rng.shuffle(candidates)
    records = []
    for scene in candidates:
        filename = scene["image_filename"]
        if not SAFE_NAME.fullmatch(filename):
            raise ValueError("Unsafe dataset image name")
        path = download(f"{base}/images/train/{filename}", root / "images/clevr" / filename, 8_000_000)
        caption, constraints = clevr_caption(scene)
        records.append(image_record(root, path, id="clevr:" + filename, source="clevr",
            caption=caption, positive=True, constraints=constraints,
            license={"images": "CC-BY-4.0", "attribution": "Justin Johnson et al.; CLEVR"},
            provenance={"repository": CLEVR_REPO, "revision": CLEVR_REVISION,
                        "mirror": True, "original": "https://cs.stanford.edu/people/jcjohns/clevr/",
                        "annotation_sha256": file_sha256(metadata), "original_split": scene["split"]}))
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"CLEVR has only {len(records)} eligible images, requested {count}")
    print(f"CLEVR: {len(records)} complete annotated scenes", flush=True)
    return records


def read_zip_items(archive: Path, filename: str, prefix: str):
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(filename) as stream:
            yield from ijson.items(stream, prefix, use_float=True)


def coco_license(image: dict) -> dict:
    if image["license"] != 4:
        raise ValueError("Pilot accepts only COCO CC-BY-2.0 image license id 4")
    return {"images": "CC-BY-2.0", "url": "https://creativecommons.org/licenses/by/2.0/",
            "image_id": image["id"], "original_image_url": image.get("flickr_url"),
            "coco_license_id": 4}


def get_coco_image(root: Path, image: dict, split: str) -> Path:
    filename = image["file_name"]
    if not SAFE_NAME.fullmatch(filename):
        raise ValueError("Unsafe COCO image name")
    url = f"{COCO_BASE}/{split}/{filename}"
    return download(url, root / "images" / split / filename, 8_000_000)


def prepare_vsr(root: Path, archive: Path, count: int, rng: random.Random) -> list[dict]:
    revision_file = root / "sources/vsr_revision.json"
    if revision_file.is_file():
        revision = json.loads(revision_file.read_text())["revision"]
    else:
        response = requests.get("https://api.github.com/repos/cambridgeltl/visual-spatial-reasoning/commits/master",
                                timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
        atomic_json(revision_file, {"revision": revision})
    base = f"https://raw.githubusercontent.com/cambridgeltl/visual-spatial-reasoning/{revision}"
    metadata = download(f"{base}/data/data_files/all_vsr_validated_data.jsonl",
                        root / "sources/vsr.jsonl", 10_000_000)
    download(f"{base}/LICENSE", root / "sources/vsr_LICENSE", 100_000)
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line]
    positives = [r for r in rows if r["label"] == 1 and
                 len(ast.literal_eval(r["vote_true_validator_id"])) >= 3 and
                 len(ast.literal_eval(r["vote_false_validator_id"])) == 0 and
                 "/train2017/" in r["image_link"]]
    wanted = {int(Path(r["image"]).stem) for r in positives}
    images = {i["id"]: i for i in read_zip_items(archive, "annotations/instances_train2017.json", "images.item")
              if i["id"] in wanted and i["license"] == 4}
    eligible = [r for r in positives if int(Path(r["image"]).stem) in images]
    rng.shuffle(eligible)
    chosen, seen = [], set()
    for row in eligible:
        if row["image"] in seen:
            continue
        chosen.append(row)
        seen.add(row["image"])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Insufficient licensed VSR consensus positives")
    records = []
    provenance = {"repository": "cambridgeltl/visual-spatial-reasoning", "revision": revision,
                  "annotation_sha256": file_sha256(metadata), "consensus_positive_only": True}
    for index, row in enumerate(chosen):
        image = images[int(Path(row["image"]).stem)]
        path = get_coco_image(root, image, "train2017")
        records.append(image_record(root, path, id=f"vsr:positive:{index}:{row['image']}", source="vsr",
            caption=row["caption"], positive=True,
            constraints=[{"type": "spatial", "relation": row["relation"], "truth": True,
                          "validated_statement": row["caption"]}],
            license={**coco_license(image), "annotations": "Apache-2.0"}, provenance=provenance))
        negatives = [n for n in rows if n["image"] == row["image"] and n["label"] == 0 and
                     len(ast.literal_eval(n["vote_false_validator_id"])) >= 3]
        for negative_index, negative in enumerate(negatives[:1]):
            if negative["caption"].strip().lower() == row["caption"].strip().lower():
                continue
            records.append(image_record(root, path, id=f"vsr:negative:{index}:{negative_index}", source="vsr",
                caption=negative["caption"], positive=False,
                constraints=[{"type": "spatial", "relation": negative["relation"], "truth": False,
                              "validated_statement": negative["caption"]}],
                license={**coco_license(image), "annotations": "Apache-2.0"}, provenance=provenance))
    print(f"VSR: {len(chosen)} licensed consensus positives; {len(records)-len(chosen)} evaluation-only negatives", flush=True)
    return records


def prepare_coco(root: Path, archive: Path, count: int, rng: random.Random) -> list[dict]:
    filename = "annotations/instances_val2017.json"
    images = {i["id"]: i for i in read_zip_items(archive, filename, "images.item") if i["license"] == 4}
    categories = {c["id"]: c["name"] for c in read_zip_items(archive, filename, "categories.item")}
    annotations = defaultdict(lambda: defaultdict(list))
    for annotation in read_zip_items(archive, filename, "annotations.item"):
        if annotation["image_id"] in images:
            annotations[annotation["image_id"]][annotation["category_id"]].append(annotation)
    candidates = []
    targets = {"cat", "dog", "bird", "car", "bus", "bottle", "cup", "apple", "chair", "person", "horse"}
    for image_id, category_groups in annotations.items():
        image = images[image_id]
        for category, objects in category_groups.items():
            if (categories[category] in targets and 1 <= len(objects) <= 4 and
                    not any(a["iscrowd"] or a["area"] / (image["width"] * image["height"]) < .01
                            for a in objects)):
                candidates.append((image, categories[category], objects))
    rng.shuffle(candidates)
    archive_hash = file_sha256(archive)
    records, seen = [], set()
    for image, name, objects in candidates:
        if image["id"] in seen:
            continue
        path = get_coco_image(root, image, "val2017")
        n = len(objects)
        plural = "people" if name == "person" else ("buses" if name == "bus" else name + "s")
        caption = f"A photograph containing {n} {name if n == 1 else plural}."
        records.append(image_record(root, path, id=f"coco:{image['id']}:{name}", source="coco",
            caption=caption, positive=True,
            constraints=[{"type": "object_count", "class_name": name, "count": n,
                          "annotation_ids": [a["id"] for a in objects],
                          "bounding_boxes_xywh": [a["bbox"] for a in objects],
                          "scope": "noncrowd_instance_annotations"}],
            license={**coco_license(image), "annotations": "CC-BY-4.0"},
            provenance={"original": "https://cocodataset.org/#download", "original_split": "val2017",
                        "archive_sha256": archive_hash}))
        seen.add(image["id"])
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError("Insufficient licensed COCO count images")
    print(f"COCO: {len(records)} count examples with visible noncrowd instances", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/training/constraint_pilot")
    parser.add_argument("--per-source", type=int, default=36)
    parser.add_argument("--clevr-count", type=int, default=None,
                        help="Separate bounded CLEVR count; the pinned mirror has 131 eligible scenes")
    parser.add_argument("--reserve-from", action="append", default=None,
                        help="Prior records.jsonl whose image split assignments must be preserved")
    parser.add_argument("--reuse-from", default="data/training/constraint_pilot",
                        help="Reuse validated local pilot images and annotation archives")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 10 <= args.per_source <= 1000:
        parser.error("The bounded acquisition command supports 10-1000 examples per source")
    clevr_count = args.clevr_count if args.clevr_count is not None else args.per_source
    if not 10 <= clevr_count <= 131:
        parser.error("The pinned CLEVR mirror supports 10-131 eligible 3-4-object scenes")
    root = project_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "records.jsonl"
    reuse_root = project_path(args.reuse_from)
    default_reservation = project_path("data/training/constraint_pilot/records.jsonl")
    reservation_paths = ([project_path(p) for p in args.reserve_from] if args.reserve_from is not None
                         else [default_reservation] if default_reservation.is_file() and
                         default_reservation.resolve() != manifest_path.resolve() else [])
    reserved, reservation_audit = split_reservations(reservation_paths)
    if manifest_path.exists():
        _, audit = validate_records(manifest_path)
        saved_audit = json.loads((root / "audit.json").read_text())
        if saved_audit.get("dataset_sha256") != audit["dataset_sha256"]:
            raise ValueError("Dataset manifest changed after its acquisition audit")
        if (saved_audit["seed"] != args.seed or
                saved_audit["requested_positive_per_source"] != args.per_source or
                saved_audit.get("requested_clevr", args.per_source) != clevr_count or
                saved_audit.get("prior_split_manifests", []) != reservation_audit):
            raise ValueError("Existing dataset has a different acquisition profile; use a new output directory")
        print("Existing validated dataset:", audit, flush=True)
        return
    if reuse_root.resolve() != root.resolve() and (reuse_root / "records.jsonl").is_file():
        previous, _ = validate_records(reuse_root / "records.jsonl")
        # Copy rather than hard-link: changes to a new dataset cannot mutate the
        # archived pilot. Validation above checked every reused image's digest.
        copies = {row["image"]: reuse_root / row["image"] for row in previous}
        source_names = ("annotations_trainval2017.zip", "clevr_COPYRIGHT.txt", "clevr_LICENSE.txt",
                        "clevr_train_scenes.json", "vsr.jsonl", "vsr_LICENSE", "vsr_revision.json")
        copies.update({"sources/" + name: reuse_root / "sources" / name for name in source_names})
        for relative, source_path in copies.items():
            if not source_path.is_file():
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError("Dataset reuse destination escapes its output root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_name(destination.name + ".copying")
                shutil.copyfile(source_path, temporary)
                if file_sha256(temporary) != file_sha256(source_path):
                    raise ValueError("Local dataset cache copy changed")
                temporary.replace(destination)
    rng = random.Random(args.seed)
    records = prepare_clevr(root, clevr_count, rng)
    archive = download(COCO_ANNOTATIONS, root / "sources/annotations_trainval2017.zip", 350_000_000)
    print("COCO annotations downloaded; streaming selected metadata", flush=True)
    records += prepare_vsr(root, archive, args.per_source, rng)
    records += prepare_coco(root, archive, args.per_source, rng)
    records = assign_splits(records, args.seed, reserved_splits=reserved if reservation_paths else None)
    partial = root / "records.pending.jsonl"
    partial.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    _, audit = validate_records(partial)
    audit.update(seed=args.seed, requested_positive_per_source=args.per_source,
                 requested_clevr=clevr_count, prior_split_manifests=reservation_audit,
                 prior_reserved_hash_keys=len(reserved),
                 split_method="seeded_sha256_image_groups_80_10_10_with_prior_reservations",
                 preprocessing="full_image_letterbox_no_crop_no_flip",
                 training_truth_limit="COCO annotation exhaustiveness and VSR validator errors remain possible")
    atomic_json(root / "audit.json", audit)
    partial.replace(manifest_path)  # publish only after the complete audit exists
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
