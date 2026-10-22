"""Download a bounded public I-HallA evaluation bundle without running agents.

Source normal/weird labels and QA answers are unreviewed benchmark annotations,
not positive diffusion targets or calibrated human judgments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil

from PIL import Image

from src.research.ihalla import read_sheet
from src.training.common import atomic_json, file_sha256, project_path
from src.training.prepare import download
from src.core.files import directory_lock


REVISION = "e750fd025b1473819234d07ee2d467bb05b3cd9c"
SHEETS = {
    "captions": "1QdIse3GloFp7HkwoFmK_L3zHZkODoE11N1jIt6O_NAc",
    "qa_history": "146LXyXh8ivRz_psXU1mY3iMUgMZIh1hadOhnkiM_jUE",
    "qa_science": "1H7DSbVBOzzKMsUoI_u-64bSZiR510Ags",
    "reasoning_history": "1HS52GVFosc9NaSgINgupSPo4OkaOQiRr",
    "reasoning_science": "1Zz_13Fqe7jAadUCcFAg5dtPMm9xxgd99",
}
# Resolved from the public models/ folder linked by the pinned upstream README.
FOLDERS = {
    ("history", "normal"): "1aPCG_Go9mT2Ndx8dnVYxo9EjegLkbhI1",
    ("history", "weird"): "1ZzC1XOOCG-Dk4B2oSXPsTOjJ32BCClkC",
    ("science", "normal"): "16llafW6DiU742ubrAlIICh2H81EtZbst",
    ("science", "weird"): "1FM4lLgTp1cCYIaN5IBz3YjdUhEmKBzBy",
}
PUBLIC_ID = re.compile(r"[A-Za-z0-9_-]{15,100}\Z")
IMAGE_NAME = re.compile(r"([1-9][0-9]{0,3})\.(jpe?g|png) Image\Z", re.I)


class FolderImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        match = IMAGE_NAME.fullmatch(attrs.get("data-tooltip", ""))
        identifier = attrs.get("data-id", "")
        if match and PUBLIC_ID.fullmatch(identifier):
            index = int(match[1])
            item = {"index": index, "file_id": identifier, "extension": match[2].lower()}
            if index in self.images and self.images[index] != item:
                raise ValueError("Conflicting I-HallA image identities in public folder")
            self.images[index] = item


def indexed_captions(path: Path, domain: str) -> dict[int, str]:
    captions = {}
    for row in read_sheet(path, sheet=domain)[1:]:
        if not row.get("B", "").strip():
            continue
        number = float(row["A"])
        if not number.is_integer() or number < 1 or int(number) in captions:
            raise ValueError("Invalid or duplicate domain caption identity")
        captions[int(number)] = row["B"].strip()
    return captions


def indexed_qa(path: Path) -> dict[int, list[dict]]:
    questions = {}
    for row in read_sheet(path)[1:]:
        if not row.get("C", "").strip():
            continue
        number = float(row["A"])
        if not number.is_integer() or number < 1 or any(not row.get(c, "").strip() for c in "BCDE"):
            raise ValueError("Incomplete I-HallA QA annotation")
        questions.setdefault(int(number), []).append({
            "composition": row["B"], "question": row["C"],
            "choices": row["D"], "source_answer": row["E"],
        })
    return questions


def validate_bundle(output: Path, manifest: dict, per_group: int):
    if (manifest.get("source_revision") != REVISION or manifest.get("per_group") != per_group or
            manifest.get("role") != "held_out_evaluation_only" or
            manifest.get("training_allowed_by_this_bundle") is not False or
            manifest.get("quality_status") != "unverified" or manifest.get("evaluation_score") is not None):
        raise ValueError("I-HallA bundle profile changed; use another output directory")
    for entry in manifest["files"]:
        path = (output / entry["file"]).resolve()
        if (not path.is_relative_to(output.resolve()) or not path.is_file() or
                path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]):
            raise ValueError("I-HallA acquisition evidence is missing, changed or outside its root")
    if manifest["status"] == "acquired_pending_human_review_and_vqa":
        expected = {f"ihalla:dalle-3:{domain}:{variant}:{i}"
                    for domain, variant in FOLDERS for i in range(1, per_group + 1)}
        if ({row["id"] for row in manifest["records"]} != expected or
                len(manifest["records"]) != len(expected)):
            raise ValueError("I-HallA completed bundle has missing or duplicate image cases")
        known_files = {entry["file"]: entry["sha256"] for entry in manifest["files"]}
        for row in manifest["records"]:
            if (known_files.get(row["file"]) != row["image_sha256"] or row["split"] != "test"
                    or not row["questions"] or not row["caption"]):
                raise ValueError("I-HallA case is not bound to intact held-out image evidence")


def acquire(output: Path, per_group=10) -> dict:
    with directory_lock(output):
        existing = output / "manifest.json"
        if existing.is_file():
            manifest = json.loads(existing.read_text(encoding="utf-8"))
            validate_bundle(output, manifest, per_group)
            return manifest
        progress = output / "acquisition.json"
        if progress.is_file():
            validate_bundle(output, json.loads(progress.read_text(encoding="utf-8")), per_group)
        return _acquire(output, per_group)


def _acquire(output: Path, per_group=10) -> dict:
    if not 1 <= per_group <= 25:
        raise ValueError("I-HallA acquisition supports 1-25 images per domain/variant")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source": "kaist-cvml/I-HallA-v1.0",
                "source_revision": REVISION, "role": "held_out_evaluation_only",
                "training_allowed_by_this_bundle": False, "status": "acquiring",
                "quality_status": "unverified", "evaluation_score": None,
                "source_image_rights": "not established by repository MIT code license",
                "per_group": per_group, "files": [], "records": []}
    atomic_json(output / "acquisition.json", manifest)

    def remember(path, url, **extra):
        manifest["files"].append({"file": path.relative_to(output).as_posix(), "url": url,
                                  "bytes": path.stat().st_size, "sha256": file_sha256(path), **extra})

    try:
        prior = project_path("reports/research_sources/ihalla_vsm_2026-10-16/ihalla")
        for name, identifier in SHEETS.items():
            url = f"https://docs.google.com/spreadsheets/d/{identifier}/export?format=xlsx"
            path = output / "metadata" / f"{name}.xlsx"
            if name in {"captions", "qa_history"} and not path.exists() and (prior / path.name).is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prior / path.name, path)
            download(url, path, 5_000_000)
            rows = read_sheet(path)
            remember(path, url, rows_including_blank_and_header=len(rows))
        caption_maps = {d: indexed_captions(output / "metadata/captions.xlsx", d)
                        for d in ("history", "science")}
        qa_maps = {d: indexed_qa(output / f"metadata/qa_{d}.xlsx") for d in caption_maps}
        manifest["metadata_counts"] = {d: {"captions": len(caption_maps[d]),
            "qa_cases": len(qa_maps[d]), "questions": sum(map(len, qa_maps[d].values()))}
            for d in caption_maps}
        planned = []
        for (domain, variant), identifier in FOLDERS.items():
            url = f"https://drive.google.com/drive/folders/{identifier}"
            path = download(url, output / f"metadata/folder_{domain}_{variant}.html", 2_000_000)
            parser = FolderImages()
            parser.feed(path.read_text(encoding="utf-8"))
            required = list(range(1, per_group + 1))
            if any(i not in parser.images or i not in caption_maps[domain] or i not in qa_maps[domain]
                   for i in required):
                raise ValueError("Public listing or domain annotations do not cover the planned image prefix")
            remember(path, url, listing_scope="bounded public HTML; not a full Drive inventory")
            for index in required:
                planned.append((domain, variant, parser.images[index]))

        def acquire_image(plan):
            domain, variant, item = plan
            url = f"https://drive.google.com/uc?export=download&id={item['file_id']}"
            relative = f"images/dalle-3/{domain}/{variant}/{item['index']}.{item['extension']}"
            path = download(url, output / relative, 10_000_000)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG"} or max(image.size) > 8192:
                    raise ValueError("Unsupported or oversized I-HallA image")
                width, height = image.size
                image.verify()
            return path, url, {"id": f"ihalla:dalle-3:{domain}:{variant}:{item['index']}",
                "domain": domain, "source_model": "dalle-3", "source_variant": variant,
                "data_index": item["index"], "file": relative,
                "image_sha256": file_sha256(path), "width": width, "height": height,
                "caption": caption_maps[domain][item["index"]],
                "questions": qa_maps[domain][item["index"]],
                "truth_status": "source_annotations_not_human_reviewed", "split": "test"}

        with ThreadPoolExecutor(max_workers=3) as pool:
            for path, url, record in pool.map(acquire_image, planned):
                remember(path, url)
                manifest["records"].append(record)
                print(f"I-HallA: {len(manifest['records'])}/{len(planned)} verified image downloads", flush=True)
                atomic_json(output / "acquisition.json", manifest)
        manifest["status"] = "acquired_pending_human_review_and_vqa"
        manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
        atomic_json(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error_type=type(error).__name__)
        raise
    finally:
        atomic_json(output / "acquisition.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/evaluation/ihalla_v1_prefix")
    parser.add_argument("--per-group", type=int, default=10)
    args = parser.parse_args()
    result = acquire(project_path(args.output), args.per_group)
    print(json.dumps({key: result[key] for key in ("status", "bytes", "metadata_counts")}), flush=True)


if __name__ == "__main__":
    main()
