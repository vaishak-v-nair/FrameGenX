"""Bounded XLSX metadata inspection; benchmark facts are never training labels.

Does not evaluate spreadsheet formulas or call providers. External image rights
must be established separately from the repository's MIT code license.
"""
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from src.research.artifacts import sha256

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_sheet(path: Path, max_rows=5000, sheet: str | None = None):
    if path.stat().st_size > 5_000_000:
        raise ValueError("I-HallA metadata file exceeds budget")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 20_000_000:
            raise ValueError("I-HallA workbook expansion exceeds budget")
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(item.itertext()) for item in root.findall("s:si", NS)]
        target = "xl/worksheets/sheet1.xml"
        if sheet is not None:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            matching = [s for s in workbook.findall("s:sheets/s:sheet", NS)
                        if s.attrib.get("name") == sheet]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one worksheet named {sheet}")
            relation = matching[0].attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            links = [r for r in relationships if r.attrib.get("Id") == relation]
            if len(links) != 1 or links[0].attrib.get("TargetMode") == "External":
                raise ValueError("Invalid or external worksheet relationship")
            target = posixpath.normpath(posixpath.join("xl", links[0].attrib["Target"]))
            if not target.startswith("xl/worksheets/") or not target.endswith(".xml"):
                raise ValueError("Worksheet relationship escapes its expected archive directory")
        root = ET.fromstring(archive.read(target))
        rows = []
        for row in root.findall("s:sheetData/s:row", NS):
            if len(rows) >= max_rows:
                raise ValueError("I-HallA metadata row budget exceeded")
            values = {}
            for cell in row.findall("s:c", NS):
                if cell.find("s:f", NS) is not None:
                    raise ValueError("Formula cells are not benchmark annotations")
                value = cell.find("s:v", NS)
                text = value.text if value is not None else "".join(cell.itertext())
                if cell.attrib.get("t") == "s":
                    text = strings[int(text)]
                values[cell.attrib["r"].rstrip("0123456789")] = text or ""
            rows.append(values)
    return rows


def inspect_ihalla(root: Path):
    files = []
    for name in ("captions.xlsx", "qa_history.xlsx"):
        path = root / name
        if not path.is_file():
            files.append({"name": name, "status": "missing"})
            continue
        rows = read_sheet(path)
        files.append({"name": name, "status": "metadata_available", "sha256": sha256(path),
                      "rows_including_header": len(rows), "columns": list(rows[0]) if rows else []})
    return {"source": "kaist-cvml/I-HallA-v1.0", "revision": "e750fd025b1473819234d07ee2d467bb05b3cd9c",
            "role": "held_out_factual_qa_evaluation", "metadata": files,
            "training_status": "blocked_data_rights_and_disjoint_training_partition",
            "evaluation_status": "unavailable_images_and_permitted_vqa_not_configured",
            "score": None, "quality_status": "unverified"}
