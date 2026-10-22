import zipfile

import pytest

from src.research.ihalla import read_sheet
from src.datasets.ihalla import FolderImages, indexed_captions


def workbook(path, target="worksheets/history.xml"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", '''<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="history" r:id="second"/></sheets></workbook>''')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships><Relationship Id="second" Target="{target}"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>')
        archive.writestr("xl/worksheets/history.xml", '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="inlineStr"><is><t>index</t></is></c><c r="B1" t="inlineStr"><is><t>prompt</t></is></c></row><row><c r="A2"><v>1</v></c><c r="B2" t="inlineStr"><is><t>A historical map</t></is></c></row><row><c r="A3"/></row></sheetData></worksheet>''')


def test_named_worksheet_is_resolved_by_relationship_not_numeric_sheet_id(tmp_path):
    path = tmp_path / "captions.xlsx"
    workbook(path)
    assert read_sheet(path) == []
    assert indexed_captions(path, "history") == {1: "A historical map"}
    with pytest.raises(ValueError, match="exactly one worksheet"):
        read_sheet(path, sheet="science")


def test_worksheet_cannot_escape_the_expected_archive_directory(tmp_path):
    path = tmp_path / "captions.xlsx"
    workbook(path, "../../outside.xml")
    with pytest.raises(ValueError, match="escapes"):
        read_sheet(path, sheet="history")


def test_public_folder_listing_rejects_untrusted_names_and_conflicting_ids():
    parser = FolderImages()
    parser.feed('<div data-id="abcdefghijklmnop" data-tooltip="1.jpg Image"></div>')
    parser.feed('<div data-id="qrstuvwxyzabcdef" data-tooltip="../2.jpg Image"></div>')
    assert list(parser.images) == [1]
    with pytest.raises(ValueError, match="Conflicting"):
        parser.feed('<div data-id="qrstuvwxyzabcdef" data-tooltip="1.jpg Image"></div>')
