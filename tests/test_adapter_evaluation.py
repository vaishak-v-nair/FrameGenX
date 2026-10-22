"""Resume cannot mix experiment identities or silently reuse corrupted outputs."""
from copy import deepcopy

from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from src.training.common import file_sha256
from src.training.evaluation import expected_images, validate_progress


@pytest.fixture
def progress(tmp_path):
    protocol = {"cases": [{"id": "case", "prompt": "Two bottles.",
                           "requirements": ["Two bottles are visible."]}], "seeds": [42]}
    expected = expected_images(protocol)
    latent = tmp_path / "case_42_baseline.safetensors"
    save_file({"latent": torch.ones((1, 4, 32, 32))}, str(latent))
    signature = {"adapter_sha256": "fixture", "code_sha256": "fixture"}
    saved = {"generation_signature": signature, "quality_status": "unverified",
             "promotion_status": "unqualified", "full_contract_CSR": None,
             "images": [{**expected[0], "latent_sha256": file_sha256(latent)}]}
    return tmp_path, saved, signature, expected


def test_resume_accepts_only_intact_partial_samples(progress):
    root, saved, signature, expected = progress
    items = validate_progress(root, saved, signature, expected, 256)
    assert set(items) == {"case_42_baseline.png"}
    assert not items["case_42_baseline.png"].get("image_sha256")


def test_resume_rejects_changed_adapter_or_code(progress):
    root, saved, signature, expected = progress
    with pytest.raises(ValueError, match="changed"):
        validate_progress(root, saved, {**signature, "adapter_sha256": "changed"}, expected, 256)


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "wrong_prompt", "quality_claim"])
def test_resume_rejects_misidentified_or_qualified_progress(progress, mutation):
    root, saved, signature, expected = progress
    if mutation == "duplicate":
        saved["images"].append(deepcopy(saved["images"][0]))
    elif mutation == "unexpected":
        saved["images"][0]["file"] = "../outside.png"
    elif mutation == "wrong_prompt":
        saved["images"][0]["prompt"] = "Three bottles."
    else:
        saved["full_contract_CSR"] = 1.0
    with pytest.raises(ValueError):
        validate_progress(root, saved, signature, expected, 256)


def test_resume_rejects_tampered_latent_and_image(progress):
    root, saved, signature, expected = progress
    image_file = root / saved["images"][0]["file"]
    image = Image.new("RGB", (256, 256), "white")
    image.putpixel((0, 0), (0, 0, 0))
    image.save(image_file)
    saved["images"][0]["image_sha256"] = file_sha256(image_file)
    image_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image is missing or changed"):
        validate_progress(root, saved, signature, expected, 256)
    saved["images"][0].pop("image_sha256")
    (root / "case_42_baseline.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="latent is missing or changed"):
        validate_progress(root, saved, signature, expected, 256)


@pytest.mark.parametrize("value", [torch.ones(1, 4, 16, 16), torch.full((1, 4, 32, 32), float("nan"))])
def test_resume_rejects_invalid_latent_shape_or_values(progress, value):
    root, saved, signature, expected = progress
    path = root / "case_42_baseline.safetensors"
    save_file({"latent": value}, str(path))
    saved["images"][0]["latent_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="shape or values"):
        validate_progress(root, saved, signature, expected, 256)
