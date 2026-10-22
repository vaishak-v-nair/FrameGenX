"""Scientific, data-integrity and isolation gates before any real research fit."""
from io import BytesIO
import json
import os
import tarfile
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from src.research.artifacts import atomic_json, experiment_lock, json_digest, workspace_path
from src.research.cards import CardTemplates, EXPECTED_TEMPLATES, extract_pngs, family_split
from src.research.config import load_config, CardsConfig
from src.research.diffusion import LearnedVarianceDiffusion
from src.research.experiment import corrupt_image, paired_summary, wilson
from src.research.ihalla import read_sheet, inspect_ihalla


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        raise AssertionError("Research CPU tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    torch.set_num_threads(2)


def test_cpu_config_rejects_gpu_and_unknown_training_options():
    values = load_config().model_dump(by_alias=True)
    for changes in ({"device": "cuda"}, {"steps": 10000}, {"resolution": 512},
                    {"quality_status": "qualified"}, {"rank": 4},
                    {"sample_seeds": [10000, 10000]}, {"data_root": "../private"}):
        with pytest.raises(ValueError):
            CardsConfig.model_validate({**values, **changes})


def test_atomic_json_rejects_nonfinite_and_keeps_previous(tmp_path):
    path = tmp_path / "report.json"
    atomic_json(path, {"value": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_workspace_escape_is_rejected():
    with pytest.raises(ValueError):
        workspace_path("../../outside")


def test_cpu_experiment_lock_excludes_a_second_owner(tmp_path):
    with experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already owns"):
            with experiment_lock(tmp_path):
                pytest.fail("Second owner must not enter")
    with experiment_lock(tmp_path):
        pass


@pytest.mark.parametrize("member_name,link", [("../evil.png", False), ("/evil.png", False),
    ("card_imgs/x.png", True), ("card_imgs\\x.png", False), ("C:/evil.png", False)])
def test_archive_rejects_traversal_links_and_windows_paths(tmp_path, member_name, link):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(member_name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../target"
        archive.addfile(info)
    with pytest.raises(ValueError):
        extract_pngs(path, tmp_path / "output", limit=10)
    assert not (tmp_path / "evil.png").exists()


def test_extraction_does_not_execute_nonimage_members(tmp_path):
    raw = BytesIO()
    Image.new("RGB", (128, 128), "white").save(raw, format="PNG")
    path = tmp_path / "data.tar"
    with tarfile.open(path, "w") as archive:
        for name, data in (("card_imgs/grid_1.png", raw.getvalue()), ("untrusted.py", b"raise RuntimeError()")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    files, ignored = extract_pngs(path, tmp_path / "output", limit=10)
    assert len(files) == 1 and ignored == ["untrusted.py"]
    assert not (tmp_path / "output/untrusted.py").exists()


def test_extraction_cannot_write_through_a_destination_directory_link(tmp_path):
    outside = tmp_path / "other-directory"
    outside.mkdir()
    destination = tmp_path / "linked-output"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(outside), str(destination))
    else:
        destination.symlink_to(outside, target_is_directory=True)
    raw = BytesIO()
    Image.new("RGB", (128, 128), "white").save(raw, format="PNG")
    archive_path = tmp_path / "data.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("card_imgs/grid_1.png")
        info.size = len(raw.getvalue())
        archive.addfile(info, BytesIO(raw.getvalue()))
    with pytest.raises(ValueError, match="directory link"):
        extract_pngs(archive_path, destination, limit=1)
    assert list(outside.iterdir()) == []


def test_rgb_checker_rejects_wrong_color_and_blank_grids(tmp_path):
    rng = np.random.default_rng(17)
    originals = []
    for name in sorted(EXPECTED_TEMPLATES):
        tile = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(tile).save(tmp_path / f"{name}.png")
        originals.append(tile)
    checker = CardTemplates(tmp_path)
    grid = np.concatenate([np.concatenate(originals[:2], 1), np.concatenate(originals[2:4], 1)], 0)
    assert checker.inspect(grid)["status"] == "valid"
    altered = grid.copy()
    altered[:64, :64, 0] = 0
    altered[:64, :64, 1] = 100
    altered[:64, :64, 2] = 230
    assert checker.inspect(altered)["status"] == "invalid"
    assert checker.inspect(np.full_like(grid, 255))["status"] == "invalid"


def test_all_forty_templates_are_required(tmp_path):
    assert len(EXPECTED_TEMPLATES) == 40
    with pytest.raises(ValueError, match="40-template"):
        CardTemplates(tmp_path)


def test_split_is_stable_for_every_variant_of_a_family():
    assert len({family_split("H1|C2|S3|D4") for _ in range(20)}) == 1
    assert set(family_split(f"family-{i}") for i in range(1000)) == {"train", "validation", "test"}


def test_rho_zero_equivalence_and_variance_gradient_routing():
    diffusion = LearnedVarianceDiffusion(100)
    generator = torch.Generator().manual_seed(2)
    clean = torch.rand((2, 3, 8, 8), generator=generator)*2-1
    noise = torch.randn(clean.shape, generator=generator)
    timestep = torch.tensor([0, 60])
    noisy = diffusion.noisy(clean, timestep, noise)
    epsilon = torch.zeros_like(clean, requires_grad=True)
    variance = torch.zeros_like(clean, requires_grad=True)
    output = torch.cat((epsilon, variance), 1)
    terms = diffusion.loss(output, clean, noisy, noise, timestep, 0.)
    assert terms["loss"].item() == pytest.approx((terms["mse"]+terms["vlb"]).item())
    assert terms["smooth"].item() == 0
    terms["vlb"].backward(retain_graph=True)
    assert torch.count_nonzero(epsilon.grad) == 0
    assert torch.isfinite(variance.grad).all() and torch.count_nonzero(variance.grad)
    epsilon.grad = None
    variance.grad = None
    regularized = diffusion.loss(output, clean, noisy, noise, timestep, .1)
    assert regularized["loss"] > terms["loss"]
    regularized["smooth"].backward()
    assert torch.count_nonzero(variance.grad)
    assert torch.count_nonzero(epsilon.grad) == 0


def test_variance_mapping_is_bounded_and_regularizer_is_finite():
    diffusion = LearnedVarianceDiffusion(100)
    reference = torch.zeros((2, 3, 8, 8))
    output = torch.cat((reference, torch.stack([torch.full((3,8,8), -100.), torch.full((3,8,8),100.)])), 1)
    timestep = torch.tensor([1, 80])
    _, logvar = diffusion.distribution(output, reference, timestep)
    assert torch.all(logvar >= diffusion.take(diffusion.posterior_log, timestep, reference)-1e-6)
    assert torch.all(logvar <= diffusion.take(diffusion.beta.log(), timestep, reference)+1e-6)
    assert torch.isfinite(torch.sigmoid(2*logvar)).all()
    with pytest.raises(ValueError, match="separate epsilon"):
        diffusion.distribution(reference, reference, timestep)


def test_nonfinite_diffusion_cannot_be_reported_as_training_success():
    diffusion = LearnedVarianceDiffusion(100)
    image = torch.zeros((1, 3, 8, 8))
    output = torch.full((1, 6, 8, 8), float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        diffusion.loss(output, image, image, image, torch.tensor([1]), .1)


def test_missing_or_duplicate_samples_never_shrink_denominator():
    config = load_config()
    with pytest.raises(ValueError, match="Missing, duplicate"):
        paired_summary({"baseline": [], "candidate": []}, config)
    entries = [{"seed": seed, "status": "invalid", "sha256": str(seed), "nearest_train_16px_rmse": .2}
               for seed in config.sample_seeds]
    report = paired_summary({"baseline": entries, "candidate": entries}, config)
    assert report["arms"]["baseline"]["n"] == 32
    assert report["arms"]["baseline"]["counts"]["invalid"] == 32
    assert report["full_contract_CSR"] is None and report["reduction_claim"] is False
    assert report["paired_valid_rate_delta"] == 0
    assert wilson(0, 0) is None
    assert wilson(0, 32)[1] > 0


def test_corruptions_are_reproducible_and_not_positive_training_data():
    pixels = np.arange(128*128*3, dtype=np.uint8).reshape(128,128,3)
    for kind in load_config().corruptions:
        assert np.array_equal(corrupt_image(pixels, kind, 42), corrupt_image(pixels, kind, 42))
        assert not np.array_equal(corrupt_image(pixels, kind, 42), pixels)


def test_ihalla_missing_assets_remain_unavailable_not_a_score(tmp_path):
    report = inspect_ihalla(tmp_path)
    assert report["score"] is None and report["quality_status"] == "unverified"
    assert report["training_status"].startswith("blocked")
    assert all(item["status"] == "missing" for item in report["metadata"])


def test_ihalla_formula_is_not_executed_or_used_as_fact(tmp_path):
    path = tmp_path / "metadata.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>')
    with pytest.raises(ValueError, match="Formula"):
        read_sheet(path)


def test_protocol_digest_preserves_meaningful_changes():
    original = load_config().model_dump(by_alias=True)
    assert json_digest(original) == json_digest(dict(reversed(list(original.items()))))
    assert json_digest(original) != json_digest({**original, "candidate_rho": .2})
