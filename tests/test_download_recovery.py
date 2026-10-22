"""Failed acquisition must not publish partial bytes or hide the actual error."""
from pathlib import Path

import pytest
import requests

from src.core import files
from src.training import prepare
from src.training.common import atomic_json


class Response:
    def __init__(self, chunks, declared=None):
        self.chunks = chunks
        self.headers = {} if declared is None else {"Content-Length": str(declared)}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, *args): yield from self.chunks


def test_transient_windows_replace_failure_retries_without_redownloading(tmp_path, monkeypatch):
    destination = tmp_path / "image.png"
    requests_made, attempts = [], []
    original = files.os.replace
    def replace(source, target):
        attempts.append(1)
        if len(attempts) <= 2:
            raise PermissionError("synthetic Windows sharing violation")
        original(source, target)
    monkeypatch.setattr(files.os, "replace", replace)
    monkeypatch.setattr(files.time, "sleep", lambda seconds: None)
    def get(*a, **kw):
        requests_made.append(1)
        return Response([b"complete-image"], 14)
    monkeypatch.setattr(prepare.requests, "get", get)
    assert prepare.download("https://example.invalid/image", destination, 100) == destination
    assert destination.read_bytes() == b"complete-image"
    assert len(requests_made) == 1 and len(attempts) == 3
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("response,error", [(Response([b"abc"], 4), requests.RequestException),
                                           (Response([b"x" * 11]), ValueError),
                                           (Response([]), ValueError)])
def test_truncated_oversized_and_empty_download_never_becomes_a_cached_success(tmp_path, monkeypatch, response, error):
    destination = tmp_path / "data.bin"
    monkeypatch.setattr(prepare.requests, "get", lambda *a, **kw: response)
    with pytest.raises(error):
        prepare.download("https://example.invalid/data", destination, 10, retries=1)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_json_publication_preserves_old_value_if_replacement_stays_locked(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "old"})
    def locked(*args): raise PermissionError("synthetic persistent sharing violation")
    monkeypatch.setattr(files.os, "replace", locked)
    monkeypatch.setattr(files.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        atomic_json(path, {"status": "new"})
    assert '"old"' in path.read_text()
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_json_never_replaces_previous_success(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "old"})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert '"old"' in path.read_text()
