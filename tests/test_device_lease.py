"""Exercise OS ownership without querying CUDA or loading a model."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from src.core import device


@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    path = tmp_path / "device.lock"
    monkeypatch.setattr(device, "_lock_path", lambda: path)
    return path


def test_nested_owner_and_exception_release(lock_path):
    with pytest.raises(KeyboardInterrupt):
        with device.device_lease():
            assert device.owns_device()
            with device.device_lease():
                assert device.owns_device()
            assert device.owns_device()
            raise KeyboardInterrupt()
    assert not device.owns_device()
    with device.device_lease():
        assert device.owns_device()


def test_another_thread_cannot_inherit_ownership(lock_path):
    def contender():
        with device.device_lease():
            pytest.fail("Another thread cannot share a device owner")
    with device.device_lease(), ThreadPoolExecutor(1) as executor:
        with pytest.raises(device.DeviceBusyError):
            executor.submit(contender).result(timeout=5)


def test_brief_os_lock_release_delay_recovers_before_any_device_work(lock_path, monkeypatch):
    import errno
    original = device._os_lock
    attempts = []
    def delayed(stream):
        attempts.append(1)
        if len(attempts) <= 2:
            raise OSError(errno.EACCES, "synthetic delayed OS lock cleanup")
        return original(stream)
    monkeypatch.setattr(device, "_os_lock", delayed)
    with device.device_lease():
        assert device.owns_device()
    assert len(attempts) == 3


CHILD = """
import sys
from pathlib import Path
from src.core import device
device._lock_path = lambda: Path(sys.argv[1])
try:
    with device.device_lease():
        if len(sys.argv) > 2:
            Path(sys.argv[2]).write_text('ready')
            sys.stdin.read()
except device.DeviceBusyError:
    sys.exit(73)
"""


def test_separate_process_is_excluded_and_abrupt_exit_releases_lock(lock_path, tmp_path):
    ready = tmp_path / "ready"
    owner = subprocess.Popen([sys.executable, "-B", "-c", CHILD, str(lock_path), str(ready)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=Path(__file__).resolve().parents[1])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "Child owner failed to acquire its lock before the deadline"
        with pytest.raises(device.DeviceBusyError):
            with device.device_lease():
                pytest.fail("The parent's separate process must not acquire the lease")
        contender = subprocess.run([sys.executable, "-B", "-c", CHILD, str(lock_path)],
                                   capture_output=True, timeout=10,
                                   cwd=Path(__file__).resolve().parents[1])
        assert contender.returncode == 73, contender.stderr.decode(errors="replace")
        owner.kill()
        owner.wait(timeout=10)
        with device.device_lease():
            assert device.owns_device()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)


def test_construction_is_cuda_free_and_unguarded_load_fails_before_probe(monkeypatch):
    import torch
    from src.generation.engine import GenerationEngine
    from src.generation.refine_stage import RefineStage
    from src.verification.qa import QAVerifier
    from src.verification.verify_stage import CLIPScoreVerifier
    from src.verification.instance_color import InstanceColorVerifier
    from src.training.model import load_trainable_unet
    from src.training.common import TrainingConfig

    def forbidden():
        pytest.fail("Construction or an unowned call queried CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    engine, qa, clip = GenerationEngine({}), QAVerifier(), CLIPScoreVerifier({})
    instances, refine = InstanceColorVerifier(), RefineStage({})
    for action in (engine.ensure_loaded, qa._load_models, clip._load, instances._load,
                   refine._build_pipe, lambda: load_trainable_unet(TrainingConfig())):
        with pytest.raises(device.DeviceLeaseRequired):
            action()


def test_pipeline_cleans_resident_models_before_releasing_ownership(lock_path, monkeypatch):
    from src.core.stage import Pipeline
    seen = []
    def teardown():
        seen.append(device.owns_device())
    pipeline = object.__new__(Pipeline)
    pipeline.stages = {"generate": SimpleNamespace(engine=SimpleNamespace(teardown=teardown))}
    with pytest.raises(ValueError, match="required stage"):
        pipeline.run("a cat")
    assert seen == [True]
    assert not device.owns_device()


def test_api_contends_with_non_api_owner_and_never_starts_generation(lock_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src import api
    monkeypatch.setattr(api.pipeline.db, "get_run_by_prompt", lambda *a, **k: None)
    monkeypatch.setattr(api.pipeline, "run", lambda *a, **k: pytest.fail("Busy device must not generate"))
    with device.device_lease(), TestClient(api.app) as client:
        for route, payload in (("/jobs", {"prompt": "a cat", "force": True}),
                               ("/generate", {"prompt": "a cat"})):
            response = client.post(route, json=payload)
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "device_busy"
    assert not api.GPU_BUSY.locked()


def test_worker_start_failure_releases_admission_lock(lock_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src import api
    monkeypatch.setattr(api.pipeline.db, "get_run_by_prompt", lambda *a, **k: None)
    def broken_start(self):
        raise RuntimeError("synthetic worker creation failure")
    monkeypatch.setattr(api.threading.Thread, "start", broken_start)
    # Call the synchronous route directly; TestClient itself needs a thread.
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as failure:
        api.create_job(api.JobRequest(prompt="a cat", force=True))
    assert failure.value.status_code == 503
    assert not api.GPU_BUSY.locked()


def test_cleanup_failure_keeps_device_unavailable_until_process_exit(lock_path):
    code = """
import sys
from pathlib import Path
from src.core import device
device._lock_path = lambda: Path(sys.argv[1])
class Failing:
    def cleanup(self): raise RuntimeError('cleanup failed')
    @device.exclusive_device(cleanup='cleanup')
    def run(self): return 1
try: Failing().run()
except RuntimeError: pass
try:
    with device.device_lease(): pass
except device.DeviceBusyError: sys.exit(0)
sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(lock_path)],
                            capture_output=True, timeout=10,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with device.device_lease():
        assert device.owns_device()
