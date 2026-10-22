import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict
import yaml

from src.core.context import PROJECT_ROOT
from src.core.files import atomic_replace, remove_temporary


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    method: Literal["lora", "qlora"] = "qlora"
    dataset: str = "data/training/constraint_pilot/records.jsonl"
    output_root: str = "runs/training"
    tracking_db: str = "runs/mlflow.db"
    experiment: str = "framegenx-constraint-adapters"
    resolution: Literal[256, 512] = 512
    rank: int = Field(default=4, ge=1, le=32)
    lora_alpha: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-5, gt=0, le=1e-3)
    max_steps: int = Field(default=48, ge=1, le=10000)
    gradient_accumulation: int = Field(default=2, ge=1, le=32)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    cpu_threads: int = Field(default=4, ge=1, le=4)
    checkpoint_every: int = Field(default=16, ge=1)
    validation_every: int = Field(default=16, ge=1)
    validation_limit: int = Field(default=8, ge=1, le=100)
    max_vram_gb: float = Field(default=5.5, gt=0, le=6)
    min_free_vram_gb: float = Field(default=3.0, gt=0, le=6)
    max_runtime_seconds: int = Field(default=3600, ge=1, le=86400)
    max_total_runtime_seconds: int = Field(default=5400, ge=1, le=86400)
    max_process_rss_gb: float = Field(default=7.0, gt=0, le=10)
    min_available_ram_gb: float = Field(default=1.0, gt=0, le=4)
    snr_gamma: float = Field(default=5.0, gt=0, le=100)
    target_modules: list[str] = Field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"], min_length=1)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(name, path)
    finally:
        remove_temporary(name)


def load_training_config(path: str | Path) -> TrainingConfig:
    with project_path(path).open(encoding="utf-8") as stream:
        return TrainingConfig.model_validate(yaml.safe_load(stream))
