"""Separate CPU Cards research contract; never accepted as an SDXL adapter."""
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.research.artifacts import workspace_path


class CardsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["framegenx-cards-config-v1"] = Field(alias="schema")
    device: Literal["cpu"] = "cpu"
    threads: int = Field(default=4, ge=1, le=4)
    dataset_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    cards_archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    templates_archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_root: str
    output_root: str = "runs/research_cards"
    max_records: int = Field(default=4096, ge=16, le=4096)
    resolution: Literal[128] = 128
    width: Literal[16] = 16
    batch_size: int = Field(default=8, ge=1, le=8)
    steps: int = Field(default=500, ge=1, le=500)
    diffusion_steps: Literal[100] = 100
    learning_rate: float = Field(default=.001, gt=0, le=.001)
    seed: int = Field(default=1729, ge=0, le=2**31-1)
    candidate_rho: float = Field(default=.1, gt=0, le=1)
    variance_mapping: Literal["tanh_learned_range"] = "tanh_learned_range"
    vsm_schedule: Literal["inverse_sqrt_noise"] = "inverse_sqrt_noise"
    ncc_threshold: float = Field(default=.95, gt=0, le=1)
    ambiguity_margin: float = Field(default=.01, ge=0, lt=1)
    corruptions: list[Literal["noise", "missing_tile", "pixel_shuffle", "color_change"]]
    sample_seeds: list[int] = Field(min_length=2, max_length=32)
    bootstrap_seed: int = Field(default=2718, ge=0)
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)
    max_seconds_per_arm: int = Field(default=1800, ge=1, le=1800)
    max_total_seconds: int = Field(default=5400, ge=1, le=5400)
    quality_status: Literal["unverified"] = "unverified"
    study_scope: Literal["cards_cpu_feasibility_only"] = "cards_cpu_feasibility_only"

    @model_validator(mode="after")
    def validate_protocol(self):
        for value in (self.data_root, self.output_root):
            workspace_path(value)
        if len(set(self.sample_seeds)) != len(self.sample_seeds):
            raise ValueError("Sampling seeds must be unique")
        if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in self.sample_seeds):
            raise ValueError("Invalid sampling seed")
        if len(set(self.corruptions)) != 4:
            raise ValueError("Freeze all four distinct corruption families")
        return self


def load_config(path="configs/research_cards.json") -> CardsConfig:
    return CardsConfig.model_validate(json.loads(workspace_path(path).read_text(encoding="utf-8")))
