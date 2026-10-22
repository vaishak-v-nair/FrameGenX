import yaml
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    actual_path = PROJECT_ROOT / config_path
    if not actual_path.exists():
        return {}
    with open(actual_path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class RunContext:
    """
    Manages state, intermediate artifacts, and logging for a single generation run.
    Avoids arbitrary disk I/O by formalizing where and how artifacts are stored.
    """

    prompt: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=load_config)

    # In-memory representations
    # (field -> writer stage -> reader stages; see design doc RunContext contract)
    contract: Optional[Dict[str, Any]] = None            # intent/retrieval -> plan, verify, reflect, DB
    scene_plan: Optional[Dict[str, Any]] = None          # plan -> generate
    verification_report: Optional[Dict[str, Any]] = None  # verify -> reflect, DB

    # Stage-graph fields
    config_name: str = "interactive"                     # benchmark runs override this label
    is_benchmark: bool = False                           # explicit provenance for dashboard metrics
    benchmark_seed: Optional[int] = None
    cache_key: Optional[str] = None                 # current code/config provenance for interactive reuse
    attempt: int = 0                                     # current repair-loop attempt
    final_prompt: str = ""                               # plan -> generate
    gen_config: Dict[str, Any] = field(default_factory=dict)      # plan -> generate
    control_hints: Dict[str, Any] = field(default_factory=dict)   # plan -> generate
    masks: Dict[str, Any] = field(default_factory=dict)           # plan -> generate
    generation_meta: Dict[str, Any] = field(default_factory=dict)  # generate -> DB
    ref_image_paths: list = field(default_factory=list)  # retrieval -> (inspection artifact only)
    reward_history: list = field(default_factory=list)   # reflect -> Pipeline, DB
    prompt_edits: list = field(default_factory=list)     # reflect -> plan (next attempt)
    decision: Dict[str, Any] = field(default_factory=dict)  # reflect -> Pipeline

    # State flags
    is_successful: bool = False
    repair_loops_executed: int = 0

    def metrics(self) -> Dict[str, Any]:
        """Final metrics blob persisted to the DB metrics_json column."""
        last = self.reward_history[-1] if self.reward_history else {}
        return {
            "clip_norm": last.get("clip_norm"),
            "vqa_score": last.get("vqa_score"),
            "reward": last.get("reward"),
            "attempts": self.attempt + 1,
            "latency_s": round(self.get_duration(), 2),
        }

    @property
    def run_dir(self) -> Path:
        base_dir_str = self.config.get("paths", {}).get("output_dir", "runs")
        base_dir = PROJECT_ROOT / base_dir_str
        run_path = base_dir / self.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def save_artifact(self, name: str, data: Any, is_json: bool = True):
        """Saves an intermediate artifact to the run directory."""
        import json

        file_path = self.run_dir / name
        if is_json:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Assume it's a PIL Image or similar that has a save() method
            # Or binary data
            if hasattr(data, "save"):
                data.save(file_path)
            elif isinstance(data, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(data)
            else:
                with open(file_path, "wb") as f:
                    f.write(data)

    def get_duration(self) -> float:
        return time.time() - self.start_time
