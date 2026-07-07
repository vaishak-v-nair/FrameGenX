import yaml
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
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
    contract: Optional[Dict[str, Any]] = None
    scene_plan: Optional[Dict[str, Any]] = None
    verification_report: Optional[Dict[str, Any]] = None
    
    # State flags
    is_successful: bool = False
    repair_loops_executed: int = 0
    
    @property
    def run_dir(self) -> Path:
        base_dir = Path(self.config.get("paths", {}).get("output_dir", "runs"))
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
