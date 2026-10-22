from pathlib import Path

from src.core.context import RunContext, load_config
from src.core.db import RunTrackerDB
from src.core.stage import Pipeline
from src.reporting.analyzer import Reporter


class FrameGenXPipeline:
    """
    Thin facade over the config-driven stage Pipeline.
    Preserves the public contract used by src/api.py, src/cli.py and
    benchmark.py: .config, .db, .run(prompt) -> RunContext.
    """

    def __init__(self, config_path: str = "config.yaml", config_overrides: dict = None):
        self.config = load_config(config_path)
        if config_overrides:
            self.config = _deep_merge(self.config, config_overrides)
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        db_path_str = self.config.get("paths", {}).get("db_path", "runs/framegenx.db")
        self.db = RunTrackerDB(str(PROJECT_ROOT / db_path_str))
        self.pipeline = Pipeline(self.config, db=self.db)
        self.cache_key = self.pipeline.cache_key

    def run(self, prompt: str, config_name: str = "interactive", progress_cb=None,
            is_benchmark: bool = False, benchmark_seed: int = None,
            benchmark_key: str = None, provided_contract: dict = None) -> RunContext:
        ctx = self.pipeline.run(prompt, config_name=config_name, progress_cb=progress_cb,
                                is_benchmark=is_benchmark, benchmark_seed=benchmark_seed,
                                benchmark_key=benchmark_key, provided_contract=provided_contract)
        try:
            reporter = Reporter(ctx.run_dir)
            reporter.generate_report(ctx.contract, ctx.verification_report or {})
        except Exception as e:
            print(f"[report] skipped: {e}")
        return ctx


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    pipeline = FrameGenXPipeline()
    pipeline.run("A person holding a red umbrella in the rain.")
