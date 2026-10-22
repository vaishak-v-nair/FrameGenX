"""
Stage graph core.

Pipeline control flow (design doc, single owner):

    intent -> retrieval (once) -> LOOP: plan -> adapt(attempt 0) -> generate
              -> verify -> reflect -> accept? break : next attempt

Stages mutate RunContext in place. The Pipeline owns the repair loop;
ReflectStage only computes the accept/retry decision and prompt edits.
"""
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from typing import Dict, Type, List, Any, Optional

from src.core.context import RunContext
from src.core.device import exclusive_device


class Stage(ABC):
    """A pluggable pipeline stage. Stages mutate ctx and return None."""

    name: str = "stage"

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def enabled(self) -> bool:
        return self.name in self.config.get("pipeline", {}).get("stages", [])

    @abstractmethod
    def run(self, ctx: RunContext) -> None: ...


STAGE_REGISTRY: Dict[str, Type[Stage]] = {}


def register_stage(cls: Type[Stage]) -> Type[Stage]:
    STAGE_REGISTRY[cls.name] = cls
    return cls


DEFAULT_STAGE_ORDER = ["intent", "retrieval", "plan", "adapt", "generate", "verify", "reflect", "refine"]
# Stages that run once before the repair loop vs inside it
PRE_LOOP_STAGES = ["intent", "retrieval"]
LOOP_STAGES = ["plan", "adapt", "generate", "verify", "reflect"]


def current_cache_key(config: Dict[str, Any]) -> str:
    """Fingerprint the backend behavior that certified an interactive image."""
    project_root = Path(__file__).resolve().parent.parent.parent
    digest = hashlib.sha256(b"framegenx-interactive-cache-v1\0")
    digest.update(json.dumps(config, sort_keys=True, default=str).encode("utf-8"))
    for path in sorted((project_root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        digest.update(requirements.read_bytes())
    return digest.hexdigest()


def build_stages(config: Dict[str, Any]) -> Dict[str, Stage]:
    """Instantiate enabled stages from config. Unknown names are an error."""
    # Import stage modules so their @register_stage decorators run.
    import src.intent.extractor  # noqa: F401
    import src.retrieval.retriever  # noqa: F401
    import src.scene.planner  # noqa: F401
    import src.generation.lora  # noqa: F401
    import src.generation.engine  # noqa: F401
    import src.verification.verify_stage  # noqa: F401
    import src.reflection.reflector  # noqa: F401

    import src.generation.refine_stage  # noqa: F401

    names: List[str] = config.get("pipeline", {}).get("stages", DEFAULT_STAGE_ORDER)
    unknown = [n for n in names if n not in STAGE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {unknown}. Known: {sorted(STAGE_REGISTRY)}")
    return {n: STAGE_REGISTRY[n](config) for n in names}


class Pipeline:
    """Config-driven stage pipeline. Owns the repair loop."""

    def __init__(self, config: Dict[str, Any], db=None):
        self.config = config
        self.db = db
        self.cache_key = current_cache_key(config)
        self.stages = build_stages(config)
        # Wire AdaptStage's LoRA hook onto the GenerateStage's engine —
        # the hook must fire between pipe construction and cpu offload.
        if "adapt" in self.stages and "generate" in self.stages:
            self.stages["adapt"].bind_engine(self.stages["generate"].engine)
        # RefineStage reuses the live generate pipe (no second SDXL load).
        if "refine" in self.stages and "generate" in self.stages:
            self.stages["refine"].bind_engine(self.stages["generate"].engine)

    def _get(self, name: str):
        return self.stages.get(name)

    def _release_device_resources(self):
        """Surrender live model references while the shared device is owned."""
        generate = self.stages.get("generate")
        if generate is not None and getattr(generate, "engine", None) is not None:
            generate.engine.teardown()
        verify = self.stages.get("verify")
        if verify is not None:
            for name in ("qa", "clip", "local_instances"):
                verifier = getattr(verify, name, None)
                if verifier is not None and getattr(verifier, "model", None) is not None:
                    verifier.model.to("cpu")
            ocr = getattr(verify, "easyocr_verifier", None)
            if ocr is not None:
                ocr._unload()

    @exclusive_device(cleanup="_release_device_resources")
    def run(self, prompt: str, config_name: str = "interactive", progress_cb=None,
            is_benchmark: bool = False, benchmark_seed: Optional[int] = None,
            benchmark_key: Optional[str] = None, provided_contract: Optional[dict] = None) -> RunContext:
        missing = [name for name in ("generate", "verify") if name not in self.stages]
        if missing:
            raise ValueError(f"Cannot run without required stage(s): {', '.join(missing)}")
        if provided_contract is not None:
            from copy import deepcopy
            from datetime import datetime, timezone
            from src.intent.models import ContractV2
            if is_benchmark:
                raise ValueError("Provided contracts require their own benchmark manifest")
            if provided_contract.get("prompt") != prompt:
                raise ValueError("Provided contract prompt differs from the requested prompt")
            provided_contract = deepcopy(provided_contract)
            provided_contract.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            validated = ContractV2.model_validate(provided_contract)
            if not validated.entities or any(not e.class_name.strip() for e in validated.entities):
                raise ValueError("Provided contract requires nonempty object entities")
            provided_contract["entities"] = [e.model_dump() for e in validated.entities]

        def emit(event: str, **data):
            # Progress events are best-effort observability for the web UI;
            # a broken callback must never kill a generation.
            if progress_cb is None:
                return
            try:
                progress_cb({"event": event, **data})
            except Exception:
                pass

        ctx = RunContext(prompt=prompt, config=self.config)
        if provided_contract is not None:
            ctx.contract = provided_contract
            ctx.save_artifact("contract_v2.json", provided_contract)
            ctx.save_artifact("contract_source.json", {"type": "provided_contract",
                "sha256": hashlib.sha256(json.dumps(provided_contract, sort_keys=True).encode()).hexdigest()})
        ctx.config_name = config_name
        ctx.is_benchmark = is_benchmark
        ctx.benchmark_seed = benchmark_seed
        ctx.cache_key = benchmark_key if is_benchmark and benchmark_key else self.cache_key
        if provided_contract is not None:
            ctx.cache_key = hashlib.sha256((ctx.cache_key + json.dumps(provided_contract, sort_keys=True)).encode()).hexdigest()
        print(f"\n--- Pipeline run [{config_name}]: '{prompt}' (id {ctx.run_id}) ---")
        emit("run_started", run_id=ctx.run_id, prompt=prompt, config_name=config_name)

        for name in PRE_LOOP_STAGES:
            if name == "intent" and provided_contract is not None:
                continue
            stage = self._get(name)
            if stage is not None:
                emit("stage_started", stage=name, attempt=0)
                stage.run(ctx)
                emit("stage_finished", stage=name, attempt=0)

        max_loops = self.config.get("pipeline", {}).get("max_repair_loops", 3)
        reflect_enabled = self._get("reflect") is not None

        for attempt in range(max_loops + 1):
            ctx.attempt = attempt
            # A previous attempt's report must never certify a new image.
            ctx.verification_report = None
            if attempt > 0:
                ctx.repair_loops_executed += 1
                print(f"--- [REPAIR {attempt}/{max_loops}] edits: {ctx.prompt_edits} ---")

            for name in ["plan", "adapt", "generate", "verify"]:
                stage = self._get(name)
                if stage is None:
                    continue
                if name == "adapt" and attempt > 0:
                    continue  # LoRA loads once, on the first pipe construction
                emit("stage_started", stage=name, attempt=attempt)
                stage.run(ctx)
                emit("stage_finished", stage=name, attempt=attempt)

            report = ctx.verification_report
            if not isinstance(report, dict) or not isinstance(report.get("hard_failures"), list):
                raise RuntimeError(
                    f"Verification stage produced no valid hard-failure report for attempt {attempt}"
                )
            hard_ok = not report["hard_failures"]
            if reflect_enabled:
                emit("stage_started", stage="reflect", attempt=attempt)
                self.stages["reflect"].run(ctx)
                emit("stage_finished", stage="reflect", attempt=attempt)
                accepted = ctx.decision.get("accept", False)
                ctx.prompt_edits = ctx.decision.get("prompt_edits", [])
            else:
                # Baseline semantics: DINO hard failures alone drive the loop.
                accepted = hard_ok
                ctx.decision = {"accept": accepted, "prompt_edits": []}
                ctx.prompt_edits = []

            ctx.verification_report = ctx.verification_report or {}
            ctx.verification_report["accepted"] = accepted
            # Persist the final decision too: the verifier's earlier snapshot
            # deliberately says pending/false before reflection decides.
            ctx.save_artifact(f"verification_{attempt}.json", ctx.verification_report)
            ctx.save_artifact(f"decision_{attempt}.json", ctx.decision)
            emit("decision", attempt=attempt, accepted=accepted,
                 edits=ctx.decision.get("prompt_edits", []),
                 reward=ctx.decision.get("reward"))

            if accepted:
                print(f"[OK] accepted on attempt {attempt}")
                break
                
            # --- SRRL Failure Memory Logging ---
            if ctx.prompt_edits and (self.config.get("retrieval") or {}).get("failure_memory_enabled", True):
                try:
                    from src.retrieval.memory import FailureMemoryDB
                    if not hasattr(self, "_failure_memory"):
                        self._failure_memory = FailureMemoryDB()
                    self._failure_memory.log_failure(
                        prompt=ctx.prompt,
                        hard_failures=ctx.verification_report.get("hard_failures", []),
                        edits=ctx.prompt_edits
                    )
                except Exception as e:
                    print(f"[memory] Failed to log failure: {e}")
                finally:
                    if hasattr(self, "_failure_memory"):
                        self._failure_memory.unload()
            # -----------------------------------

            print(f"[X] rejected on attempt {attempt}: "
                  f"hard={ctx.verification_report.get('hard_failures')} "
                  f"R={ctx.decision.get('reward')}")

        ctx.is_successful = bool(ctx.verification_report and ctx.verification_report.get("accepted"))
        
        if ctx.is_successful and "refine" in self.stages and self.config.get("refinement", {}).get("enabled"):
            # Refine reuses the generate engine's pipe (from_pipe, zero
            # reload) — no second SDXL model, so no teardown needed. The
            # legacy from-disk path inside RefineStage handles the case
            # where no reusable pipe exists.
            emit("stage_started", stage="refine", attempt=0)
            self.stages["refine"].run(ctx)
            emit("stage_finished", stage="refine", attempt=0)

        emit("run_finished", run_id=ctx.run_id, accepted=ctx.is_successful,
             attempts=ctx.attempt + 1)
        if ctx.reward_history:
            ctx.save_artifact("reward_history.json", ctx.reward_history)
        if self.db is not None:
            self.db.log_run(ctx)
            print(f"Run logged ({ctx.get_duration():.1f}s) -> {self.db.db_path}")
        return ctx
