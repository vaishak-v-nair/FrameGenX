"""
AdaptStage — dynamic LoRA loading (plug-and-play adaptation).

Registers a pre-offload hook on the shared GenerationEngine: LoRA weights
load right after pipe construction and BEFORE enable_model_cpu_offload
(loading into an already-offloaded pipe is a known diffusers footgun).

config.yaml:
  lora:
    adapters:
      - name: my-style
        source: some-hf-repo/lora-id   # or local path to .safetensors
        weight: 1.0
"""
from src.core.stage import Stage, register_stage
from src.core.context import RunContext


class LoraManager:
    def __init__(self, config):
        self.adapters = (config.get("lora", {}) or {}).get("adapters", []) or []
        self.base_model = config.get("models", {}).get("generator", "stabilityai/sdxl-turbo")
        self.load_report = []

    def hook(self, pipe):
        """Runs between pipe construction and cpu offload."""
        # AdaptStage has already handed this list to RunContext. Preserve the
        # reference so the serialized generation metadata sees load outcomes.
        self.load_report.clear()
        names, weights = [], []
        for a in self.adapters:
            name = a.get("name", "adapter")
            source = a.get("source")
            weight = float(a.get("weight", 1.0))
            if not a.get("enabled", True):
                self.load_report.append({"name": name, "source": source,
                                         "weight": weight, "loaded": False,
                                         "skipped": "enabled: false"})
                print(f"[adapt] LoRA disabled, skipping: {name}")
                continue
            try:
                provenance = None
                load_kwargs = {}
                load_source = source
                if isinstance(source, str):
                    from src.training.registry import adapter_manifest
                    from src.training.common import project_path
                    provenance = adapter_manifest(source, self.base_model,
                        allow_unqualified=a.get("allow_unqualified") is True,
                        require_manifest=a.get("kind") == "trained")
                    local = project_path(source)
                    if local.is_file():
                        # Diffusers offline loading cannot guess a filename,
                        # even when the original source names a local file.
                        load_source = str(local.parent)
                        load_kwargs.update(weight_name=local.name, local_files_only=True)
                    elif a.get("weight_name"):
                        load_kwargs["weight_name"] = a["weight_name"]
                pipe.load_lora_weights(load_source, adapter_name=name, **load_kwargs)
                names.append(name)
                weights.append(weight)
                self.load_report.append({"name": name, "source": source,
                                         "weight": weight, "loaded": True,
                                         "activated": False,
                                         "training_provenance": provenance})
                print(f"[adapt] LoRA loaded: {name} ({source}) w={weight}")
            except Exception as e:
                # Fetch/compat failure must not kill the run — skip and record.
                self.load_report.append({"name": name, "source": source,
                                         "weight": weight, "loaded": False,
                                         "error": str(e)[:200]})
                print(f"[adapt] LoRA FAILED, skipping: {name} ({e})")
        if names:
            try:
                pipe.set_adapters(names, adapter_weights=weights)
            except Exception as e:
                for item in self.load_report:
                    if item.get("loaded"):
                        item["activation_error"] = str(e)[:200]
                print(f"[adapt] set_adapters failed: {e}")
            else:
                for item in self.load_report:
                    if item.get("loaded"):
                        item["activated"] = True


@register_stage
class AdaptStage(Stage):
    name = "adapt"

    def __init__(self, config):
        super().__init__(config)
        self.manager = LoraManager(config)
        self._engine = None

    def bind_engine(self, engine):
        """Pipeline wires this to the GenerateStage's engine instance."""
        self._engine = engine
        if self.manager.adapters:
            engine.register_pre_offload_hook(self.manager.hook)

    def run(self, ctx: RunContext) -> None:
        if not self.manager.adapters:
            print("[adapt] no adapters configured — no-op")
            return
        if self._engine is None:
            print("[adapt] WARNING: no engine bound; LoRA hook inactive")
            return
        print(f"[adapt] {len(self.manager.adapters)} adapter(s) registered for load-time injection")
        ctx.generation_meta["lora_adapters"] = self.manager.adapters
        # Report is populated when the engine actually constructs the pipe;
        # persist whatever we know now, engine meta carries the final state.
        ctx.generation_meta["lora_load_report"] = self.manager.load_report
