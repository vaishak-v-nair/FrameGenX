"""Freeze and run a single adapter study; intact reruns do not refit silently."""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform

from src.core.files import directory_lock
from src.training.common import atomic_json, file_sha256, project_path
from src.training.data import validate_records
from src.training.train import code_fingerprint, run_training, source_files
from src.training.model import training_model_source


PACKAGES = ("torch", "numpy", "Pillow", "diffusers", "peft", "transformers",
            "bitsandbytes", "safetensors", "psutil", "mlflow-skinny")


def freeze_study(config, protocol_document: Path) -> dict:
    _, audit = validate_records(project_path(config.dataset))
    source = Path(training_model_source(config.model))
    return {"schema": "framegenx-expanded-study-v1", "config": config.model_dump(),
            "dataset_sha256": audit["dataset_sha256"], "training_code_sha256": code_fingerprint(),
            "protocol_document_sha256": file_sha256(protocol_document),
            "packages": {name: version(name) for name in PACKAGES},
            "python": platform.python_version(), "platform": platform.system(),
            "model_snapshot": source.name,
            "model_sha256": file_sha256(source / "unet/diffusion_pytorch_model.fp16.safetensors"),
            "quality_status": "unverified", "promotion_status": "unqualified",
            "full_contract_CSR": None, "automatic_activation": False}


def checked_result(config, root: Path, pointer: dict) -> dict:
    frozen = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    path = project_path(pointer["result_file"])
    if not path.is_relative_to(project_path(config.output_root)) or file_sha256(path) != pointer["result_sha256"]:
        raise ValueError("Study result is missing, changed or outside the configured output root")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("status") != "trained_pending_evaluation" or result.get("quality_status") != "unverified" or
            result.get("promotion_status") != "unqualified" or result["steps_completed"] != config.max_steps or
            result["distinct_train_images_seen"] != result["train_images"]):
        raise ValueError("Study did not complete its planned fitting and image coverage")
    if (result.get("config") != config.model_dump() or result.get("packages") != frozen["packages"] or
            result.get("model_snapshot") != frozen["model_snapshot"] or
            any(result.get("signature", {}).get(key) != frozen[key] for key in
                ("training_code_sha256", "dataset_sha256", "model_sha256"))):
        raise ValueError("Study result belongs to different frozen inputs")
    digest = hashlib.sha256()
    for _, name in source_files():
        digest.update(name.encode())
        digest.update((path.parent / "code_snapshot" / name).read_bytes())
    if digest.hexdigest() != frozen["training_code_sha256"]:
        raise ValueError("Study source snapshot changed")
    checkpoint = project_path(result["latest_checkpoint"])
    if not checkpoint.is_relative_to(path.parent):
        raise ValueError("Study checkpoint escapes its run directory")
    manifest = json.loads((checkpoint / "adapter.manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "framegenx-adapter-v1" or manifest.get("signature") != result["signature"] or
            manifest.get("step") != config.max_steps or manifest.get("quality_status") != "unverified" or
            manifest["promotion_status"] != "unqualified" or
            manifest["adapter_sha256"] != file_sha256(checkpoint / "adapter.safetensors") or
            manifest["state_sha256"] != file_sha256(checkpoint / "training_state.pt")):
        raise ValueError("Study checkpoint weights or resumable training state changed")
    return result


def run_study(config, protocol_document: Path, output: Path) -> tuple[dict, bool]:
    """Return (actual result, reused). Failed attempts require explicit recovery."""
    with directory_lock(output):
        frozen = freeze_study(config, protocol_document)
        protocol_file = output / "protocol.json"
        if protocol_file.is_file():
            if json.loads(protocol_file.read_text(encoding="utf-8")) != frozen:
                raise ValueError("Study inputs changed; freeze a new study directory")
        else:
            atomic_json(protocol_file, frozen)
        pointer_file = output / "completed.json"
        if pointer_file.is_file():
            return checked_result(config, output, json.loads(pointer_file.read_text(encoding="utf-8"))), True
        journal = output / "execution.json"
        if journal.is_file():
            raise ValueError("A previous study attempt is incomplete; inspect its journal before explicit recovery")
        atomic_json(journal, {"status": "running", "quality_status": "unverified"})
        try:
            result = run_training(config)
            result_file = project_path(config.output_root) / result["run_id"] / "result.json"
            pointer = {"result_file": str(result_file), "result_sha256": file_sha256(result_file)}
            checked_result(config, output, pointer)
            atomic_json(pointer_file, pointer)
            atomic_json(journal, {"status": "completed", "run_id": result["run_id"], "quality_status": "unverified"})
            return result, False
        except BaseException as error:
            atomic_json(journal, {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                  "error_type": type(error).__name__, "quality_status": "unverified"})
            raise
