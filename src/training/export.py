"""Complete PEFT metadata and a tensor-preserving repair for early pilot exports."""
import argparse
from enum import Enum
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from src.training.common import atomic_json, file_sha256, project_path


def complete_lora_metadata(config) -> dict:
    def encode(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Enum):
            return value.value
        raise TypeError(f"Unsupported PEFT metadata value: {type(value).__name__}")
    return json.loads(json.dumps(config.to_dict(), default=encode))


def repair_export(checkpoint: Path) -> dict:
    from peft import LoraConfig
    adapter = checkpoint / "adapter.safetensors"
    manifest_path = adapter.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "framegenx-adapter-v1" or file_sha256(adapter) != manifest["adapter_sha256"]:
        raise ValueError("Export provenance changed; refuse to repair unknown weights")
    with safe_open(str(adapter), framework="pt", device="cpu") as stream:
        header = stream.metadata()
    old_config = json.loads(header["lora_adapter_metadata"])
    if "unet.rank_pattern" in old_config:
        return {"checkpoint": str(checkpoint), "status": "already_complete"}
    profile = manifest["signature"]["profile"]
    config = LoraConfig(r=profile["rank"], lora_alpha=profile["lora_alpha"], lora_dropout=0.0,
                        target_modules=profile["target_modules"], init_lora_weights="gaussian")
    full = complete_lora_metadata(config)
    metadata = {**header, "lora_adapter_metadata": json.dumps({"unet." + k: v for k, v in full.items()},
                                                               sort_keys=True)}
    weights = load_file(str(adapter), device="cpu")
    backup = checkpoint / "adapter.incomplete_metadata.safetensors"
    if not backup.exists():
        shutil.copy2(adapter, backup)
    temporary = checkpoint / "adapter.metadata_repair.tmp.safetensors"
    save_file(weights, str(temporary), metadata=metadata)
    repaired = load_file(str(temporary), device="cpu")
    if weights.keys() != repaired.keys() or not all(torch.equal(weights[k], repaired[k]) for k in weights):
        raise RuntimeError("Metadata repair changed adapter tensors")
    original_sha = manifest["adapter_sha256"]
    temporary.replace(adapter)
    manifest["adapter_sha256"] = file_sha256(adapter)
    manifest["export_repair"] = {"reason": "incomplete_PEFT_metadata_missing_rank_pattern",
        "original_adapter_sha256": original_sha, "original_file": backup.name,
        "tensors_identical": True, "tensor_count": len(weights),
        "export_code_sha256": file_sha256(Path(__file__))}
    atomic_json(manifest_path, manifest)
    return {"checkpoint": str(checkpoint), "status": "repaired", "tensors_identical": True,
            "tensor_count": len(weights), "adapter_sha256": manifest["adapter_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(repair_export(project_path(args.checkpoint)), indent=2))


if __name__ == "__main__":
    main()
