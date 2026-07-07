import gc
import json
import torch
from pathlib import Path

def free_memory():
    """Aggressively free system RAM and GPU VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_vram():
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU: {device_name}")
        print(f"Total VRAM: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
