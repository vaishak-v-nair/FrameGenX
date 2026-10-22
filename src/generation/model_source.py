"""Resolve locally cached Diffusers checkpoints without requiring Hub metadata.

An intentionally partial snapshot can contain all files needed by Diffusers
while ``snapshot_download(local_files_only=True)`` rejects unrelated missing
repository files. Resolve the model marker file instead and let Diffusers
validate the actual pipeline components when it loads the local directory.
"""
import os
from pathlib import Path


_TRUE = {"1", "ON", "YES", "TRUE"}


def resolve_model_source(source: str, marker: str = "model_index.json") -> str:
    """Return a local snapshot path in offline mode, or the configured source."""
    source = str(source)
    local = Path(source).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if local.is_absolute() or source.startswith(("./", ".\\", "../", "..\\")):
        raise FileNotFoundError(f"Configured model directory does not exist: {source}")

    offline = any(os.getenv(name, "").upper() in _TRUE
                  for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
    if not offline:
        return source

    from huggingface_hub import hf_hub_download

    try:
        marker_path = hf_hub_download(repo_id=source, filename=marker,
                                      local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"Offline model '{source}' has no cached {marker}; "
            "download the checkpoint first or configure a local directory"
        ) from exc
    return str(Path(marker_path).parent)
