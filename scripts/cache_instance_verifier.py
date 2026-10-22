"""Download only the pinned public SAM2 verifier files and verify their hash."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download
from src.training.common import file_sha256
from src.verification.instance_color import SAM_MODEL, SAM_REVISION, SAM_WEIGHTS_SHA256


def main():
    source = Path(snapshot_download(SAM_MODEL, revision=SAM_REVISION, token=False, max_workers=2,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json", "README.md", "LICENSE*"]))
    if file_sha256(source / "model.safetensors") != SAM_WEIGHTS_SHA256:
        raise ValueError("Downloaded verifier weights do not match the pinned SHA256")
    print(f"Verified local instance model: {SAM_MODEL}@{SAM_REVISION}")


if __name__ == "__main__":
    main()
