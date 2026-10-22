from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import snapshot_download

# This script downloads models to E: cache without loading them into RAM/VRAM, bypassing OOM issues during transfer.
print(f"HF_HOME={os.environ.get('HF_HOME')}")
print(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")

print("Downloading diffusers/controlnet-canny-sdxl-1.0...")
snapshot_download("diffusers/controlnet-canny-sdxl-1.0")

print("Downloading stabilityai/sdxl-turbo...")
snapshot_download("stabilityai/sdxl-turbo")

print("Downloads Complete!")
