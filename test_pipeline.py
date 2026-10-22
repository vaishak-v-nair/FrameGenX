from dotenv import load_dotenv
load_dotenv()
import os

from src.pipeline import FrameGenXPipeline
from src.core.context import RunContext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "E:\\hf_cache"

prompt = "A sleek red sports car parked in front of a futuristic neon city."
print(f"Starting pipeline for: {prompt}")

pipeline = FrameGenXPipeline("config.yaml")
ctx = pipeline.run(prompt)

print("Pipeline finished.")
print("Attempts:", len(ctx.reward_history))
print("Final Verification:", ctx.verification_report.get("accepted"))
