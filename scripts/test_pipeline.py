import os
from src.pipeline import FrameGenXPipeline

def main():
    pipe = FrameGenXPipeline()
    prompt = "Panoramic view of jagged alpine peaks at sunrise, wildflowers in bloom in the foreground, layers of mist filling the valleys below, warm golden light painting the summit, ultra-wide composition, photorealistic landscape photography, hyper-detailed, no people."
    print("Running pipeline for prompt:", prompt)
    ctx = pipe.run(prompt)
    print("Accepted:", getattr(ctx, "accepted", False))
    print("Final Reward:", getattr(ctx, "reward", 0.0))
    print("Run ID:", ctx.run_id)
    print("Verification Details:")
    if hasattr(ctx, "verification_report"):
        for v in ctx.verification_report.get("details", []):
            print(v)

if __name__ == "__main__":
    main()
