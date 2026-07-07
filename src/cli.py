import argparse
import sys
from src.pipeline import FrameGenXPipeline

def main():
    parser = argparse.ArgumentParser(description="FrameGenX - Contract-Driven Visual Generation")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate Command
    generate_parser = subparsers.add_parser("generate", help="Generate an image from a prompt")
    generate_parser.add_argument("prompt", help="The text prompt to generate")
    
    # Benchmark Command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the automated benchmark suite")
    benchmark_parser.add_argument("--limit", type=int, default=5, help="Number of prompts to benchmark")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        print(f"Generating image for prompt: '{args.prompt}'")
        pipeline = FrameGenXPipeline()
        ctx = pipeline.run(args.prompt)
        
        if ctx.is_successful:
            print(f"✅ Success! Run saved to: {ctx.run_dir}")
        else:
            print(f"❌ Failed to satisfy constraints. See QA report in: {ctx.run_dir}")
            sys.exit(1)
            
    elif args.command == "benchmark":
        from benchmark import run_benchmark
        run_benchmark(limit=args.limit)

if __name__ == "__main__":
    main()
