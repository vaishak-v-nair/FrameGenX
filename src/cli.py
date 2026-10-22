import argparse
import sys
from dotenv import load_dotenv
load_dotenv()
from src.pipeline import FrameGenXPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FrameGenX - Contract-Driven Visual Generation"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate Command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate an image from a prompt"
    )
    generate_parser.add_argument("prompt", help="The text prompt to generate")
    generate_parser.add_argument("--config", default="config.yaml", help="Application YAML configuration")
    generate_parser.add_argument("--contract", help="Validated local contract JSON; bypasses intent extraction")
    generate_parser.add_argument("--seed", type=int, help="Recorded deterministic generation seed")

    # Benchmark Command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run the automated benchmark suite"
    )
    benchmark_parser.add_argument(
        "--limit", type=int, default=5, help="Number of prompts to benchmark"
    )

    args = parser.parse_args()

    if args.command == "generate":
        import json
        from src.training.common import project_path
        if not project_path(args.config).is_file():
            parser.error("Configuration file does not exist")
        if args.seed is not None and not 0 <= args.seed < 2**63:
            parser.error("Seed must be in [0, 2**63)")
        contract = json.loads(project_path(args.contract).read_text(encoding="utf-8")) if args.contract else None
        print(f"Generating image for prompt: '{args.prompt}'")
        overrides = {"generation": {"seed": args.seed}} if args.seed is not None else None
        pipeline = FrameGenXPipeline(config_path=args.config, config_overrides=overrides)
        ctx = pipeline.run(args.prompt, provided_contract=contract)

        if ctx.is_successful:
            evidence = (ctx.verification_report or {}).get("contract_evidence", {})
            label = ("Extracted checks passed" if evidence.get("status") == "passed" else
                     "Operational checks passed; remaining contract requirements need review")
            print(f"{label}. Run saved to: {ctx.run_dir}")
        else:
            print(f"Failed to satisfy constraints. See QA report in: {ctx.run_dir}")
            sys.exit(1)

    elif args.command == "benchmark":
        from benchmark import run_benchmark

        run_benchmark(limit=args.limit)


if __name__ == "__main__":
    main()
