import json
import time
from pathlib import Path
from src.pipeline import FrameGenXPipeline

def run_benchmarks():
    dataset_path = Path("benchmark_dataset.json")
    if not dataset_path.exists():
        print("Dataset not found. Run generate_benchmark.py first.")
        return
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    pipeline = FrameGenXPipeline(output_dir="benchmark_outputs")
    results = []
    
    print(f"Starting benchmark run for {len(dataset)} prompts...")
    
    for i, item in enumerate(dataset):
        prompt_id = item["id"]
        prompt = item["prompt"]
        category = item["category"]
        
        print(f"\n[{i+1}/{len(dataset)}] Benchmarking: {prompt_id} - '{prompt}'")
        
        start_time = time.time()
        try:
            report = pipeline.run(prompt)
            duration = time.time() - start_time
            
            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "accepted": report.get("accepted"),
                "failures": report.get("failures"),
                "duration_seconds": duration,
                "error": None
            })
        except Exception as e:
            print(f"Error on {prompt_id}: {e}")
            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": prompt,
                "accepted": False,
                "failures": ["CRITICAL_ERROR"],
                "duration_seconds": time.time() - start_time,
                "error": str(e)
            })
            
        # Save intermediate results
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    # Calculate stats
    total = len(results)
    accepted = sum(1 for r in results if r["accepted"])
    avg_time = sum(r["duration_seconds"] for r in results) / total if total > 0 else 0
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Total Prompts: {total}")
    print(f"Accepted: {accepted} ({(accepted/total)*100:.1f}%)")
    print(f"Avg Time/Prompt: {avg_time:.1f}s")
    
if __name__ == "__main__":
    run_benchmarks()
