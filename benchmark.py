import json
import time
from pathlib import Path
from src.pipeline import FrameGenXPipeline

def run_benchmark(dataset_path: str = "benchmark_dataset.json", limit: int = 10):
    print("--- Starting FrameGenX Benchmark ---")
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    # For MVP, we filter for object_count prompts since that's what the verification engine handles best right now
    test_cases = [d for d in dataset if d.get("category") == "object_count"]
    
    if limit:
        test_cases = test_cases[:limit]
        
    pipeline = FrameGenXPipeline()
    
    total_runs = len(test_cases)
    successful_runs = 0
    results = []
    
    start_time = time.time()
    
    for idx, case in enumerate(test_cases):
        prompt = case["prompt"]
        print(f"\n[{idx+1}/{total_runs}] Benchmarking Prompt: '{prompt}'")
        
        try:
            ctx = pipeline.run(prompt)
            success = ctx.is_successful
            if success:
                successful_runs += 1
                
            results.append({
                "id": case["id"],
                "prompt": prompt,
                "success": success,
                "run_id": ctx.run_id,
                "repair_loops": getattr(ctx, "repair_loops_executed", 0)
            })
        except Exception as e:
            print(f"Error running pipeline for '{prompt}': {e}")
            results.append({
                "id": case["id"],
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
            
    total_time = time.time() - start_time
    csr = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
    
    summary = {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "constraint_satisfaction_rate": f"{csr:.2f}%",
        "total_time_seconds": round(total_time, 2),
        "results": results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n--- Benchmark Complete ---")
    print(f"Total Runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Constraint Satisfaction Rate (CSR): {csr:.2f}%")
    print(f"Results saved to benchmark_results.json")

if __name__ == "__main__":
    # We limit to 5 for quick MVP verification, user can increase this later
    run_benchmark(limit=5)
