import json
from pathlib import Path
from src.intent.extractor import IntentExtractor
from src.scene.planner import ScenePlanner
from src.generation.engine import GenerationEngine
from src.verification.qa import QAVerifier
from src.reporting.analyzer import Reporter
from src.utils import free_memory

from src.core.context import RunContext, load_config
from src.core.db import RunTrackerDB

class FrameGenXPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.db = RunTrackerDB(self.config.get("paths", {}).get("db_path", "runs/framegenx.db"))
        
        self.extractor = IntentExtractor()
        self.planner = ScenePlanner(resolution=512)
        self.generator = GenerationEngine(self.config)
        self.qa = QAVerifier()
        
    def run(self, prompt: str) -> RunContext:
        ctx = RunContext(prompt=prompt, config=self.config)
        print(f"\n--- Running Pipeline for Prompt: '{prompt}' ---")
        print(f"Run ID: {ctx.run_id}")
        
        # 1. Intent Extraction
        print("[1/6] Extracting Intent...")
        ctx.contract = self.extractor.extract(prompt)
        ctx.save_artifact("contract_v2.json", ctx.contract)
        if self.config.get("system", {}).get("aggressive_memory_management", True):
            free_memory()
            
        max_loops = self.config.get("pipeline", {}).get("max_repair_loops", 3)
        decision = {}
        
        for attempt in range(max_loops + 1):
            is_repair = attempt > 0
            if is_repair:
                print(f"\n--- [REPAIR LOOP {attempt}/{max_loops}] Triggered ---")
                ctx.repair_loops_executed += 1
                
            # 2. Scene Planning
            print("[2/6] Planning Scene...")
            # If in repair, we would dynamically alter the layout based on `decision["hard_failures"]`
            # If in repair, we dynamically alter the layout based on `decision["hard_failures"]`
            ctx.scene_plan, control_hints, gen_config, final_prompt, masks = self.planner.plan(ctx.contract, attempt)
            ctx.save_artifact(f"scene_plan_{attempt}.json", ctx.scene_plan)
            ctx.save_artifact(f"generation_config_{attempt}.json", gen_config)
            ctx.save_artifact(f"final_prompt_{attempt}.txt", final_prompt, is_json=False)
            
            for mask_name, mask_img in masks.items():
                ctx.save_artifact(f"{mask_name}_{attempt}.png", mask_img, is_json=False)
                
            # 3. Generation (Unified pass with ControlNet)
            print("[3/6] Generating Final Image...")
            output_path = ctx.run_dir / f"generated_{attempt}.png"
            gen_meta = self.generator.generate(final_prompt, gen_config, control_hints, masks, output_path)
            ctx.save_artifact(f"generation_meta_{attempt}.json", gen_meta)
            if self.config.get("system", {}).get("aggressive_memory_management", True):
                free_memory()
                
            # 4. QA Verification
            print("[4/6] Verifying Quality...")
            decision = self.qa.verify(output_path, ctx.contract)
            ctx.verification_report = decision
            ctx.save_artifact(f"decision_{attempt}.json", decision)
            
            if decision.get("accepted", False):
                print(f"✅ QA Passed on attempt {attempt}!")
                break
            else:
                print(f"❌ QA Failed: {decision.get('hard_failures')}")
                if attempt == max_loops:
                    print("Maximum repair loops exhausted. Giving up.")
        
        # 5. Analysis & Reporting
        print("[5/6] Generating Report...")
        reporter = Reporter(ctx.run_dir)
        report = reporter.generate_report(ctx.contract, decision)
        
        ctx.is_successful = decision.get("accepted", False)
        
        # Finalize and Log
        self.db.log_run(ctx)
        
        print(f"Run completed in {ctx.get_duration():.2f}s. Metadata saved to {self.db.db_path}")
        return ctx

if __name__ == "__main__":
    pipeline = FrameGenXPipeline()
    pipeline.run("A person holding a red umbrella in the rain.")
