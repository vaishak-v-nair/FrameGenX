import json
from pathlib import Path

class Reporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def generate_report(self, contract: dict, decision: dict):
        signals = decision.get("signals", {})
        hard_failures = decision.get("hard_failures", [])
        
        root_causes = []
        if signals.get("object_count", 0) < 2:
            root_causes.append("Scene under-populated -> weak object planning")
        if signals.get("embedding_score", 0.0) < 0.6:
            root_causes.append("Prompt misalignment -> RAG or prompt issue")
        if signals.get("spatial_spread", 0.0) < 50:
            root_causes.append("Objects clustered -> bad composition")
            
        actions = []
        for f in hard_failures:
            if "object_count" in f:
                actions.append("Increase object emphasis in prompt (N2)")
            elif "alignment" in f:
                actions.append("Refine prompt semantics (N1/N2)")
            elif "spatial" in f:
                actions.append("Fix size relationships in scene graph (N2)")
                
        report = {
            "accepted": decision.get("accepted", False),
            "failures": hard_failures,
            "signals": signals,
            "root_causes": root_causes,
            "actions": actions
        }
        
        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        return report
