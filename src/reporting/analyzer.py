import json
from pathlib import Path


class Reporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_report(self, contract: dict, decision: dict):
        signals = decision.get("signals", {})
        hard_failures = decision.get("hard_failures", [])

        # A failed check identifies the symptom, not its cause. Keep this
        # field empty until a cause is supported by recorded evidence.
        root_causes = []
        actions = []
        for f in hard_failures:
            if f.get("type") == "count_mismatch":
                actions.append(
                    f"Review the {f.get('class_name', 'object')} count: "
                    f"expected {f.get('expected')}, detected {f.get('detected')}."
                )
            elif f.get("type") == "spatial_violation":
                actions.append(
                    f"Review {f.get('class_name', 'the relation')}: "
                    f"expected {f.get('expected')}; {f.get('detected')}."
                )

        report = {
            "accepted": decision.get("accepted", False),
            "failures": hard_failures,
            "signals": signals,
            "root_causes": root_causes,
            "actions": actions,
        }

        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
