"""Explicit local MLflow tracking, with no autologging or remote endpoint."""
from pathlib import Path

from src.training.common import project_path


class ExperimentTracker:
    def __init__(self, db_path: str, experiment: str, run_name: str):
        from mlflow import MlflowClient

        path = project_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient(tracking_uri="sqlite:///" + path.as_posix())
        found = self.client.get_experiment_by_name(experiment)
        artifact_dir = path.parent / "mlflow_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = found.experiment_id if found else self.client.create_experiment(
            experiment, artifact_location=artifact_dir.resolve().as_uri())
        self.run_id = self.client.create_run(experiment_id, tags={
            "mlflow.runName": run_name, "quality_status": "unverified",
            "metric_scope": "training_objective_only",
        }).info.run_id

    def params(self, values: dict):
        for name, value in values.items():
            self.client.log_param(self.run_id, name, str(value))

    def metrics(self, values: dict, step: int = 0):
        import math
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite metric: {name}")
            self.client.log_metric(self.run_id, name, float(value), step=step)

    def artifact(self, path: Path):
        self.client.log_artifact(self.run_id, str(path))

    def finish(self, status: str, quality_status: str = "unverified"):
        if status not in {"FINISHED", "FAILED", "KILLED"}:
            raise ValueError("Invalid MLflow lifecycle status")
        self.client.set_tag(self.run_id, "quality_status", quality_status)
        self.client.set_terminated(self.run_id, status=status)
