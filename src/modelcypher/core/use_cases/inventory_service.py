from __future__ import annotations

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.use_cases.system_service import SystemService


class InventoryService:
    def __init__(self, store: FileSystemStore | None = None, system: SystemService | None = None) -> None:
        self.store = store or FileSystemStore()
        self.system = system or SystemService()

    def inventory(self) -> dict:
        models = [
            {
                "id": model.id,
                "alias": model.alias,
                "format": model.format,
                "sizeBytes": model.size_bytes,
                "path": model.path,
            }
            for model in self.store.list_models()
        ]
        datasets = [
            {
                "id": dataset.id,
                "name": dataset.name,
                "path": dataset.path,
                "sizeBytes": dataset.size_bytes,
                "exampleCount": dataset.example_count,
            }
            for dataset in self.store.list_datasets()
        ]
        checkpoints = [
            {"jobId": c.job_id, "step": c.step, "loss": c.loss, "path": c.file_path}
            for c in self.store.list_checkpoints()
        ]
        jobs = [
            {
                "jobId": job.job_id,
                "status": job.status.value,
                "progress": (job.current_step / job.total_steps) if job.total_steps else 0.0,
                "modelId": job.model_id,
                "datasetPath": job.dataset_path,
            }
            for job in self.store.list_jobs()
        ]
        workspace = {
            "cwd": __import__("os").getcwd(),
            "jobStore": str(self.store.paths.jobs),
        }
        return {
            "system": {
                **self.system.status(),
                "node": __import__("platform").node(),
                "machine": __import__("platform").machine(),
                "processor": __import__("platform").processor(),
            },
            "buckets": {
                "models": models,
                "datasets": datasets,
                "checkpoints": checkpoints,
                "jobs": jobs,
            },
            "models": models, # Keep for backward compatibility
            "datasets": datasets,
            "checkpoints": checkpoints,
            "jobs": jobs,
            "paths": {
                "base": str(self.store.paths.base),
                "jobs": str(self.store.paths.jobs),
                "logs": str(self.store.paths.logs),
                "models": str(self.store.paths.models),
                "datasets": str(self.store.paths.datasets),
                "evaluations": str(self.store.paths.evaluations),
                "comparisons": str(self.store.paths.comparisons),
            },
            "workspace": workspace,
            "mlxVersion": self.system._mlx_version(),
            "policies": {
                "safeGPU": True,
                "evalRequired": True,
                "tokenizerSplit": True,
                "logging": "python-logging",
                "automaticPruning": False,
            },
            "version": __import__("modelcypher").__version__,
        }
