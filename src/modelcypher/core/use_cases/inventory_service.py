from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from modelcypher.core.domain.models import (
        CheckpointRecord,
        DatasetInfo,
        ModelInfo,
        TrainingJob,
    )
    from modelcypher.core.use_cases.system_service import SystemService


class _InventoryPaths(Protocol):
    """Protocol for paths needed by InventoryService."""

    base: Path
    jobs: Path
    logs: Path
    models: Path
    datasets: Path
    evaluations: Path
    comparisons: Path


@runtime_checkable
class InventoryStore(Protocol):
    """Protocol for the store required by InventoryService.

    Requires access to models, datasets, checkpoints, jobs, and paths.
    The FileSystemStore implements all these methods.
    """

    paths: _InventoryPaths

    def list_models(self) -> list["ModelInfo"]: ...
    def list_datasets(self) -> list["DatasetInfo"]: ...
    def list_checkpoints(self, job_id: str | None = None) -> list["CheckpointRecord"]: ...
    def list_jobs(self) -> list["TrainingJob"]: ...


class InventoryService:
    def __init__(self, store: InventoryStore, system: "SystemService") -> None:
        """Initialize InventoryService with required dependencies.

        Args:
            store: Inventory store port implementation (REQUIRED).
                   Must implement list_models, list_datasets, list_checkpoints,
                   list_jobs, and have a paths attribute.
            system: System service for status information (REQUIRED).
        """
        self.store = store
        self.system = system

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
            "models": models,  # Keep for backward compatibility
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
