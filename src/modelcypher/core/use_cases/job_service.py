from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.training import TrainingStatus


class JobService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def list_job_records(self, status: str | None = None, active_only: bool = False) -> list[TrainingJob]:
        status_enum = TrainingStatus(status) if status else None
        return self.store.list_jobs(status=status_enum, active_only=active_only)

    def list_jobs(self, status: str | None = None, active_only: bool = False) -> list[dict]:
        status_enum = TrainingStatus(status) if status else None
        jobs = self.store.list_jobs(status=status_enum, active_only=active_only)
        return [
            {
                "jobId": job.job_id,
                "modelId": job.model_id,
                "status": job.status.value,
                "currentStep": job.current_step,
                "totalSteps": job.total_steps,
                "createdAt": job.created_at.isoformat() + "Z",
            }
            for job in jobs
        ]

    def show_job(self, job_id: str, include_loss_history: bool = False) -> dict:
        job = self.store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job not found: {job_id}")
        checkpoints = self.store.list_checkpoints(job_id)
        config = job.config
        if hasattr(config, "__dataclass_fields__"):
            config_payload = asdict(config)
        elif isinstance(config, dict):
            config_payload = config
        else:
            config_payload = {}

        payload = {
            "jobId": job.job_id,
            "status": job.status.value,
            "createdAt": job.created_at.isoformat() + "Z",
            "startedAt": job.started_at.isoformat() + "Z" if job.started_at else None,
            "completedAt": job.completed_at.isoformat() + "Z" if job.completed_at else None,
            "modelId": job.model_id,
            "datasetPath": job.dataset_path,
            "progress": (job.current_step / job.total_steps) if job.total_steps else 0.0,
            "finalLoss": job.loss,
            "checkpoints": [
                {
                    "identifier": f"checkpoint-{c.step}",
                    "step": c.step,
                    "loss": c.loss,
                    "timestamp": c.timestamp.isoformat() + "Z",
                    "filePath": c.file_path,
                }
                for c in checkpoints
            ],
            "hyperparameters": config_payload,
        }
        if include_loss_history:
            payload["lossHistory"] = job.loss_history or []
        return payload

    def delete_job(self, job_id: str) -> dict:
        self.store.delete_job(job_id)
        return {"deleted": job_id}

    def attach(self, job_id: str, since: str | None = None) -> list[str]:
        log_path = self.store.paths.logs / f"{job_id}.events.jsonl"
        if not log_path.exists():
            return []
        lines = log_path.read_text(encoding="utf-8").splitlines()
        if since:
            since_dt = _parse_iso_timestamp(since)
            if since_dt is None:
                return [line for line in lines if since in line]
            filtered = []
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = payload.get("ts")
                ts_dt = _parse_iso_timestamp(ts) if ts else None
                if ts_dt and ts_dt >= since_dt:
                    filtered.append(line)
            return filtered
        return lines


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
