from __future__ import annotations

from dataclasses import asdict

from modelcypher.adapters.local_training import LocalTrainingEngine
from modelcypher.core.domain.training import TrainingConfig


class TrainingService:
    def __init__(self, engine: LocalTrainingEngine | None = None) -> None:
        self.engine = engine or LocalTrainingEngine()

    def preflight(self, config: TrainingConfig) -> dict:
        result = self.engine.preflight(config)
        return {
            "predictedBatchSize": result.predicted_batch_size,
            "estimatedVRAMUsageBytes": result.estimated_vram_bytes,
            "availableVRAMBytes": result.available_vram_bytes,
            "canProceed": result.can_proceed,
        }

    def start(self, config: TrainingConfig, stream: bool = False, detach: bool = False) -> tuple[dict, list[dict]]:
        job, events = self.engine.start(config, stream_events=stream, detach=detach)
        # Support both old config.batch_size and new config.hyperparameters.batch_size
        batch_size = config.hyperparameters.batch_size if hasattr(config, 'hyperparameters') else getattr(config, 'batch_size', 1)
        return {
            "jobId": job.job_id,
            "batchSize": batch_size,
        }, events

    def status(self, job_id: str) -> dict:
        job = self.engine.status(job_id)
        return {
            "jobId": job.job_id,
            "status": job.status.value,
            "currentStep": job.current_step,
            "totalSteps": job.total_steps,
            "currentEpoch": job.current_epoch,
            "totalEpochs": job.total_epochs,
            "loss": job.loss,
            "learningRate": job.learning_rate,
            "createdAt": job.created_at.isoformat() + "Z",
            "updatedAt": job.updated_at.isoformat() + "Z",
            "modelId": job.model_id,
            "datasetPath": job.dataset_path,
        }

    def pause(self, job_id: str) -> dict:
        job = self.engine.pause(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def resume(self, job_id: str) -> dict:
        job = self.engine.resume(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def cancel(self, job_id: str) -> dict:
        job = self.engine.cancel(job_id)
        return {"jobId": job.job_id, "status": job.status.value}

    def logs(self, job_id: str, tail: int = 100) -> list[str]:
        return self.engine.logs(job_id, tail=tail)
