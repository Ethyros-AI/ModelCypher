# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Training Service for LoRA adapter fine-tuning.

Orchestrates model training jobs including preflight checks, job management,
and progress monitoring. Supports pause/resume and checkpoint recovery.

Example:
    service = TrainingService(engine=training_engine)
    preflight = service.preflight(config)
    if preflight["canProceed"]:
        job, events = service.start(config, stream=True)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain.training import TrainingConfig

if TYPE_CHECKING:
    from modelcypher.ports.training import TrainingEngine


class TrainingService:
    def __init__(self, engine: "TrainingEngine") -> None:
        self.engine = engine

    def preflight(self, config: TrainingConfig) -> dict:
        result = self.engine.preflight(config)
        return {
            "predictedBatchSize": result.predicted_batch_size,
            "estimatedVRAMUsageBytes": result.estimated_vram_bytes,
            "availableVRAMBytes": result.available_vram_bytes,
            "canProceed": result.can_proceed,
        }

    def start(
        self, config: TrainingConfig, stream: bool = False, detach: bool = False
    ) -> tuple[dict, list[dict]]:
        job, events = self.engine.start(config, stream_events=stream, detach=detach)
        # Support both old config.batch_size and new config.hyperparameters.batch_size
        batch_size = (
            config.hyperparameters.batch_size
            if hasattr(config, "hyperparameters")
            else getattr(config, "batch_size", 1)
        )
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
