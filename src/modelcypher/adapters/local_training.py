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

import asyncio
import json
import logging
import multiprocessing
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.optimizers as optim

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.backends import default_backend
from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import (
    Hyperparameters as DomainHyperparameters,
)
from modelcypher.core.domain.training import (
    LoRAConfig as DomainLoRAConfig,
)
from modelcypher.core.domain.training import (
    PreflightResult,
    TrainingStatus,
)
from modelcypher.core.domain.training import (
    TrainingConfig as DomainTrainingConfig,
)
from modelcypher.core.domain.training import (
    TrainingEngine as DomainTrainingEngine,
)
from modelcypher.core.domain.training import (
    TrainingProgress as DomainTrainingProgress,
)
from modelcypher.ports.backend import Backend
from modelcypher.ports.training import TrainingEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import expand_path

from .model_loader import load_model_for_training
from .training_dataset import TrainingDataset

logger = logging.getLogger(__name__)


def _get_hp_attr(config: Any, attr: str, default: Any = None) -> Any:
    """Get hyperparameter attribute from config, supporting both old and new formats.

    New format: config.hyperparameters.attr
    Old format: config.attr
    """
    if hasattr(config, "hyperparameters") and config.hyperparameters is not None:
        return getattr(config.hyperparameters, attr, default)
    return getattr(config, attr, default)


class LocalTrainingEngine(TrainingEngine):
    """
    Production-ready Training Engine for local MLX fine-tuning.

    Wires the ModelCypher adapters to the domain TrainingEngine.
    """

    def __init__(
        self, store: FileSystemStore | None = None, backend: Backend | None = None
    ) -> None:
        self.store = store or FileSystemStore()
        self.backend = backend or default_backend()
        self.paths = self.store.paths
        self.lock = FileLock(self.paths.base / "training.lock")
        self.domain_engine = DomainTrainingEngine()
        self._loop = None

    def preflight(self, config: Any) -> PreflightResult:
        """Estimate resources before starting."""
        # Simple heuristic for VRAM
        dataset_path = expand_path(config.dataset_path)
        dataset_size = os.path.getsize(dataset_path) if dataset_path.exists() else 0

        # Assume 4-bit model takes ~4.5GB, float16 takes ~15GB for 7B
        # This is a guestimate.
        estimated_vram = 5 * 1024 * 1024 * 1024 + int(dataset_size * 1.5)
        available_memory = self._available_memory_bytes()

        can_proceed = estimated_vram < available_memory
        batch_size = _get_hp_attr(config, "batch_size", 1)
        return PreflightResult(
            predicted_batch_size=batch_size or 1,
            estimated_vram_bytes=estimated_vram,
            available_vram_bytes=available_memory,
            can_proceed=can_proceed,
        )

    def start(
        self, config: Any, stream_events: bool = False, detach: bool = False
    ) -> tuple[TrainingJob, list[dict]]:
        """Start a real fine-tuning job."""
        # Pre-check lock
        if self.lock.is_locked():
            raise RuntimeError("Another training job is already running on this machine.")

        job_id = f"job-{uuid.uuid4()}"
        created_at = datetime.utcnow()

        job = TrainingJob(
            job_id=job_id,
            status=TrainingStatus.running,
            model_id=config.model_id,
            dataset_path=config.dataset_path,
            created_at=created_at,
            updated_at=created_at,
            started_at=created_at,
            total_epochs=_get_hp_attr(config, "epochs", 3),
            learning_rate=_get_hp_attr(config, "learning_rate", 1e-5),
            config=config,
            metrics={},
            metrics_history=[],
        )
        self.store.save_job(job)

        if detach:
            # Spawn background process
            process = multiprocessing.Process(
                target=self._run_training_loop, args=(job_id, config, False)
            )
            process.daemon = False  # We want it to survive
            process.start()

            logger.info("Training detached. Job ID: %s", job_id)
            return job, [{"type": "detached", "data": {"jobId": job_id}}]

        # Synchronous execution
        return self._run_training_loop(job_id, config, stream_events)

    def _run_training_loop(
        self, job_id: str, config: Any, stream_events: bool
    ) -> tuple[TrainingJob, list[dict]]:
        """Internal training loop runner (can be called in background)."""
        try:
            self.lock.acquire()
        except FileLockError as exc:
            logger.error("Failed to acquire lock for job %s: %s", job_id, exc)
            return

        job = self.store.get_job(job_id)
        if not job:
            return

        events: list[dict] = []
        event_path = self._event_log_path(job_id)

        def emit(event: dict) -> None:
            event["jobId"] = job_id
            event["ts"] = datetime.utcnow().isoformat() + "Z"
            if stream_events:
                events.append(event)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

        emit(
            {
                "type": "trainingStart",
                "data": {
                    "jobId": job_id,
                    "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
                },
            }
        )

        # Map to Domain Config
        domain_hp = DomainHyperparameters(
            batch_size=_get_hp_attr(config, "batch_size", 4),
            learning_rate=_get_hp_attr(config, "learning_rate", 1e-5),
            epochs=_get_hp_attr(config, "epochs", 3),
            gradient_accumulation_steps=_get_hp_attr(config, "gradient_accumulation_steps", 1),
            seed=_get_hp_attr(config, "seed", 42),
        )

        domain_lora = None
        lora = getattr(config, "lora", None) or getattr(config, "lora_config", None)
        if lora:
            domain_lora = DomainLoRAConfig(
                rank=lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
                target_modules=getattr(
                    lora, "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
                ),
            )

        domain_config = DomainTrainingConfig(
            model_id=config.model_id,
            dataset_path=config.dataset_path,
            output_path=str(self.paths.base / "checkpoints"),
            hyperparameters=domain_hp,
            lora_config=domain_lora,
            resume_from_checkpoint_path=getattr(config, "resume_from", None)
            or getattr(config, "resume_from_checkpoint_path", None),
        )

        # Execution using Domain Engine
        try:
            # 1. Load Model
            model, tokenizer = load_model_for_training(config.model_id, domain_lora)

            # 2. Load Dataset
            dataset = TrainingDataset(
                config.dataset_path, tokenizer, batch_size=_get_hp_attr(config, "batch_size", 4)
            )

            # 3. Setup Optimizer
            optimizer = optim.AdamW(learning_rate=_get_hp_attr(config, "learning_rate", 1e-5))

            # 4. Progress Bridge
            def progress_callback(progress: DomainTrainingProgress):
                nonlocal job
                latest_metrics = progress.metrics

                # Update Job Record
                job = TrainingJob(
                    **{
                        **asdict(job),
                        "current_step": progress.step,
                        "total_steps": progress.total_steps,
                        "current_epoch": progress.epoch,
                        "loss": progress.loss,
                        "learning_rate": progress.learning_rate,
                        "updated_at": datetime.utcnow(),
                        "metrics": latest_metrics,
                    }
                )
                self.store.update_job(job)

                # Emit Event
                emit(
                    {
                        "type": "trainingProgress",
                        "sequence": progress.step,
                        "data": {
                            "step": progress.step,
                            "totalSteps": progress.total_steps,
                            "loss": progress.loss,
                            "learningRate": progress.learning_rate,
                            "tokensPerSecond": progress.tokens_per_second,
                            "metrics": latest_metrics,
                        },
                    }
                )

            # 5. Run Training Loop (Synchronous wrap for now)
            async def run_train():
                await self.domain_engine.train(
                    job_id=job_id,
                    config=domain_config,
                    model=model,
                    optimizer=optimizer,
                    data_provider=dataset,
                    progress_callback=progress_callback,
                )

            asyncio.run(run_train())

            # 6. Finalize
            job = TrainingJob(
                **{
                    **asdict(job),
                    "status": TrainingStatus.completed,
                    "completed_at": datetime.utcnow(),
                }
            )
            self.store.update_job(job)
            emit({"type": "trainingCompleted", "data": {"final_loss": job.loss}})

            return job, events

        except Exception as exc:
            logger.error("Training failed for job %s: %s", job_id, exc, exc_info=True)
            job = self.store.get_job(job_id) or job
            job = TrainingJob(
                **{
                    **asdict(job),
                    "status": TrainingStatus.failed,
                    "completed_at": datetime.utcnow(),
                }
            )
            self.store.update_job(job)
            emit({"type": "error", "data": {"message": str(exc)}})
            raise
        finally:
            self.lock.release()

    def status(self, job_id: str) -> TrainingJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job not found: {job_id}")
        return job

    def pause(self, job_id: str) -> TrainingJob:
        self.domain_engine._paused_jobs.add(job_id)
        if job_id in self.domain_engine._pause_events:
            self.domain_engine._pause_events[job_id].clear()

        job = self.status(job_id)
        job = TrainingJob(
            **{**asdict(job), "status": TrainingStatus.paused, "updated_at": datetime.utcnow()}
        )
        self.store.update_job(job)
        return job

    def resume(self, job_id: str) -> TrainingJob:
        self.domain_engine._paused_jobs.discard(job_id)
        if job_id in self.domain_engine._pause_events:
            self.domain_engine._pause_events[job_id].set()

        job = self.status(job_id)
        job = TrainingJob(
            **{**asdict(job), "status": TrainingStatus.running, "updated_at": datetime.utcnow()}
        )
        self.store.update_job(job)
        return job

    def cancel(self, job_id: str) -> TrainingJob:
        self.domain_engine._cancelled_jobs.add(job_id)

        job = self.status(job_id)
        job = TrainingJob(
            **{**asdict(job), "status": TrainingStatus.cancelled, "updated_at": datetime.utcnow()}
        )
        self.store.update_job(job)
        return job

    def logs(self, job_id: str, tail: int = 100) -> list[str]:
        path = self._event_log_path(job_id)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        return [line.rstrip("\n") for line in lines[-tail:]]

    def _event_log_path(self, job_id: str) -> Path:
        return self.paths.logs / f"{job_id}.events.jsonl"

    @staticmethod
    def _available_memory_bytes() -> int:
        try:
            # For Apple Silicon / Mac
            import subprocess

            output = subprocess.check_output(["sysctl", "hw.memsize"])
            return int(output.decode().split(":")[1].strip())
        except Exception:
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(pages * page_size)
            except (ValueError, AttributeError, OSError):
                return 0
