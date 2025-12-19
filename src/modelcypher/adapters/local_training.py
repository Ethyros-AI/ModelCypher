from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from modelcypher.adapters.filesystem_storage import FileSystemStore, StoragePaths
from modelcypher.backends import default_backend
from modelcypher.core.domain.models import CheckpointRecord, TrainingJob
from modelcypher.core.domain.training import PreflightResult, TrainingConfig, TrainingStatus
from modelcypher.ports.backend import Backend
from modelcypher.ports.training import TrainingEngine
from modelcypher.utils.locks import FileLock, FileLockError
from modelcypher.utils.paths import expand_path


class LocalTrainingEngine(TrainingEngine):
    def __init__(self, store: FileSystemStore | None = None, backend: Backend | None = None) -> None:
        self.store = store or FileSystemStore()
        self.backend = backend or default_backend()
        self.paths = self.store.paths
        self.lock = FileLock(self.paths.base / "training.lock")

    def preflight(self, config: TrainingConfig) -> PreflightResult:
        dataset_size = os.path.getsize(expand_path(config.dataset_path)) if config.dataset_path else 0
        available_memory = self._available_memory_bytes()
        estimated_vram = int(dataset_size * 2)
        predicted_batch = max(1, config.batch_size)
        can_proceed = estimated_vram < available_memory
        return PreflightResult(
            predicted_batch_size=predicted_batch,
            estimated_vram_bytes=estimated_vram,
            available_vram_bytes=available_memory,
            can_proceed=can_proceed,
        )

    def start(self, config: TrainingConfig, stream_events: bool = False) -> tuple[TrainingJob, list[dict]]:
        try:
            self.lock.acquire()
        except FileLockError as exc:
            raise RuntimeError("Another job is already running") from exc

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
            total_epochs=config.epochs,
            learning_rate=config.learning_rate,
            config=config,
        )
        self.store.save_job(job)

        events: list[dict] = []
        event_path = self._event_log_path(job_id)
        if event_path.exists():
            event_path.unlink()

        def emit(event: dict) -> None:
            event["jobId"] = job_id
            event["ts"] = datetime.utcnow().isoformat() + "Z"
            if stream_events:
                events.append(event)
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

        emit({"schemaVersion": "0.1.0", "type": "trainingStart", "data": {"jobId": job_id}})

        weights, lora = self._load_or_init_weights(config)
        batch_size = config.batch_size
        dataset = self._load_dataset(config.dataset_path, batch_size)
        total_steps = len(dataset) * config.epochs

        job = TrainingJob(
            **{
                **asdict(job),
                "total_steps": total_steps,
            }
        )
        self.store.update_job(job)

        loss_history: list[dict] = []
        step = 0
        start_time = time.time()

        try:
            for epoch in range(1, config.epochs + 1):
                for batch in dataset:
                    step += 1
                    loss = self._train_step(weights, lora, batch, config)
                    elapsed = max(time.time() - start_time, 1e-6)
                    tokens_per_second = (step * batch_size) / elapsed
                    loss_history.append({"step": step, "loss": loss})
                    job = TrainingJob(
                        **{
                            **asdict(job),
                            "current_step": step,
                            "current_epoch": epoch,
                            "loss": loss,
                            "learning_rate": config.learning_rate,
                            "updated_at": datetime.utcnow(),
                        }
                    )
                    self.store.update_job(job)

                    emit(
                        {
                            "schemaVersion": "0.1.0",
                            "type": "trainingProgress",
                            "sequence": step,
                            "data": {
                                "step": step,
                                "totalSteps": total_steps,
                                "loss": loss,
                                "learningRate": config.learning_rate,
                                "tokensPerSecond": tokens_per_second,
                            },
                        }
                    )

            checkpoint_path = self._save_checkpoint(job_id, step, loss, weights, lora)
            checkpoint = CheckpointRecord(
                job_id=job_id,
                step=step,
                loss=loss,
                timestamp=datetime.utcnow(),
                file_path=checkpoint_path,
            )
            self.store.add_checkpoint(checkpoint)

            job = TrainingJob(
                **{
                    **asdict(job),
                    "status": TrainingStatus.completed,
                    "completed_at": datetime.utcnow(),
                    "loss_history": loss_history,
                }
            )
            self.store.update_job(job)
            emit({"schemaVersion": "0.1.0", "type": "trainingCompleted", "data": {"step": step}})
            return job, events
        except Exception as exc:
            job = TrainingJob(
                **{
                    **asdict(job),
                    "status": TrainingStatus.failed,
                    "completed_at": datetime.utcnow(),
                }
            )
            self.store.update_job(job)
            emit({"schemaVersion": "0.1.0", "type": "error", "data": {"message": str(exc)}})
            raise
        finally:
            self.lock.release()

    def status(self, job_id: str) -> TrainingJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job not found: {job_id}")
        return job

    def pause(self, job_id: str) -> TrainingJob:
        job = self.status(job_id)
        job = TrainingJob(**{**asdict(job), "status": TrainingStatus.paused, "updated_at": datetime.utcnow()})
        self.store.update_job(job)
        return job

    def resume(self, job_id: str) -> TrainingJob:
        job = self.status(job_id)
        job = TrainingJob(**{**asdict(job), "status": TrainingStatus.running, "updated_at": datetime.utcnow()})
        self.store.update_job(job)
        return job

    def cancel(self, job_id: str) -> TrainingJob:
        job = self.status(job_id)
        job = TrainingJob(**{**asdict(job), "status": TrainingStatus.cancelled, "updated_at": datetime.utcnow()})
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

    def _load_or_init_weights(self, config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any] | None]:
        path = expand_path(config.resume_from or config.model_id)
        weight_path = None
        if path.is_dir():
            candidate = path / "weights.npz"
            if candidate.exists():
                weight_path = candidate
        elif path.suffix == ".npz":
            weight_path = path

        if weight_path and weight_path.exists():
            data = np.load(weight_path)
            w = self.backend.array(data["W"], dtype=np.float32)
            lora = None
            if "lora_A" in data and "lora_B" in data:
                lora = {
                    "A": self.backend.array(data["lora_A"], dtype=np.float32),
                    "B": self.backend.array(data["lora_B"], dtype=np.float32),
                    "scale": float(data.get("lora_scale", 1.0)),
                }
            return {"W": w}, lora

        input_dim = 32
        output_dim = 32
        rng = np.random.default_rng(config.seed)
        w = self.backend.array(rng.standard_normal((output_dim, input_dim)).astype(np.float32))
        lora = None
        if config.lora:
            rank = config.lora.rank
            lora = {
                "A": self.backend.array(rng.standard_normal((input_dim, rank)).astype(np.float32) * 0.01),
                "B": self.backend.array(rng.standard_normal((rank, output_dim)).astype(np.float32) * 0.01),
                "scale": float(config.lora.scale),
            }
        return {"W": w}, lora

    def _train_step(
        self,
        weights: dict[str, Any],
        lora: dict[str, Any] | None,
        batch: np.ndarray,
        config: TrainingConfig,
    ) -> float:
        x = self.backend.array(batch.astype(np.float32))
        w = weights["W"]
        if lora:
            a = lora["A"]
            b = lora["B"]
            delta = self.backend.matmul(self.backend.transpose(b), self.backend.transpose(a))
            w_eff = w + lora["scale"] * delta
        else:
            w_eff = w
        preds = self.backend.matmul(x, self.backend.transpose(w_eff))
        diff = preds - x
        loss_tensor = self.backend.sum(diff * diff) / float(np.prod(diff.shape))
        self.backend.eval(loss_tensor)
        loss = float(loss_tensor.item())

        grad_scale = 2.0 / float(np.prod(diff.shape))
        grad = diff * grad_scale
        grad_w = self.backend.matmul(self.backend.transpose(grad), x)

        if lora:
            g_c = lora["scale"] * self.backend.transpose(grad_w)
            grad_a = self.backend.matmul(g_c, self.backend.transpose(b))
            grad_b = self.backend.matmul(self.backend.transpose(a), g_c)
            lora["A"] = a - config.learning_rate * grad_a
            lora["B"] = b - config.learning_rate * grad_b
        else:
            weights["W"] = w - config.learning_rate * grad_w

        return loss

    def _save_checkpoint(
        self,
        job_id: str,
        step: int,
        loss: float,
        weights: dict[str, Any],
        lora: dict[str, Any] | None,
    ) -> str:
        checkpoint_dir = self.paths.base / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{job_id}-step-{step}.npz"
        payload = {"W": self.backend.to_numpy(weights["W"])}
        if lora:
            payload["lora_A"] = self.backend.to_numpy(lora["A"])
            payload["lora_B"] = self.backend.to_numpy(lora["B"])
            payload["lora_scale"] = float(lora["scale"])
        np.savez(path, **payload)
        return str(path)

    def _load_dataset(self, path: str, batch_size: int) -> list[np.ndarray]:
        resolved = expand_path(path)
        data: list[list[float]] = []
        with resolved.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = payload.get("text")
                if text is None and "messages" in payload:
                    text = " ".join(msg.get("content", "") for msg in payload["messages"])
                if not text:
                    continue
                data.append(self._vectorize_text(str(text)))

        if not data:
            data = [self._vectorize_text("empty")]

        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batches.append(np.array(batch, dtype=np.float32))
        return batches

    @staticmethod
    def _vectorize_text(text: str, dim: int = 32) -> list[float]:
        vector = [0.0] * dim
        for token in text.split():
            idx = hash(token) % dim
            vector[idx] += 1.0
        return vector

    @staticmethod
    def _available_memory_bytes() -> int:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (ValueError, AttributeError, OSError):
            return 0
