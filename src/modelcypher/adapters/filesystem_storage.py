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

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from modelcypher.core.domain.models import (
    CheckpointRecord,
    CompareCheckpointResult,
    CompareSession,
    EvaluationResult,
    ModelInfo,
    TrainingJob,
)
from modelcypher.core.domain.training import TrainingStatus
from modelcypher.ports.storage import (
    CompareStore,
    EvaluationStore,
    JobStore,
    ModelStore,
)
from modelcypher.utils.locks import FileLock
from modelcypher.utils.paths import ensure_dir, expand_path


class StoragePaths:
    def __init__(self) -> None:
        base = Path(os.environ.get("MODELCYPHER_HOME", "~/.modelcypher"))
        self.base = ensure_dir(base)
        self.models = self.base / "models.json"
        self.jobs = ensure_dir(self.base / "jobs")
        self.checkpoints = self.base / "checkpoints.json"
        self.evaluations = self.base / "evaluations.json"
        self.comparisons = self.base / "comparisons.json"
        self.logs = ensure_dir(self.base / "logs")


class FileSystemStore(ModelStore, JobStore, EvaluationStore, CompareStore):
    def __init__(self, paths: StoragePaths | None = None) -> None:
        self.paths = paths or StoragePaths()

    def list_models(self) -> list[ModelInfo]:
        """List all registered models.

        Returns
        -------
        list of ModelInfo
            All registered model records.
        """
        payload = self._read_list(self.paths.models)
        return [self._model_from_dict(item) for item in payload]

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get model by ID or alias.

        Parameters
        ----------
        model_id : str
            Model ID or alias to look up.

        Returns
        -------
        ModelInfo or None
            Model record if found, None otherwise.
        """
        return next(
            (m for m in self.list_models() if m.id == model_id or m.alias == model_id), None
        )

    def register_model(self, model: ModelInfo) -> None:
        """Register a new model or update existing registration.

        Parameters
        ----------
        model : ModelInfo
            Model record to register.
        """
        with FileLock(self._lock_path(self.paths.models)):
            models = self.list_models()
            models = [m for m in models if m.id != model.id and m.alias != model.alias]
            models.append(model)
            self._write_list(self.paths.models, [self._model_to_dict(m) for m in models])

    def delete_model(self, model_id: str) -> None:
        """Delete model registration by ID or alias.

        Parameters
        ----------
        model_id : str
            Model ID or alias to delete.
        """
        with FileLock(self._lock_path(self.paths.models)):
            models = [m for m in self.list_models() if m.id != model_id and m.alias != model_id]
            self._write_list(self.paths.models, [self._model_to_dict(m) for m in models])

    def save_job(self, job: TrainingJob) -> None:
        """Save training job to storage.

        Parameters
        ----------
        job : TrainingJob
            Training job record to save.
        """
        path = self.paths.jobs / f"{job.job_id}.json"
        self._write_json(path, self._job_to_dict(job))

    def update_job(self, job: TrainingJob) -> None:
        """Update existing training job.

        Parameters
        ----------
        job : TrainingJob
            Training job record to update.
        """
        self.save_job(job)

    def list_jobs(
        self, status: TrainingStatus | None = None, active_only: bool = False
    ) -> list[TrainingJob]:
        """List training jobs with optional filtering.

        Parameters
        ----------
        status : TrainingStatus or None
            Filter by specific status if provided.
        active_only : bool
            If True, return only pending, running, or paused jobs.

        Returns
        -------
        list of TrainingJob
            Filtered list of training jobs.
        """
        jobs = []
        for job_file in sorted(self.paths.jobs.glob("*.json")):
            payload = self._read_json(job_file)
            if payload:
                jobs.append(self._job_from_dict(payload))
        if status:
            jobs = [job for job in jobs if job.status == status]
        if active_only:
            jobs = [
                job
                for job in jobs
                if job.status
                in {TrainingStatus.pending, TrainingStatus.running, TrainingStatus.paused}
            ]
        return jobs

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get training job by ID.

        Parameters
        ----------
        job_id : str
            Job ID to look up.

        Returns
        -------
        TrainingJob or None
            Job record if found, None otherwise.
        """
        path = self.paths.jobs / f"{job_id}.json"
        if not path.exists():
            return None
        return self._job_from_dict(self._read_json(path))

    def delete_job(self, job_id: str) -> None:
        """Delete training job by ID.

        Parameters
        ----------
        job_id : str
            Job ID to delete.
        """
        path = self.paths.jobs / f"{job_id}.json"
        if path.exists():
            path.unlink()

    def list_checkpoints(self, job_id: str | None = None) -> list[CheckpointRecord]:
        """List training checkpoints with optional job filtering.

        Parameters
        ----------
        job_id : str or None
            Filter by specific job ID if provided.

        Returns
        -------
        list of CheckpointRecord
            List of checkpoint records.
        """
        payload = self._read_list(self.paths.checkpoints)
        checkpoints = [self._checkpoint_from_dict(item) for item in payload]
        if job_id:
            checkpoints = [c for c in checkpoints if c.job_id == job_id]
        return checkpoints

    def add_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        """Add training checkpoint to storage.

        Parameters
        ----------
        checkpoint : CheckpointRecord
            Checkpoint record to add.
        """
        with FileLock(self._lock_path(self.paths.checkpoints)):
            checkpoints = self.list_checkpoints()
            checkpoints.append(checkpoint)
            self._write_list(
                self.paths.checkpoints, [self._checkpoint_to_dict(c) for c in checkpoints]
            )

    def delete_checkpoint(self, path: str) -> None:
        """Delete checkpoint by path.

        Parameters
        ----------
        path : str
            Checkpoint file path to delete.
        """
        with FileLock(self._lock_path(self.paths.checkpoints)):
            checkpoints = [c for c in self.list_checkpoints() if c.file_path != path]
            self._write_list(
                self.paths.checkpoints, [self._checkpoint_to_dict(c) for c in checkpoints]
            )
        resolved = expand_path(path)
        if resolved.exists():
            resolved.unlink()

    def list_evaluations(self, limit: int) -> list[EvaluationResult]:
        """List evaluation results, most recent first.

        Parameters
        ----------
        limit : int
            Maximum number of results to return.

        Returns
        -------
        list of EvaluationResult
            List of evaluation results, newest first.
        """
        payload = self._read_list(self.paths.evaluations)
        evaluations = [self._evaluation_from_dict(item) for item in payload]
        evaluations.sort(key=lambda e: e.timestamp, reverse=True)
        return evaluations[:limit]

    def save_evaluation(self, result: EvaluationResult) -> None:
        """Save evaluation result to storage.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to save.
        """
        with FileLock(self._lock_path(self.paths.evaluations)):
            evaluations = self._read_list(self.paths.evaluations)
            evaluations = [item for item in evaluations if item.get("id") != result.id]
            evaluations.append(self._evaluation_to_dict(result))
            self._write_list(self.paths.evaluations, evaluations)

    def get_evaluation(self, eval_id: str) -> EvaluationResult | None:
        """Get evaluation result by ID.

        Parameters
        ----------
        eval_id : str
            Evaluation ID to look up.

        Returns
        -------
        EvaluationResult or None
            Evaluation result if found, None otherwise.
        """
        evaluations = self._read_list(self.paths.evaluations)
        for item in evaluations:
            if item.get("id") == eval_id:
                return self._evaluation_from_dict(item)
        return None

    def list_sessions(self, limit: int, status: str | None = None) -> list[CompareSession]:
        """List comparison sessions with optional status filtering.

        Parameters
        ----------
        limit : int
            Maximum number of sessions to return.
        status : str or None
            Filter by specific status if provided.

        Returns
        -------
        list of CompareSession
            List of comparison sessions, newest first.
        """
        payload = self._read_list(self.paths.comparisons)
        sessions = [self._compare_from_dict(item) for item in payload]
        if status:
            sessions = [session for session in sessions if session.config.get("status") == status]
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def save_session(self, session: CompareSession) -> None:
        """Save comparison session to storage.

        Parameters
        ----------
        session : CompareSession
            Comparison session to save.
        """
        with FileLock(self._lock_path(self.paths.comparisons)):
            sessions = self._read_list(self.paths.comparisons)
            sessions = [item for item in sessions if item.get("id") != session.id]
            sessions.append(self._compare_to_dict(session))
            self._write_list(self.paths.comparisons, sessions)

    def get_session(self, session_id: str) -> CompareSession | None:
        """Get comparison session by ID.

        Parameters
        ----------
        session_id : str
            Session ID to look up.

        Returns
        -------
        CompareSession or None
            Comparison session if found, None otherwise.
        """
        sessions = self._read_list(self.paths.comparisons)
        for item in sessions:
            if item.get("id") == session_id:
                return self._compare_from_dict(item)
        return None

    def _read_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return self._read_json(path) or []

    def _write_list(self, path: Path, payload: list[dict]) -> None:
        self._write_json(path, payload)

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1_000_000)}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _lock_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    @staticmethod
    def _to_iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    @staticmethod
    def _from_iso(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    def _model_to_dict(self, model: ModelInfo) -> dict:
        payload = asdict(model)
        payload["created_at"] = self._to_iso(model.created_at)
        return payload

    def _model_from_dict(self, payload: dict) -> ModelInfo:
        return ModelInfo(
            id=payload["id"],
            alias=payload["alias"],
            architecture=payload["architecture"],
            format=payload["format"],
            path=payload["path"],
            size_bytes=payload.get("size_bytes", 0),
            parameter_count=payload.get("parameter_count"),
            is_default_chat=payload.get("is_default_chat", False),
            created_at=self._from_iso(payload.get("created_at")) or datetime.utcnow(),
        )

    def _checkpoint_to_dict(self, checkpoint: CheckpointRecord) -> dict:
        payload = asdict(checkpoint)
        payload["timestamp"] = self._to_iso(checkpoint.timestamp)
        return payload

    def _checkpoint_from_dict(self, payload: dict) -> CheckpointRecord:
        return CheckpointRecord(
            job_id=payload["job_id"],
            step=int(payload["step"]),
            loss=float(payload["loss"]),
            timestamp=self._from_iso(payload.get("timestamp")) or datetime.utcnow(),
            file_path=payload["file_path"],
        )

    def _job_to_dict(self, job: TrainingJob) -> dict:
        payload = asdict(job)
        payload["status"] = job.status.value
        payload["created_at"] = self._to_iso(job.created_at)
        payload["updated_at"] = self._to_iso(job.updated_at)
        payload["started_at"] = self._to_iso(job.started_at)
        payload["completed_at"] = self._to_iso(job.completed_at)
        config_payload = payload.get("config")
        if isinstance(config_payload, dict):
            level = config_payload.get("geometric_instrumentation_level")
            if hasattr(level, "value"):
                config_payload["geometric_instrumentation_level"] = level.value
            payload["config"] = config_payload
        return payload

    def _job_from_dict(self, payload: dict) -> TrainingJob:
        return TrainingJob(
            job_id=payload["job_id"],
            status=TrainingStatus(payload["status"]),
            model_id=payload["model_id"],
            dataset_path=payload["dataset_path"],
            created_at=self._from_iso(payload.get("created_at")) or datetime.utcnow(),
            updated_at=self._from_iso(payload.get("updated_at")) or datetime.utcnow(),
            started_at=self._from_iso(payload.get("started_at")),
            completed_at=self._from_iso(payload.get("completed_at")),
            current_step=payload.get("current_step", 0),
            total_steps=payload.get("total_steps", 0),
            current_epoch=payload.get("current_epoch", 0),
            total_epochs=payload.get("total_epochs", 0),
            loss=payload.get("loss"),
            learning_rate=payload.get("learning_rate"),
            config=payload.get("config"),
            checkpoints=payload.get("checkpoints"),
            loss_history=payload.get("loss_history"),
            metrics=payload.get("metrics"),
            metrics_history=payload.get("metrics_history"),
        )

    def _evaluation_to_dict(self, result: EvaluationResult) -> dict:
        payload = asdict(result)
        payload["timestamp"] = self._to_iso(result.timestamp)
        return payload

    def _evaluation_from_dict(self, payload: dict) -> EvaluationResult:
        return EvaluationResult(
            id=payload["id"],
            model_path=payload["model_path"],
            model_name=payload["model_name"],
            dataset_path=payload["dataset_path"],
            dataset_name=payload["dataset_name"],
            average_loss=payload["average_loss"],
            perplexity=payload["perplexity"],
            sample_count=payload["sample_count"],
            timestamp=self._from_iso(payload.get("timestamp")) or datetime.utcnow(),
            config=payload.get("config", {}),
            sample_results=payload.get("sample_results", []),
            adapter_path=payload.get("adapter_path"),
        )

    def _compare_to_dict(self, session: CompareSession) -> dict:
        payload = asdict(session)
        payload["created_at"] = self._to_iso(session.created_at)
        return payload

    def _compare_from_dict(self, payload: dict) -> CompareSession:
        checkpoints = [CompareCheckpointResult(**item) for item in payload.get("checkpoints", [])]
        return CompareSession(
            id=payload["id"],
            created_at=self._from_iso(payload.get("created_at")) or datetime.utcnow(),
            prompt=payload.get("prompt", ""),
            config=payload.get("config", {}),
            checkpoints=checkpoints,
            notes=payload.get("notes"),
            tags=payload.get("tags"),
        )
