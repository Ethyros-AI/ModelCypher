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

from typing import Protocol, runtime_checkable

from modelcypher.core.domain.geometry.manifold_profile import ManifoldPoint, ManifoldProfile
from modelcypher.core.domain.models import (
    CheckpointRecord,
    CompareSession,
    EvaluationResult,
    ModelInfo,
    TrainingJob,
)
from modelcypher.core.domain.training import TrainingStatus


@runtime_checkable
class ModelStore(Protocol):
    """Port for model registry storage."""

    def list_models(self) -> list[ModelInfo]:
        """Return all registered models."""
        ...
    def get_model(self, model_id: str) -> ModelInfo | None:
        """Lookup a model by ID or alias."""
        ...
    def register_model(self, model: ModelInfo) -> None:
        """Create or update a model record."""
        ...
    def delete_model(self, model_id: str) -> None:
        """Remove a model record by ID or alias."""
        ...


@runtime_checkable
class JobStore(Protocol):
    """Port for training job persistence."""

    def save_job(self, job: TrainingJob) -> None:
        """Persist a new training job."""
        ...
    def update_job(self, job: TrainingJob) -> None:
        """Update an existing training job."""
        ...
    def list_jobs(
        self, status: TrainingStatus | None = None, active_only: bool = False
    ) -> list[TrainingJob]:
        """Return jobs filtered by status or active-only flag."""
        ...
    def get_job(self, job_id: str) -> TrainingJob | None:
        """Fetch a training job by ID."""
        ...
    def delete_job(self, job_id: str) -> None:
        """Delete a training job by ID."""
        ...
    def list_checkpoints(self, job_id: str | None = None) -> list[CheckpointRecord]:
        """List checkpoints, optionally filtered by job ID."""
        ...
    def add_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        """Persist a checkpoint record."""
        ...
    def delete_checkpoint(self, path: str) -> None:
        """Delete a checkpoint by path."""
        ...


@runtime_checkable
class EvaluationStore(Protocol):
    """Port for evaluation result persistence."""

    def list_evaluations(self, limit: int) -> list[EvaluationResult]:
        """List recent evaluation results."""
        ...
    def save_evaluation(self, result: EvaluationResult) -> None:
        """Persist an evaluation result."""
        ...
    def get_evaluation(self, eval_id: str) -> EvaluationResult | None:
        """Fetch an evaluation result by ID."""
        ...


@runtime_checkable
class CompareStore(Protocol):
    """Port for comparison session storage."""

    def list_sessions(self, limit: int, status: str | None = None) -> list[CompareSession]:
        """List comparison sessions, optionally filtered by status."""
        ...
    def save_session(self, session: CompareSession) -> None:
        """Persist a comparison session."""
        ...
    def get_session(self, session_id: str) -> CompareSession | None:
        """Fetch a comparison session by ID."""
        ...


@runtime_checkable
class ManifoldProfileStore(Protocol):
    """Port for manifold profile persistence."""

    def load(self, model_id: str) -> ManifoldProfile | None:
        """Load a manifold profile by model ID."""
        ...
    def list(self, limit: int | None = None) -> list[ManifoldProfile]:
        """List manifold profiles, optionally limited."""
        ...
    def save(self, profile: ManifoldProfile) -> None:
        """Persist a manifold profile."""
        ...
    def delete(self, model_id: str) -> None:
        """Delete a manifold profile by model ID."""
        ...
    def add_point(self, point: ManifoldPoint, model_id: str, model_name: str) -> None:
        """Append a manifold point to a profile."""
        ...
    def get_statistics(self, model_id: str) -> ManifoldProfile.Statistics | None:
        """Fetch cached statistics for a profile."""
        ...
