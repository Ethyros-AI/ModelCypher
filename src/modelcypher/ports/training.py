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

from modelcypher.core.domain.models import TrainingJob
from modelcypher.core.domain.training import PreflightResult, TrainingConfig


@runtime_checkable
class TrainingEngine(Protocol):
    """Port for running training jobs on a backend."""

    def preflight(self, config: TrainingConfig) -> PreflightResult:
        """Validate training configuration and return a preflight report."""
        ...
    def start(
        self, config: TrainingConfig, stream_events: bool = False
    ) -> tuple[TrainingJob, list[dict]]:
        """Start a training job and return the job plus any initial events."""
        ...
    def status(self, job_id: str) -> TrainingJob:
        """Return the current status for a job."""
        ...
    def pause(self, job_id: str) -> TrainingJob:
        """Pause a running training job."""
        ...
    def resume(self, job_id: str) -> TrainingJob:
        """Resume a paused training job."""
        ...
    def cancel(self, job_id: str) -> TrainingJob:
        """Cancel a training job."""
        ...
    def logs(self, job_id: str, tail: int = 100) -> list[str]:
        """Return the latest log lines for a job."""
        ...
