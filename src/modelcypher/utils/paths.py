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

import os
from pathlib import Path


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = expand_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_modelcypher_home() -> Path:
    """Get the ModelCypher home directory, creating it if needed."""
    base = Path(os.environ.get("MODELCYPHER_HOME", "~/.modelcypher"))
    return ensure_dir(base)


def get_jobs_dir() -> Path:
    """Get the jobs directory, creating it if necessary.

    Returns:
        Path to $MODELCYPHER_HOME/jobs (defaults to ~/.modelcypher/jobs)
    """
    return ensure_dir(get_modelcypher_home() / "jobs")


def get_logs_dir() -> Path:
    """Get the logs directory, creating it if necessary.

    Returns:
        Path to ~/.modelcypher/logs
    """
    return ensure_dir(get_modelcypher_home() / "logs")


def get_merge_log_path(pipeline_id: str | None = None) -> Path:
    """Get a log file path for a merge operation.

    Args:
        pipeline_id: Optional pipeline ID. If None, generates timestamp-based name.

    Returns:
        Path to log file (e.g., ~/.modelcypher/logs/merge-2025-01-16-123456.log)
    """
    from datetime import datetime

    logs_dir = get_logs_dir()
    if pipeline_id:
        filename = f"merge-{pipeline_id}.log"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"merge-{timestamp}.log"
    return logs_dir / filename
