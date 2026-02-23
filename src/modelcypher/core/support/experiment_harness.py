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

"""Experiment harness for geometry-only hard mode experiments.

Manages artifact emission per the registered schema defined in
``docs/research/geometry_only_hard_mode_experiment_matrix.md``.

Each experiment run produces:
- ``config.json`` — experiment configuration
- ``raw_metrics.jsonl`` — per-step raw metrics (append-only)
- ``bootstrap_metrics.json`` — bootstrap CI results
- ``decision.json`` — promotion decision with status and rationale

Results are stored at:
``results/geometry_only/<experiment_id>/<model_pair>/<domain>/<timestamp>/``

Pure Python — zero framework dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# Decision statuses per experiment matrix §6
PROMOTE = "PROMOTE"
HOLD = "HOLD"
INCONCLUSIVE = "INCONCLUSIVE"
DISPROVEN = "DISPROVEN"

VALID_STATUSES = {PROMOTE, HOLD, INCONCLUSIVE, DISPROVEN}


@dataclass
class DecisionResult:
    """Promotion decision for a geometry-only experiment."""

    experiment_id: str
    module: str
    status: str
    resolved_domains: list[str] = field(default_factory=list)
    unresolved_domains: list[str] = field(default_factory=list)
    metric_deltas: dict[str, float] = field(default_factory=dict)
    ci_width_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_mode: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}', must be one of {VALID_STATUSES}"
            )


class ExperimentRun:
    """Context for a single experiment run with artifact management.

    Usage::

        with ExperimentRun("G1", "350M-700M", "math") as run:
            run.log_config({"legacy_constant": 5.0, "method": "changepoint"})
            run.log_metric({"step": 1, "rss_reduction": 0.42})
            run.log_bootstrap({"ci_lower": 3, "ci_upper": 5})
            run.log_decision(DecisionResult(
                experiment_id="G1",
                module="variance_concentration",
                status=PROMOTE,
                metric_deltas={"transfer_strength": 0.02},
            ))
    """

    def __init__(
        self,
        experiment_id: str,
        model_pair: str = "default",
        domain: str = "default",
        results_root: str | Path = "results/geometry_only",
    ) -> None:
        self.experiment_id = experiment_id
        self.model_pair = model_pair
        self.domain = domain

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = (
            Path(results_root) / experiment_id / model_pair / domain / timestamp
        )
        self._metrics_file: Path | None = None
        self._started = False

    def start(self) -> None:
        """Create run directory and initialize artifact files."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.run_dir / "raw_metrics.jsonl"
        self._started = True

    def log_config(self, config: dict[str, Any]) -> None:
        """Write experiment configuration."""
        self._ensure_started()
        config_with_meta = {
            "experiment_id": self.experiment_id,
            "model_pair": self.model_pair,
            "domain": self.domain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **config,
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(config_with_meta, indent=2, default=str) + "\n"
        )

    def log_metric(self, metric: dict[str, Any]) -> None:
        """Append a raw metric line to the JSONL file."""
        self._ensure_started()
        assert self._metrics_file is not None
        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(metric, default=str) + "\n")

    def log_bootstrap(self, bootstrap_data: dict[str, Any]) -> None:
        """Write bootstrap CI results."""
        self._ensure_started()
        (self.run_dir / "bootstrap_metrics.json").write_text(
            json.dumps(bootstrap_data, indent=2, default=str) + "\n"
        )

    def log_cross_basis(self, data: dict[str, Any]) -> None:
        """Write cross-basis check results."""
        self._ensure_started()
        (self.run_dir / "cross_basis_metrics.json").write_text(
            json.dumps(data, indent=2, default=str) + "\n"
        )

    def log_cross_precision(self, data: dict[str, Any]) -> None:
        """Write cross-precision check results."""
        self._ensure_started()
        (self.run_dir / "cross_precision_metrics.json").write_text(
            json.dumps(data, indent=2, default=str) + "\n"
        )

    def log_decision(self, decision: DecisionResult) -> None:
        """Write the promotion decision."""
        self._ensure_started()
        (self.run_dir / "decision.json").write_text(
            json.dumps(asdict(decision), indent=2, default=str) + "\n"
        )

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("ExperimentRun.start() must be called first")

    def __enter__(self) -> "ExperimentRun":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
