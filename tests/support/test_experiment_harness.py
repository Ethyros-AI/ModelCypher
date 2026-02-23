# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json

import pytest

from modelcypher.core.support.experiment_harness import (
    DISPROVEN,
    HOLD,
    INCONCLUSIVE,
    PROMOTE,
    DecisionResult,
    ExperimentRun,
)


class TestDecisionResult:
    def test_valid_statuses(self) -> None:
        for status in (PROMOTE, HOLD, INCONCLUSIVE, DISPROVEN):
            d = DecisionResult(experiment_id="G1", module="test", status=status)
            assert d.status == status

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            DecisionResult(experiment_id="G1", module="test", status="INVALID")


class TestExperimentRun:
    def test_full_workflow(self, tmp_path) -> None:
        """Full lifecycle: start → log config → log metric → log decision."""
        with ExperimentRun("G1", "350M-700M", "math", results_root=tmp_path) as run:
            run.log_config({"legacy_constant": 5.0, "method": "changepoint"})
            run.log_metric({"step": 1, "value": 0.42})
            run.log_metric({"step": 2, "value": 0.55})
            run.log_bootstrap({"ci_lower": 3, "ci_upper": 5})
            run.log_decision(
                DecisionResult(
                    experiment_id="G1",
                    module="variance_concentration",
                    status=PROMOTE,
                    resolved_domains=["math"],
                    metric_deltas={"transfer_strength": 0.02},
                )
            )

        # Verify artifacts exist
        config = json.loads((run.run_dir / "config.json").read_text())
        assert config["legacy_constant"] == 5.0
        assert config["experiment_id"] == "G1"

        # Verify raw_metrics.jsonl has 2 lines
        lines = (run.run_dir / "raw_metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1

        # Verify bootstrap
        bootstrap = json.loads((run.run_dir / "bootstrap_metrics.json").read_text())
        assert bootstrap["ci_lower"] == 3

        # Verify decision
        decision = json.loads((run.run_dir / "decision.json").read_text())
        assert decision["status"] == "PROMOTE"
        assert decision["resolved_domains"] == ["math"]

    def test_not_started_raises(self, tmp_path) -> None:
        run = ExperimentRun("G1", results_root=tmp_path)
        with pytest.raises(RuntimeError, match="start"):
            run.log_config({})

    def test_directory_structure(self, tmp_path) -> None:
        with ExperimentRun("G2", "1.2B-8B", "code", results_root=tmp_path) as run:
            run.log_config({})

        # Verify directory has experiment_id/model_pair/domain/timestamp structure
        parts = run.run_dir.relative_to(tmp_path).parts
        assert parts[0] == "G2"
        assert parts[1] == "1.2B-8B"
        assert parts[2] == "code"
        assert len(parts) == 4  # timestamp directory

    def test_cross_basis_and_precision(self, tmp_path) -> None:
        with ExperimentRun("G1", results_root=tmp_path) as run:
            run.log_cross_basis({"rotation": "random_orthogonal", "cka": 0.99})
            run.log_cross_precision({"fp32": 0.98, "bf16": 0.97})

        cb = json.loads((run.run_dir / "cross_basis_metrics.json").read_text())
        assert cb["rotation"] == "random_orthogonal"
        cp = json.loads((run.run_dir / "cross_precision_metrics.json").read_text())
        assert cp["fp32"] == 0.98
