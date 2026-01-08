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

"""Evaluation execution engine for semantic evaluation scenarios.

This package provides:
- EvaluationExecutionEngine: Orchestrates scenario-based model evaluation
- EvaluationScenario: Definition of evaluation scenarios with prompts and target concepts
- EvaluationConfig: Configuration for evaluation runs
"""

from .engine import (
    EntropyFn,
    EvaluationConfig,
    EvaluationExecutionEngine,
    EvaluationScenario,
    InferenceFn,
    MetricType,
    PromptResult,
    ScenarioResult,
    ScoringFn,
)

__all__ = [
    "EntropyFn",
    "EvaluationConfig",
    "EvaluationExecutionEngine",
    "EvaluationScenario",
    "InferenceFn",
    "MetricType",
    "PromptResult",
    "ScenarioResult",
    "ScoringFn",
]
