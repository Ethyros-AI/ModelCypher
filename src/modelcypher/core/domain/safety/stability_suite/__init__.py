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

"""Stability Suite module.

Provides types and prompt batteries for running stability evaluations on
adapters and checkpoints. The Stability Suite tests model stability across
baseline tasks, identity probes, jailbreak resistance, structured output,
and domain shifts.
"""

from modelcypher.core.domain.safety.stability_suite.stability_suite_models import (
    ActionSchemaResult,
    AggregateMetrics,
    DistributionSummary,
    EntropyManifoldSummary,
    HistogramBin,
    PromptResult,
    StabilitySuiteGenerationConfig,
    StabilitySuiteProgress,
    StabilitySuitePrompt,
    StabilitySuitePromptCategory,
    StabilitySuiteReport,
    StabilitySuiteReportSummary,
    StabilitySuiteRunRequest,
    StabilitySuiteTarget,
    StabilitySuiteTargetKind,
    StabilitySuiteTier,
)
from modelcypher.core.domain.safety.stability_suite.stability_suite_prompt_battery import (
    StabilitySuitePromptBattery,
)

__all__ = [
    "ActionSchemaResult",
    "AggregateMetrics",
    "DistributionSummary",
    "EntropyManifoldSummary",
    "HistogramBin",
    "PromptResult",
    "StabilitySuiteGenerationConfig",
    "StabilitySuiteProgress",
    "StabilitySuitePrompt",
    "StabilitySuitePromptBattery",
    "StabilitySuitePromptCategory",
    "StabilitySuiteReport",
    "StabilitySuiteReportSummary",
    "StabilitySuiteRunRequest",
    "StabilitySuiteTarget",
    "StabilitySuiteTargetKind",
    "StabilitySuiteTier",
]
