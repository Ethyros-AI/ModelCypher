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

"""EXPERIMENTAL: LoRA Geometry Validation Framework.

This module implements measurement studies for understanding the relationship
between weight-space geometry and output-space behavior in LoRA adapters.

This is research code - NOT validated for production use.
See: docs/research/lora_geometry_validation_plan.md
"""

from modelcypher.experimental.lora_geometry.measurements import (
    AdapterMeasurement,
    LayerMeasurement,
    collect_layer_measurements,
    collect_adapter_measurement,
)
from modelcypher.experimental.lora_geometry.statistics import (
    CorrelationResult,
    PermutationTestResult,
    BootstrapCI,
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_bootstrap_ci,
    compute_permutation_test,
)
from modelcypher.experimental.lora_geometry.four_condition import (
    ConditionType,
    create_synthetic_adapter,
    FourConditionExperiment,
)
from modelcypher.experimental.lora_geometry.subspace_analysis import (
    SubspaceOverlapResult,
    compute_principal_angles,
    compute_spectral_overlap,
    compute_behavioral_overlap,
)
from modelcypher.experimental.lora_geometry.id_trajectory import (
    IDTrajectoryPoint,
    IDTrajectory,
    IDTrajectoryTracker,
    measure_id_at_checkpoint,
)

__all__ = [
    # Measurements
    "AdapterMeasurement",
    "LayerMeasurement",
    "collect_layer_measurements",
    "collect_adapter_measurement",
    # Statistics
    "CorrelationResult",
    "PermutationTestResult",
    "BootstrapCI",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    "compute_bootstrap_ci",
    "compute_permutation_test",
    # Four-condition
    "ConditionType",
    "create_synthetic_adapter",
    "FourConditionExperiment",
    # Subspace analysis
    "SubspaceOverlapResult",
    "compute_principal_angles",
    "compute_spectral_overlap",
    "compute_behavioral_overlap",
    # ID trajectory
    "IDTrajectoryPoint",
    "IDTrajectory",
    "IDTrajectoryTracker",
    "measure_id_at_checkpoint",
]
