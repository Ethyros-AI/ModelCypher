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

"""Orthogonal probe generation for alignment coverage.

This module is a backward-compatibility shim that re-exports all functionality
from the focused submodules:

- null_space: Null space computation and result types
- probe_generator: OrthogonalProbeGenerator and token finding
- trajectory_analysis: Trajectory collection and subspace analysis
- trajectory_projection: Trajectory tangent projection

All imports from this module continue to work as before.
"""

from __future__ import annotations

# Re-export everything from submodules for backward compatibility
from modelcypher.core.domain.geometry.null_space import (
    OrthogonalProbeResult,
    SequenceProbeResult,
    RankAugmentationResult,
    VarianceNullSpaceResult,
    compute_numerical_rank,
    compute_null_space_basis,
    compute_variance_null_space,
)

from modelcypher.core.domain.geometry.probe_generator import (
    OrthogonalProbeGenerator,
    validate_full_rank_coverage,
    score_tokens_for_null_space,
    find_null_space_tokens_closed_form,
    augment_rank_closed_form,
    find_null_space_sequences,
    find_null_space_texts,
)

from modelcypher.core.domain.geometry.trajectory_analysis import (
    TrajectoryResult,
    TrajectorySubspaceResult,
    collect_trajectory,
    compute_trajectory_subspace,
    compute_trajectory_null_space,
    collect_trajectories_batch,
    find_trajectory_null_space,
)

from modelcypher.core.domain.geometry.trajectory_projection import (
    TrajectoryTangentResult,
    compute_trajectory_tangent_null_space,
    project_delta_to_trajectory_tangent,
    project_delta_to_variance_null_space,
)

__all__ = [
    "OrthogonalProbeGenerator",
    "OrthogonalProbeResult",
    "SequenceProbeResult",
    "RankAugmentationResult",
    "compute_numerical_rank",
    "compute_null_space_basis",
    "validate_full_rank_coverage",
    "score_tokens_for_null_space",
    "find_null_space_tokens_closed_form",
    "augment_rank_closed_form",
    # Sequence-based probe generation (closed-form token selection)
    "find_null_space_sequences",
    "find_null_space_texts",
    # Trajectory-based null-space discovery
    "TrajectoryResult",
    "TrajectorySubspaceResult",
    "collect_trajectory",
    "compute_trajectory_subspace",
    "compute_trajectory_null_space",
    "collect_trajectories_batch",
    "find_trajectory_null_space",
    # Trajectory-tangent null-space projection
    "TrajectoryTangentResult",
    "compute_trajectory_tangent_null_space",
    "project_delta_to_trajectory_tangent",
    # Variance-based null-space (for full-trajectory cases)
    "VarianceNullSpaceResult",
    "compute_variance_null_space",
    "project_delta_to_variance_null_space",
]
