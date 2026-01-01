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

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


@dataclass(frozen=True)
class UnifiedMergeConfig:
    """
    Configuration for unified geometric merge.

    Transplant formula:
        W' = W_target + P_null(A_boundary) @ (W_source_aligned - W_target)

    Guarantee:
        A_boundary @ W' = A_boundary @ W_target  (boundary preserved)

    This was validated empirically (Phase 6-8 research) and theoretically
    (AlphaEdit, ICLR 2025 Outstanding Paper).
    """

    # Probe mode: "precise" (CKA on activations) or "fast" (weight-level CKA)
    probe_mode: Literal["precise", "fast"] = "precise"

    # Maximum probes in precise mode (0 = all 403)
    max_probes: int = 0

    # Transplant settings - REQUIRED for effective knowledge transfer
    # Core domains define what concepts to transplant (e.g., "mathematical")
    transplant_domains: tuple[str, ...] = ()
    # NOTE: Alpha was REMOVED. The null-space projection determines preserved_fraction
    # geometrically. For best results, do sequential single-domain transplants.

    # Output quantization (None = preserve original dtype)
    output_quant: str | None = None


@dataclass
class LayerMergeState:
    """State carried through layers during merge (zipper)."""

    # Current input rotation (from previous layer's output)
    omega_in: "Array | None" = None

    # Layer index
    layer_index: int = 0

    # Accumulated metrics
    procrustes_errors: list[float] = field(default_factory=list)
    spectral_ratios: list[float] = field(default_factory=list)
    effective_alphas: list[float] = field(default_factory=list)


@dataclass
class UnifiedMergeResult:
    """Result of unified geometric merge."""

    merged_weights: dict[str, "Array"]

    # Per-stage metrics
    vocab_metrics: dict[str, Any]  # Stage 0: Vocabulary alignment
    probe_metrics: dict[str, Any]  # Stage 1: Probe
    permute_metrics: dict[str, Any]  # Stage 2: Git Re-Basin permutation
    transplant_metrics: dict[str, Any]  # Stage 3: Transplant

    # Overall quality
    mean_confidence: float
    mean_procrustes_error: float
    layer_count: int
    weight_count: int

    # Timing
    timestamp: datetime

    # Merge strategy used
    merge_strategy: str = "transplant"

    # Optional fields (must come after required fields)
    # Output path (if saved)
    output_path: str | None = None

    # Vocabulary alignment status
    vocab_aligned: bool = False

    # Stage 6: Validation metrics (raw measurements)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    # safety_verdict field was REMOVED - verdicts are subjective
    # Use geometry_metrics["mean_preserved_fraction"] for the raw measurement
    refusal_preserved: bool = True

    # Geometric confidence signals (raw measurements, no interpretation)
    # Contains: mean_preserved_fraction, mean_cka_after, mean_projection_loss,
    # transplant_ratio, and component signals from curvature alignment
    geometry_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossArchitectureInfo:
    """Information about cross-architecture model pair."""

    is_cross_architecture: bool = False
    source_layer_count: int = 0
    target_layer_count: int = 0
    source_hidden_dim: int = 0
    target_hidden_dim: int = 0
    layer_correspondence: dict[int, int] | None = None


@dataclass
class LayerGeometry:
    """Complete geometric analysis of a single layer."""

    layer_idx: int

    # Dimension analysis (Stage 2)
    intrinsic_dimension: float = 0.0
    manifold_dimension: int = 0
    curvature: float = 0.0  # Sectional curvature

    # Ollivier-Ricci curvature (Stage 2) - raw geometric measurements
    ollivier_ricci_mean: float = 0.0  # Mean edge curvature (negative = hyperbolic)
    ollivier_ricci_std: float = 0.0  # Std deviation of edge curvatures

    # Shared structure (Stage 3)
    shared_dimension: int = 0
    source_projection: "Array | None" = None
    target_projection: "Array | None" = None
    alignment_strengths: list[float] = field(default_factory=list)
    relative_rep_error: float = 0.0

    # Alignment (Stage 4)
    procrustes_rotation: "Array | None" = None
    permutation_matrix: "Array | None" = None
    alignment_quality: float = 0.0

    # Gromov-Wasserstein (Stage 2)
    gw_distance: float = 0.0
    gw_coupling: "Array | None" = None  # Transport plan for neuron correspondence

    # Interference (Stage 5)
    interference_score: float = 0.0
    wudi_loss: float = 0.0
    wudi_mean_overlap: float = 0.0
    wudi_max_overlap: float = 0.0
    transform_requirements: list[str] = field(default_factory=list)
    null_space_dim: int = 0
    null_space_projection: "Array | None" = None
    spectral_condition: float = 0.0

    # Dimension weights (Stage 6)
    dimension_alphas: "Array | None" = None
    fisher_weights: "Array | None" = None
    source_fisher: "Array | None" = None
    target_fisher: "Array | None" = None
    fisher_method: str = ""
    verb_noun_mask: "Array | None" = None
    refinement_score: float = 0.0

    # Blending (Stage 7)
    base_alpha: float = 0.5
    smoothed_alpha: float = 0.5
    sparsity_mask: "Array | None" = None


@dataclass
class MergeGeometry:
    """Complete geometric analysis for a merge operation."""

    source_model: str
    target_model: str
    layer_geometries: dict[int, LayerGeometry] = field(default_factory=dict)

    # Global metrics
    overall_cka: float = 0.0
    overall_gw_distance: float = 0.0
    mean_shared_dimension: float = 0.0
    mean_intrinsic_dimension: float = 0.0

    # Cross-architecture support
    is_cross_architecture: bool = False
    layer_correspondence: dict[int, int] | None = None  # source_layer -> target_layer
    alignment_quality: float = 0.0  # Quality of layer correspondence

    # Safety
    refusal_preserved: bool = True
    safety_score: float = 1.0

    # Ollivier-Ricci curvature (raw geometric measurement)
    mean_ollivier_ricci: float = 0.0

    # Curvature alignment (for merge confidence)
    # Score 0.0-1.0: how aligned are source/target curvature profiles
    curvature_alignment: float = 0.0
    curvature_alignment_details: dict[str, float] = field(default_factory=dict)
