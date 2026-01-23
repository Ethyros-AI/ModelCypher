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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


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


@dataclass
class UnifiedMergeResult:
    """Result of unified geometric merge."""

    merged_weights: dict[str, "Array"]

    # Per-stage metrics
    probe_metrics: dict[str, Any]  # Stage 1: Probe
    permute_metrics: dict[str, Any]  # Stage 2: Git Re-Basin permutation
    transplant_metrics: dict[str, Any]  # Stage 3: Transplant

    # Overall geometry
    mean_preserved_fraction: float
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

    # Stage 6: Validation metrics (raw measurements)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    refusal_preserved: bool = True

    # Geometric confidence signals (raw measurements, no interpretation)
    # Contains: mean_preserved_fraction, mean_cka_after, mean_projection_loss,
    # transplant_ratio, and component signals from curvature alignment
    geometry_metrics: dict[str, Any] = field(default_factory=dict)

    # Density analysis metrics (Stage 2: DENSITY)
    # Contains: overall_source_density, overall_target_density, overall_opportunity,
    # positive_opportunity_count, nonpositive_opportunity_count, concepts_analyzed
    density_metrics: dict[str, Any] = field(default_factory=dict)

    # Post-merge density (measured AFTER merge to verify density increased)
    post_merge_density: float | None = None


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
class LayerSemanticProfile:
    """Geometric profile of layers based on measured intrinsic dimension.

    NO HEURISTICS. The geometry tells us:
    - Intrinsic dimension (ID) varies by layer
    - ID peaks at semantic layers, compresses at translation layers
    - The "elbows" in the ID curve mark transitions
    - Gram rank drops to 2-3 at the bottleneck (~50% depth)

    This profile stores MEASUREMENTS, not thresholds.
    The merge code uses these measurements directly.
    """

    # Per-layer intrinsic dimension (measured, not guessed)
    intrinsic_dimensions: dict[int, float] = field(default_factory=dict)

    # Per-layer Gram rank (measured)
    gram_ranks: dict[int, int] = field(default_factory=dict)

    # Layer 0 is always embedding (this is structural, not heuristic)
    embedding_layer: int = 0

    # Total layer count
    total_layers: int = 0

    # Bottleneck layer (where Gram rank is minimum) - MEASURED
    bottleneck_layer: int | None = None

    # Per-layer sparsity (measured from probe activations)
    layer_sparsity: dict[int, float] = field(default_factory=dict)
    sparse_layers: list[int] = field(default_factory=list)
    skip_layers: list[int] = field(default_factory=list)

    # Per-layer manifold boundary radii (measured via flood fill)
    # Maps layer_idx -> boundary radius (distance from centroid where coherence drops)
    # Small radius = at stability edge (don't touch), large radius = safe to transfer
    boundary_radii: dict[int, float] = field(default_factory=dict)

    def is_embedding_layer(self, layer_idx: int) -> bool:
        """Layer 0 is structurally the embedding layer."""
        return layer_idx == self.embedding_layer

    def get_intrinsic_dimension(self, layer_idx: int) -> float | None:
        """Get measured intrinsic dimension for a layer."""
        return self.intrinsic_dimensions.get(layer_idx)

    def get_gram_rank(self, layer_idx: int) -> int | None:
        """Get measured Gram rank for a layer."""
        return self.gram_ranks.get(layer_idx)

    def get_transfer_safety(self, layer_idx: int) -> float:
        """Return normalized transfer safety based on boundary radius.

        The boundary radius measures how far from the activation centroid
        we can perturb before coherence degrades (flood fill detection).

        Small radius = layer is at stability edge, no room for perturbation
        Large radius = layer has headroom, safe to transfer

        Returns:
            0.0 = don't touch (at stability edge)
            1.0 = fully safe (unconstrained)

        If boundary radii haven't been computed, returns 1.0 (permissive fallback).
        """
        if not self.boundary_radii:
            return 1.0  # Fallback if not computed

        max_radius = max(self.boundary_radii.values())
        if max_radius <= 0:
            return 1.0  # Avoid division by zero

        layer_radius = self.boundary_radii.get(layer_idx, max_radius)
        return layer_radius / max_radius

    def compute_highway_layers(self) -> list[int]:
        """Identify highway layers based on intrinsic dimension.

        The semantic highway is where invariant geometry lives:
        - Layers with LOWEST intrinsic dimension (ID)
        - These are the semantic core - shared across all architectures
        - CKA = 1.0 is achievable here after alignment

        Entry/exit ramps (high ID) handle vocabulary-specific coordinate
        translation and should NOT be transplanted in cross-architecture merges.

        Algorithm:
        1. Find median ID across all layers
        2. Highway = layers where ID <= median
        3. Ramps = layers where ID > median (first/last layers typically)

        Returns:
            List of layer indices that are part of the semantic highway.
        """
        if not self.intrinsic_dimensions:
            # No ID data - return all layers as highway (safe fallback)
            return list(range(self.total_layers))

        id_values = sorted(self.intrinsic_dimensions.values())
        if len(id_values) < 3:
            return list(self.intrinsic_dimensions.keys())

        # Compute median ID as threshold
        mid_idx = len(id_values) // 2
        if len(id_values) % 2 == 0:
            median_id = (id_values[mid_idx - 1] + id_values[mid_idx]) / 2.0
        else:
            median_id = id_values[mid_idx]

        # Highway = layers with ID <= median (low ID = semantic core)
        highway = [
            layer_idx
            for layer_idx, id_val in self.intrinsic_dimensions.items()
            if id_val <= median_id
        ]

        return sorted(highway)

    def compute_ramp_layers(self) -> list[int]:
        """Identify ramp layers (translation layers) based on intrinsic dimension.

        Ramps are entry/exit layers that translate between:
        - 1D/2D token/embedding space
        - High-dimensional semantic manifold

        These layers are vocabulary-specific and architecture-tied.
        They should NOT be transplanted in cross-architecture merges.

        Returns:
            List of layer indices that are ramps (not highway).
        """
        if not self.intrinsic_dimensions:
            return []

        highway = set(self.compute_highway_layers())
        all_layers = set(self.intrinsic_dimensions.keys())
        ramps = all_layers - highway

        return sorted(ramps)

    def get_bottleneck_layer(self) -> int | None:
        """Return THE single bottleneck layer - minimum intrinsic dimension.

        This is the layer where information is most compressed:
        - Lowest intrinsic dimension = purest relational form
        - Universal across architectures (CKA=1.0 achievable)
        - The ONLY safe layer for cross-architecture transplant

        Layer 0 (embedding) is always excluded - it's structural.

        Returns:
            Layer index with minimum ID, or None if no ID data.
        """
        if not self.intrinsic_dimensions:
            return None

        # Find layer with minimum ID (excluding embedding layer 0)
        min_id = float("inf")
        bottleneck_layer = None

        for layer_idx, id_val in self.intrinsic_dimensions.items():
            if layer_idx == self.embedding_layer:
                continue  # Skip embedding layer
            if id_val < min_id:
                min_id = id_val
                bottleneck_layer = layer_idx

        return bottleneck_layer

    def compute_bottleneck_layers(self) -> list[int]:
        """Identify the true bottleneck - THE single most compressed layer.

        Returns a list for API compatibility, but contains only one layer.

        Returns:
            List with single bottleneck layer index, or empty if no ID data.
        """
        bottleneck = self.get_bottleneck_layer()
        return [bottleneck] if bottleneck is not None else []

    def set_cross_architecture_skip_layers(self) -> None:
        """Auto-populate skip_layers for cross-architecture merges.

        For cross-architecture, we're MUCH more conservative:
        - Only the bottleneck (minimum ID ± 10%) is safe to transplant
        - Everything else is translation layers (onramps/offramps)
        - Layer 0 (embedding) is always structural - never transplant

        The bottleneck is where the invariant relational structure lives.
        Both architectures compress to the same geometry there.
        That's the only place CKA=1.0 alignment is truly achievable.
        """
        # Get bottleneck layers (super highway)
        bottleneck = set(self.compute_bottleneck_layers())

        # Everything NOT in the bottleneck is a translation layer
        all_layers = set(self.intrinsic_dimensions.keys())
        translation_layers = all_layers - bottleneck

        # Also always skip embedding layer (structural)
        translation_layers.add(self.embedding_layer)

        # Combine with any existing skip layers
        existing = set(self.skip_layers or [])
        combined = existing.union(translation_layers)
        self.skip_layers = sorted(combined)


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
    alignment_cka: float = 0.0

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

    # Sparsity (for DARE - identifies droppable parameters)
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
    layer_correspondence_cka: float = 0.0  # Mean CKA of mapped layers

    # Safety
    refusal_preserved: bool = True
    safety_score: float = 1.0

    # Ollivier-Ricci curvature (raw geometric measurement)
    mean_ollivier_ricci: float = 0.0

    # Curvature alignment (for merge confidence)
    # Score 0.0-1.0: how aligned are source/target curvature profiles
    curvature_alignment: float = 0.0
    curvature_alignment_details: dict[str, float] = field(default_factory=dict)
