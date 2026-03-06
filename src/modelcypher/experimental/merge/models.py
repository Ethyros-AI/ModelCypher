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
    from modelcypher.core.domain.geometry.alignment import FamilySimilarity
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

    # Geometric fingerprint comparison (source vs target geometry)
    fingerprint_comparison: "FingerprintComparison | None" = None

    # Family similarity (spectral Procrustes distance for merge quality prediction)
    family_similarity: "FamilySimilarity | None" = None


@dataclass
class FingerprintComparison:
    """Geometric fingerprint comparison between source and target models.

    Raw measurements only. No interpretation.
    """

    # Gram hash (SHA-256 of flattened Gram matrix)
    source_gram_hash: str
    target_gram_hash: str

    # Condition number: κ = λ_max / λ_min
    source_condition_number: float
    target_condition_number: float

    # Effective dimensionality: (Σλ)² / Σλ² (participation ratio)
    source_effective_dim: float
    target_effective_dim: float

    # Derived: target / source
    condition_number_ratio: float
    # Derived: target - source
    effective_dim_delta: float


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
    """Geometric profile of layers based on measured variance concentration.

    NO HEURISTICS. The geometry tells us:
    - Variance concentration (var_top1) varies by layer
    - High var_top1 = bottleneck (information compressed into few directions)
    - Low effective_rank = bottleneck (few dimensions carry the signal)
    - Gram rank drops to 2-3 at the bottleneck (~50% depth)

    KEY INSIGHT: TwoNN intrinsic dimension FAILS to detect true 1D bottlenecks.
    Example: LFM2-350M layer 7 has 99.4% variance in top-1 singular value,
    but TwoNN reports ID=6.39 (not even close to 1).

    This profile stores MEASUREMENTS, not thresholds.
    The merge code uses these measurements directly.
    """

    # Per-layer variance concentration (var_top1 = % of variance in top-1 singular value)
    # High value (>0.7) = bottleneck layer
    variance_concentrations: dict[int, float] = field(default_factory=dict)

    # Per-layer effective rank (entropy-based dimensionality)
    # Low value = bottleneck layer
    effective_ranks: dict[int, float] = field(default_factory=dict)

    # Per-layer Gram rank (measured)
    gram_ranks: dict[int, int] = field(default_factory=dict)

    # Layer 0 is always embedding (this is structural, not heuristic)
    embedding_layer: int = 0

    # Total layer count
    total_layers: int = 0

    # Bottleneck layer (highest variance concentration) - MEASURED
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
            1.0 = fully safe (largest measured headroom in this profile)
        """
        if not self.boundary_radii:
            raise ValueError(
                "Transfer safety requires measured boundary_radii; none were provided."
            )

        max_radius = max(self.boundary_radii.values())
        if max_radius <= 0:
            return 0.0

        if layer_idx not in self.boundary_radii:
            raise KeyError(
                f"Layer {layer_idx} missing boundary radius; transfer safety is undefined."
            )

        layer_radius = self.boundary_radii[layer_idx]
        return layer_radius / max_radius

    def compute_highway_layers(self) -> list[int]:
        """Identify highway layers based on variance concentration.

        Algorithm:
        1. Find median variance concentration across all layers
        2. Highway = layers where var_top1 >= median
        3. Ramps are the complement (var_top1 < median)

        Returns:
            List of layer indices that are part of the semantic highway.
        """
        if not self.variance_concentrations:
            return list(range(self.total_layers))

        var_values = sorted(self.variance_concentrations.values(), reverse=True)
        if len(var_values) < 3:
            return list(self.variance_concentrations.keys())

        # Compute median as threshold
        mid_idx = len(var_values) // 2
        if len(var_values) % 2 == 0:
            median_var = (var_values[mid_idx - 1] + var_values[mid_idx]) / 2.0
        else:
            median_var = var_values[mid_idx]

        # Highway = layers with var_top1 >= median (high concentration = semantic core)
        highway = [
            layer_idx
            for layer_idx, var_val in self.variance_concentrations.items()
            if var_val >= median_var
        ]
        return sorted(highway)

    def compute_ramp_layers(self) -> list[int]:
        """Identify ramp layers based on variance concentration.

        Returns the complement of highway layers.

        Returns:
            List of layer indices that are ramps (not highway).
        """
        if not self.variance_concentrations:
            return []

        highway = set(self.compute_highway_layers())
        all_layers = set(self.variance_concentrations.keys())
        ramps = all_layers - highway
        return sorted(ramps)

    def get_bottleneck_layer(self) -> int | None:
        """Return the bottleneck layer with the highest variance concentration.

        The embedding layer is excluded.

        Returns:
            Layer index with highest var_top1, or None if no data.
        """
        if not self.variance_concentrations:
            return None

        max_var = -1.0
        bottleneck_layer = None

        for layer_idx, var_val in self.variance_concentrations.items():
            if layer_idx == self.embedding_layer:
                continue  # Skip embedding layer
            if var_val > max_var:
                max_var = var_val
                bottleneck_layer = layer_idx

        return bottleneck_layer

    def compute_bottleneck_layers(self) -> list[int]:
        """Identify the true bottleneck - THE single most compressed layer.

        Returns a list for API compatibility, but contains only one layer.

        Returns:
            List with single bottleneck layer index, or empty if no data.
        """
        bottleneck = self.get_bottleneck_layer()
        return [bottleneck] if bottleneck is not None else []

    def set_cross_architecture_skip_layers(self) -> None:
        """Auto-populate skip_layers for cross-architecture merges.

        Keeps only bottleneck layers for transplant and skips the embedding layer.
        """
        # Get bottleneck layers (super highway)
        bottleneck = set(self.compute_bottleneck_layers())

        if not self.variance_concentrations:
            # No data - skip only embedding
            self.skip_layers = sorted(set(self.skip_layers or []) | {self.embedding_layer})
            return

        all_layers = set(self.variance_concentrations.keys())

        # Everything NOT in the bottleneck is a translation layer
        translation_layers = all_layers - bottleneck

        # Also always skip embedding layer (structural)
        translation_layers.add(self.embedding_layer)

        # Combine with any existing skip layers
        existing = set(self.skip_layers or [])
        combined = existing.union(translation_layers)
        self.skip_layers = sorted(combined)

    def compute_transmission_layers(self) -> list[int]:
        """Identify transmission layers based on variance concentration and rank.

        Heuristic:
        - Low variance concentration (bottom quartile)
        - High effective rank (top quartile)

        Returns:
            List of layer indices that are transmission layers.
        """
        if not self.variance_concentrations or not self.effective_ranks:
            return []

        n = len(self.variance_concentrations)
        if n < 4:
            return []  # Not enough layers to identify transmission

        # Get layers with LOW variance concentration (bottom quartile)
        # More selective than bottom half - we want the truly linear layers
        sorted_by_var = sorted(
            self.variance_concentrations.keys(),
            key=lambda x: self.variance_concentrations[x]
        )
        low_concentration_layers = set(sorted_by_var[:n // 4])

        # Get layers with HIGH effective rank (top quartile)
        sorted_by_rank = sorted(
            self.effective_ranks.keys(),
            key=lambda x: self.effective_ranks[x],
            reverse=True
        )
        high_rank_layers = set(sorted_by_rank[:n // 4])

        # Transmission = low concentration AND high effective rank
        transmission = low_concentration_layers & high_rank_layers

        # Exclude embedding (layer 0) - structural
        transmission.discard(self.embedding_layer)

        # Exclude first 10% and last 10% of layers (true ramps at edges)
        # These are entry/exit translation layers that handle vocabulary
        edge_margin = max(1, n // 10)
        edge_layers = set(range(edge_margin)) | set(range(n - edge_margin, n))
        transmission = transmission - edge_layers

        return sorted(transmission)

    def compute_best_injection_layer(self) -> int | None:
        """Find THE SINGLE best layer for knowledge injection.

        The highway is just rolling information over - the geometry is
        identical across transmission layers. We only need ONE injection point.

        Selection criteria (in order):
        1. Must be in transmission region (linear highway)
        2. Highest variance concentration within transmission (most compressed)
        3. If tie, pick the middle layer (safest position)

        Returns:
            The single best layer index for injection, or None if no valid layer.
        """
        transmission = self.compute_transmission_layers()
        if not transmission:
            return None

        if len(transmission) == 1:
            return transmission[0]

        # Pick highest variance concentration within transmission (most compressed)
        if self.variance_concentrations:
            valid = [(idx, self.variance_concentrations.get(idx, 0.0))
                     for idx in transmission
                     if idx in self.variance_concentrations]
            if valid:
                # Sort by var_top1 (descending), then by layer index (prefer middle)
                valid.sort(key=lambda x: (-x[1], abs(x[0] - len(transmission) // 2)))
                return valid[0][0]

        # No variance data: pick the middle transmission layer
        return transmission[len(transmission) // 2]


@dataclass
class MergeLayerGeometry:
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
    layer_geometries: dict[int, MergeLayerGeometry] = field(default_factory=dict)

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
