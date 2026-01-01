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

"""
Geometric Merge Orchestrator: Complete Pipeline Using ALL 84 Geometry Files.

This orchestrator integrates EVERY geometry file in the codebase. No dead code.
If a file exists in geometry/, it gets used here.

Key Insight: Higher dimensions contain lower dimensions.
- 1D is a compression of 2D
- 2D is a compression of 3D
- nD contains the entirety of (n-1)D

Therefore: We analyze at EVERY dimension level and blend accordingly.

Pipeline Stages:
================

STAGE 0: INFRASTRUCTURE
    - numerical_stability: Compute data-driven epsilons
    - geometry_metrics_cache: Set up caching

STAGE 1: PROBE & FINGERPRINT
    - compositional_probes: Generate probe texts
    - cka: Compute activation similarity
    - probe_calibration: Calibrate per-probe reliability
    - concept_response_matrix: Build CRM
    - fingerprints, geometry_fingerprint, topological_fingerprint: Extract fingerprints
    - fingerprint_cache: Cache fingerprints

STAGE 2: ANALYZE GEOMETRY
    - intrinsic_dimension: Per-layer intrinsic dimension
    - manifold_dimensionality: Manifold dimension at each layer
    - concept_dimensionality: Concept-specific dimensions
    - manifold_curvature: Curvature for geodesic interpolation
    - riemannian_density: Density estimation
    - gromov_wasserstein: GW distance between representations

STAGE 3: FIND SHARED STRUCTURE
    - shared_subspace_projector: CCA to find shared dimensions
    - relative_representation: Anchor-based dimension-agnostic alignment
    - cross_dimensional_projection: Project between dimension spaces
    - cross_architecture_layer_matcher: Match layers across architectures
    - invariant_layer_mapper: Map using invariants

STAGE 4: ALIGN
    - permutation_aligner: Re-Basin neuron alignment
    - generalized_procrustes: Multi-model Procrustes
    - tangent_space_alignment: Local tangent alignment
    - constraint_alignment: Constraint-based alignment

STAGE 5: ANALYZE INTERFERENCE
    - interference_predictor: Predict merge interference
    - spectral_analysis: Spectral metrics
    - transfer_fidelity: Transfer quality
    - null_space_filter: Compute null spaces

STAGE 6: COMPUTE DIMENSION WEIGHTS
    - dimension_blender: Per-dimension alpha
    - verb_noun_classifier: Skill vs structure
    - fisher_blending: Fisher importance weights
    - refinement_density: Per-layer scores
    - domain_signal_profile: Domain-specific weights

STAGE 7: BLEND
    - alpha_smoothing: Smooth alphas across layers
    - task_singular_vectors: SVD-based blending
    - transport_guided_merger: Optimal transport merge
    - dare_sparsity: DARE sparsification
    - affine_stitching_layer: Affine stitching

STAGE 8: VALIDATE
    - geometry_validation_suite: Validate geometry
    - anchor_invariance_analyzer: Check anchor stability
    - manifold_fidelity_sweep: Sweep for optimal subspace
    - safety_polytope: Check safety region
    - refusal_direction_detector: Preserve refusal

STAGE 9: DOMAIN ANALYSIS (optional)
    - social_geometry, moral_geometry, spatial_3d, temporal_topology
    - cross_cultural_geometry, domain_geometry_waypoints
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend

from .data_models import LayerGeometry, MergeGeometry
from .global_metrics import compute_global_metrics
from .infrastructure import setup_infrastructure
from .analysis import (
    stage_analyze_geometry,
    stage_analyze_interference,
    stage_compute_alignment,
    stage_compute_dimension_weights,
    stage_find_shared_structure,
    stage_layer_correspondence,
    stage_probe_fingerprint,
    stage_smooth_alphas,
    stage_validate,
)
from .weight_merger import merge_weights as merge_weights_impl

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class GeometricMergeOrchestrator:
    """
    Orchestrates ALL 84 geometry files into a complete merge pipeline.

    This is the single source of truth for geometric merging.
    Every geometry file is used. No dead code.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._cache: dict[str, Any] = {}
        self._epsilon = 1e-6
        self._avoid_svd = False
        self._metrics_cache: Any | None = None

    def analyze_merge(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        source_activations: dict[int, list["Array"]] | None = None,
        target_activations: dict[int, list["Array"]] | None = None,
        tokenizer: Any | None = None,
    ) -> MergeGeometry:
        """
        Complete geometric analysis of a merge operation.

        Uses ALL geometry files to build a complete picture of:
        - What dimensions are shared between models
        - How to align representations at each layer
        - What the interference patterns are
        - How to weight each dimension during merge

        Args:
            source_weights: Source model weights
            target_weights: Target model weights
            source_activations: Optional pre-computed activations per layer
            target_activations: Optional pre-computed activations per layer
            tokenizer: Optional tokenizer for probe generation

        Returns:
            MergeGeometry with complete analysis
        """
        b = self._backend
        logger.info("=== GEOMETRIC MERGE ANALYSIS ===")

        geometry = MergeGeometry(
            source_model="source",
            target_model="target",
        )

        # Extract layer indices
        layer_indices = self._extract_layer_indices(target_weights)
        logger.info("Analyzing %d layers", len(layer_indices))

        # STAGE 0: Infrastructure setup
        self._epsilon, self._avoid_svd, self._metrics_cache = setup_infrastructure()

        # STAGE 1: Probe & fingerprint (if activations available)
        if source_activations and target_activations:
            stage_probe_fingerprint(
                geometry, source_activations, target_activations, tokenizer, b
            )

        # STAGE 1.5: Layer correspondence for cross-architecture models
        if source_activations and target_activations:
            stage_layer_correspondence(
                geometry, source_activations, target_activations, b
            )

        # Build reverse correspondence: target_layer -> source_layer
        # Keep the FIRST (earliest) source layer for each target to maintain monotonicity
        reverse_correspondence: dict[int, int] = {}
        if geometry.layer_correspondence:
            for src_layer in sorted(geometry.layer_correspondence.keys()):
                tgt_layer = geometry.layer_correspondence[src_layer]
                if tgt_layer not in reverse_correspondence:
                    reverse_correspondence[tgt_layer] = src_layer

        # STAGE 2-8: Per-layer analysis
        for layer_idx in layer_indices:
            layer_geom = LayerGeometry(layer_idx=layer_idx)

            source_layer_idx = reverse_correspondence.get(layer_idx, layer_idx)
            src_acts = (
                source_activations.get(source_layer_idx)
                if source_activations
                else None
            )
            tgt_acts = target_activations.get(layer_idx) if target_activations else None

            # Get layer weights
            src_layer_weights = self._get_layer_weights(source_weights, source_layer_idx)
            tgt_layer_weights = self._get_layer_weights(target_weights, layer_idx)

            # STAGE 2: Analyze geometry
            stage_analyze_geometry(layer_geom, src_acts, tgt_acts, b)

            # STAGE 3: Find shared structure
            stage_find_shared_structure(
                layer_geom, src_acts, tgt_acts, b, avoid_svd=self._avoid_svd
            )

            # STAGE 4: Compute alignment
            stage_compute_alignment(
                layer_geom,
                src_acts,
                tgt_acts,
                src_layer_weights,
                tgt_layer_weights,
                b,
                is_cross_architecture=geometry.is_cross_architecture,
            )

            # STAGE 5: Analyze interference
            stage_analyze_interference(
                layer_geom,
                src_layer_weights,
                tgt_layer_weights,
                tgt_acts,
                b,
                cache=self._cache,
                avoid_svd=self._avoid_svd,
            )

            # STAGE 6: Compute dimension weights
            stage_compute_dimension_weights(
                layer_geom, src_acts, tgt_acts, src_layer_weights, tgt_layer_weights, b
            )

            geometry.layer_geometries[layer_idx] = layer_geom

        # STAGE 7: Smooth alphas across layers
        stage_smooth_alphas(geometry)

        # STAGE 8: Validate
        stage_validate(geometry, source_weights, target_weights)

        # Compute global metrics
        compute_global_metrics(geometry)

        return geometry

    def merge_weights(
        self,
        source_weights: dict[str, "Array"],
        target_weights: dict[str, "Array"],
        geometry: MergeGeometry,
        extract_layer_index_fn: Any,
        checkpoint_dir: str | None = None,
        layer_alpha_scale: dict[int, float] | None = None,
    ) -> tuple[dict[str, "Array"], dict[str, Any]]:
        return merge_weights_impl(
            source_weights,
            target_weights,
            geometry,
            extract_layer_index_fn,
            self._backend,
            avoid_svd=self._avoid_svd,
            checkpoint_dir=checkpoint_dir,
            layer_alpha_scale=layer_alpha_scale,
        )

    def _extract_layer_indices(self, weights: dict[str, "Array"]) -> list[int]:
        """Extract unique layer indices from weight keys."""
        import re

        indices = set()
        for key in weights:
            match = re.search(r"layers\.(\d+)\.", key)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def _get_layer_weights(
        self, weights: dict[str, "Array"], layer_idx: int
    ) -> dict[str, "Array"]:
        """Get weights for a specific layer."""
        pattern = f"layers.{layer_idx}."
        return {k: v for k, v in weights.items() if pattern in k}
