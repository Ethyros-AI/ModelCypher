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
Geometry domain - geometric analysis and transformation of model representations.

Core modules for analyzing manifold structure, computing alignments, and
understanding the geometric properties of language model weight spaces.

Uses lazy imports to avoid loading all 95 submodules at package import time.
Import specific modules directly when needed: `from .cka import CKA`
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Module registry for lazy loading - includes all 95 geometry modules
_SUBMODULES = {
    # Core infrastructure
    "types",
    "exceptions",
    "backend_matrix_utils",
    "signature_base",
    "numerical_stability",
    # Anchor-relative grafting pipeline
    "anchor_decoder",
    "anchor_grafting",
    # Analysis and metrics
    "anchor_invariance_analyzer",
    "alignment_diagnostic",
    "alignment_validation",
    "analytic_manifolds",
    "atlas_protocols",
    "atlas_registry",
    "birkhoff_projector",
    "birkhoff_router",
    "channel_projector",
    "cka",
    "compositional_probes",
    "concept_detector",
    "concept_dimensionality",
    "concept_response_matrix",
    "constrained_transplant",
    "constraint_alignment",
    "cross_architecture_layer_matcher",
    "cross_cultural_geometry",
    "cross_dimensional_projection",
    "cross_grounding_transfer",
    "curvature_alignment",
    "curvature_profile",
    "dare_sparsity",
    "density_estimator",
    "deviation_budget",
    "dimension_cascade",
    "dimensional_alignment",
    "domain_benchmark_map",
    "domain_geometry_waypoints",
    "domain_signal_profile",
    "dora_decomposition",
    "effective_rank",
    "fingerprint_cache",
    "gate_detector",
    "generalized_procrustes",
    "geodesic_null_space",
    "geometric_lora",
    "geometry_fingerprint",
    "geometry_metrics_cache",
    "geometry_validation_suite",
    "evidence_suite",
    "gram_aligner",
    "gromov_wasserstein",
    "interference_predictor",
    "intersection_map_analysis",
    "intersection_similarity",
    "intrinsic_dimension",
    "invariant_convergence_analyzer",
    "invariant_layer_mapper",
    "knowledge_density",
    "knowledge_diff",
    "low_rank_gw",
    "manifold_clusterer",
    "manifold_accuracy",
    "manifold_curvature",
    "manifold_dimensionality",
    "manifold_evidence",
    "manifold_fidelity_sweep",
    "manifold_profile",
    "manifold_stitcher",
    "manifold_transfer",
    "metaphor_convergence_analyzer",
    "metaphor_invariance",
    "metaphor_invariants",
    "metaphor_trajectory",
    "model_fingerprints_projection",
    "model_profile",
    "moral_geometry",
    "neuron_sparsity_analyzer",
    "optimal_transport",
    "path_geometry",
    "persona_vector_monitor",
    "prime_geometry_analysis",
    "prime_geometry_baselines",
    "prime_geometry_embeddings",
    "prime_geometry_spectral",
    "prime_geometry_stats",
    "prime_geometry_types",
    "prime_geometry_utils",
    "probe_calibration",
    "profile_comparison",
    "refinement_density",
    "refusal_direction_cache",
    "refusal_direction_detector",
    "relative_representation",
    "riemannian_core_covariance",
    "riemannian_core_curvature",
    "riemannian_core_geodesic",
    "riemannian_core_mean",
    "riemannian_core_utils",
    "riemannian_density",
    "riemannian_utils",
    "safety_polytope",
    "shared_subspace_projector",
    "shared_manifold",
    "social_geometry",
    "sparse_region_domains",
    "sparse_region_locator",
    "sparse_region_prober",
    "sparse_region_validator",
    "spatial_3d",
    "spectral_analysis",
    "spectral_signature",
    "tangent_space_alignment",
    "temporal_topology",
    "thermo_path_integration",
    "topological_fingerprint",
    "transfer_fidelity",
    "transplant",
    "traversal_coherence",
    "wudi_interference",
}

# Attribute to submodule mapping for commonly used classes
# Format: "ExportedName": ("module_name", "actual_attr_name")
_ATTR_TO_MODULE = {
    # Core merge infrastructure
    "BirkhoffProjector": ("birkhoff_projector", "BirkhoffProjector"),
    "BirkhoffRouter": ("birkhoff_router", "BirkhoffRouter"),
    "RoutingMode": ("birkhoff_router", "RoutingMode"),
    "ChannelProjector": ("channel_projector", "ChannelProjector"),
    "GeneralizedProcrustes": ("generalized_procrustes", "GeneralizedProcrustes"),
    "GramAligner": ("gram_aligner", "GramAligner"),
    "GeodesicNullSpaceFilter": ("geodesic_null_space", "GeodesicNullSpaceFilter"),
    # Dimensional alignment (Priority 2) 
    "DimensionalAlignment": ("dimensional_alignment", "DimensionalAlignment"),
    "measure_dimensional_alignment": ("dimensional_alignment", "measure_dimensional_alignment"),
    # Anchor invariance analysis (Priority 2)
    "AnchorInvarianceAnalyzer": ("anchor_invariance_analyzer", "AnchorInvarianceAnalyzer"),
    # Constraint-based alignment (Priority 2)
    "ConstraintAligner": ("constraint_alignment", "ConstraintAligner"),
    # Cross-dimensional projection (Priority 2)
    "project_cross_dimensional": ("cross_dimensional_projection", "project_cross_dimensional"),
    # Sparse region analysis
    "SparseRegionProber": ("sparse_region_prober", "SparseRegionProber"),
    "SparseRegionValidator": ("sparse_region_validator", "SparseRegionValidator"),
    "SparseRegionLocator": ("sparse_region_locator", "SparseRegionLocator"),
    # Spectral analysis
    "SpectralMetrics": ("spectral_analysis", "SpectralMetrics"),
    # Geometric LoRA (Priority 5)
    "GeometricLoRAGenerator": ("geometric_lora", "GeometricLoRAGenerator"),
    # Tangent space alignment (Riemannian)
    "TangentSpaceAlignment": ("tangent_space_alignment", "TangentSpaceAlignment"),
    "compute_alignment_for_layers": ("tangent_space_alignment", "compute_alignment_for_layers"),
    # Cross-dimension transfer via relative representation (CRITICAL for cross-arch merge)
    "cross_dimension_transfer": ("relative_representation", "cross_dimension_transfer"),
    "RelativeRepresentation": ("relative_representation", "RelativeRepresentation"),
    # Anchor-relative concept grafting (CANONICAL merge pipeline)
    "AnchorDecodingResult": ("anchor_decoder", "AnchorDecodingResult"),
    "compute_anchor_decoder": ("anchor_decoder", "compute_anchor_decoder"),
    "decode_to_activation_space": ("anchor_decoder", "decode_to_activation_space"),
    "AnchorGraftingResult": ("anchor_grafting", "AnchorGraftingResult"),
    "compute_anchor_grafting_delta": ("anchor_grafting", "compute_anchor_grafting_delta"),
    # DoRA analysis
    "ChangeType": ("dora_decomposition", "ChangeType"),
    "DoRADecomposition": ("dora_decomposition", "DoRADecomposition"),
    "PathNode": ("path_geometry", "PathNode"),
    "PathSignature": ("path_geometry", "PathSignature"),
    # Backend-aware matrix utilities
    "BackendMatrixUtils": ("backend_matrix_utils", "BackendMatrixUtils"),
    "ProcrustesResult": ("types", "ProcrustesResult"),
    "PairwiseProcrustesResult": ("types", "PairwiseProcrustesResult"),
    # Signature base classes
    "SignatureMixin": ("signature_base", "SignatureMixin"),
    "LabeledSignatureMixin": ("signature_base", "LabeledSignatureMixin"),
    # Dimension cascade for real-time visualization
    "DimensionCascade": ("dimension_cascade", "DimensionCascade"),
    "CascadeResult": ("dimension_cascade", "CascadeResult"),
    # Density estimation for visualization
    "DensityEstimator": ("density_estimator", "DensityEstimator"),
    "DensityResult": ("density_estimator", "DensityResult"),
    # Deviation tracking (unified for merge + multi-modal injection)
    # Note: DeviationBudget is an alias for DeviationTracker (backward compat)
    "DeviationBudget": ("deviation_budget", "DeviationBudget"),
    "DeviationTracker": ("deviation_budget", "DeviationTracker"),
    "DeviationMeasurement": ("deviation_budget", "DeviationMeasurement"),
    # Metaphor geometry
    "MetaphorTrajectory": ("metaphor_trajectory", "MetaphorTrajectory"),
    "MetaphorTrajectoryPoint": ("metaphor_trajectory", "MetaphorTrajectoryPoint"),
    "MetaphorTrajectoryCollector": ("metaphor_trajectory", "MetaphorTrajectoryCollector"),
    "ConvergenceProfile": ("metaphor_trajectory", "ConvergenceProfile"),
    "MetaphorInvarianceResult": ("metaphor_invariance", "MetaphorInvarianceResult"),
    "MetaphorInvarianceAnalyzer": ("metaphor_invariance", "MetaphorInvarianceAnalyzer"),
    "PlatonicMetaphorValidator": ("metaphor_invariance", "PlatonicMetaphorValidator"),
    # Domain signal profiles for domain-aware merging
    "DomainSignalProfile": ("domain_signal_profile", "DomainSignalProfile"),
    "LayerSignal": ("domain_signal_profile", "LayerSignal"),
    # Safety and transfer validation
    "SafetyPolytope": ("safety_polytope", "SafetyPolytope"),
    "TransferFidelityPrediction": ("transfer_fidelity", "TransferFidelityPrediction"),
    # Metaphor invariant probe generation
    "generate_all_metaphor_probes": ("metaphor_invariants", "generate_all_metaphor_probes"),
    "MetaphorProbe": ("metaphor_invariants", "MetaphorProbe"),
}
# Filter out None values (conditional exports)
_ATTR_TO_MODULE = {k: v for k, v in _ATTR_TO_MODULE.items() if v is not None}


def __getattr__(name: str):
    """Lazy load submodules and commonly used attributes."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTR_TO_MODULE:
        module_name, attr_name = _ATTR_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules and attributes."""
    return list(_SUBMODULES) + list(_ATTR_TO_MODULE.keys())


# TYPE_CHECKING block for static analysis - these imports don't run at runtime
if TYPE_CHECKING:
    from .backend_matrix_utils import BackendMatrixUtils
    from .dora_decomposition import ChangeType, DoRADecomposition
    from .path_geometry import PathNode, PathSignature
    from .signature_base import LabeledSignatureMixin, SignatureMixin
    from .types import PairwiseProcrustesResult, ProcrustesResult
