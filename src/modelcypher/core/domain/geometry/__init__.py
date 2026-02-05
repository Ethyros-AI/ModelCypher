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

# Module registry for lazy loading - includes all geometry modules
_SUBMODULES = {
    # Core infrastructure
    "types",
    "exceptions",
    "backend_matrix_utils",
    "signature_base",
    "numerical_stability",
    # Numerical stability submodules (split from numerical_stability.py)
    "scalars",
    "precision",
    "statistics",
    "decomposition",
    "alignment",
    "spectral_init",
    "validation",
    # Cross-modal and LoRA synthesis
    "affine_bridge",
    "direct_lora_synthesis",
    # New geometric measurements (Priority 1-3)
    "parallel_transport",
    "geodesic_deviation",
    # Anchor-relative grafting pipeline
    "anchor_decoder",
    "anchor_grafting",
    # Analysis and metrics
    "alignment_diagnostic",
    "alignment_validation",
    "analytic_manifolds",
    "atlas_protocols",
    "atlas_registry",
    "birkhoff_projector",
    "birkhoff_router",
    "channel_projector",
    "cka",
    "concept_detector",
    "concept_dimensionality",
    "concept_response_matrix",
    "constrained_transplant",
    "cross_architecture_layer_matcher",
    "cross_grounding_transfer",
    "curvature_alignment",
    "curvature_profile",
    "dare_sparsity",
    "density_estimator",
    "deviation_budget",
    "dimension_cascade",
    "dimensional_alignment",
    "domain_benchmark_map",
    "domain_signal_profile",
    "dora_decomposition",
    "effective_rank",
    "entanglement_spectrum",
    "fingerprint_cache",
    "fisher_information",  # Fisher Information Geometry for merge compatibility
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
    "hot_layer_matcher",  # Hierarchical Optimal Transport layer matching (SOTA)
    "hungarian",  # Classic Hungarian algorithm for bipartite assignment
    "interference_predictor",
    "intersection_similarity",
    "intrinsic_dimension",
    "invariant_layer_mapper",
    "knowledge_density",
    "knowledge_diff",
    "lie_rotation",
    "low_rank_gw",
    "manifold_clusterer",
    "manifold_accuracy",
    "manifold_curvature",
    "manifold_dimensionality",
    "manifold_evidence",
    "manifold_profile",
    "mode_connectivity",  # Mode Connectivity for merge compatibility prediction
    "prompt_manifold",
    "manifold_stitcher",
    "manifold_transfer",
    "model_fingerprints_projection",
    "model_profile",
    "neuron_sparsity_analyzer",
    "optimal_transport",
    "path_geometry",
    "persona_vector_monitor",
    "probe_calibration",
    "profile_comparison",
    "positive_geometry",
    "refinement_density",
    "refusal_direction_cache",
    "refusal_direction_detector",
    "relative_representation",
    "representation_consistency",  # Semantic stability measurement for LoRA validation
    "rmt_signal_separation",  # Random Matrix Theory for null-space detection
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
    "sliced_wasserstein",  # Sliced Wasserstein for scalable distribution comparison
    "sparse_region_domains",
    "sparse_region_locator",
    "sparse_region_prober",
    "sparse_region_validator",
    "spatial_3d",
    "spectral_analysis",
    "spectral_signature",
    "tangent_space_alignment",
    "thermo_path_integration",
    "topological_fingerprint",
    "transfer_fidelity",
    "transplant",
    "traversal_coherence",
    # Manifold entropy (geometric self-alignment)
    "manifold_entropy",
    # Trajectory complexity (looping vs feedforward dynamics)
    "trajectory_complexity",
    # Universal LoRA transfer (cross-model adapter projection)
    "universal_lora_projector",
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
    "PositiveGrassmannSignature": ("positive_geometry", "PositiveGrassmannSignature"),
    "compute_positive_grassmann_signature": (
        "positive_geometry",
        "compute_positive_grassmann_signature",
    ),
    # DoRA analysis
    "ChangeType": ("dora_decomposition", "ChangeType"),
    "DoRADecomposition": ("dora_decomposition", "DoRADecomposition"),
    "PathNode": ("path_geometry", "PathNode"),
    "PathSignature": ("path_geometry", "PathSignature"),
    # Entanglement spectrum (CCA-based coupling measurement)
    "EntanglementSpectrum": ("entanglement_spectrum", "EntanglementSpectrum"),
    "EntanglementSpectrumResult": ("entanglement_spectrum", "EntanglementSpectrumResult"),
    "compute_entanglement_spectrum": ("entanglement_spectrum", "compute_entanglement_spectrum"),
    # HOT layer matcher (SOTA: Hierarchical Optimal Transport)
    "HOTLayerMatcher": ("hot_layer_matcher", "HOTLayerMatcher"),
    "HOTLayerMatchingResult": ("hot_layer_matcher", "HOTLayerMatchingResult"),
    "hot_layer_matching": ("hot_layer_matcher", "hot_layer_matching"),
    "coupling_to_assignment": ("hot_layer_matcher", "coupling_to_assignment"),
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
    "DeviationTracker": ("deviation_budget", "DeviationTracker"),
    "DeviationMeasurement": ("deviation_budget", "DeviationMeasurement"),
    "derive_delta_scale": ("deviation_budget", "derive_delta_scale"),
    # Domain signal profiles for domain-aware merging
    "DomainSignalProfile": ("domain_signal_profile", "DomainSignalProfile"),
    "LayerSignal": ("domain_signal_profile", "LayerSignal"),
    # Safety and transfer validation
    "SafetyPolytope": ("safety_polytope", "SafetyPolytope"),
    "TransferFidelityPrediction": ("transfer_fidelity", "TransferFidelityPrediction"),
    # Parallel Transport and Holonomy (Priority 1B)
    "ParallelTransportResult": ("parallel_transport", "ParallelTransportResult"),
    "HolonomyResult": ("parallel_transport", "HolonomyResult"),
    "ParallelTransporter": ("parallel_transport", "ParallelTransporter"),
    "parallel_transport": ("parallel_transport", "parallel_transport"),
    "compute_holonomy": ("parallel_transport", "compute_holonomy"),
    # Geodesic Deviation / Jacobi Fields (Priority 1A)
    "GeodesicDeviationResult": ("geodesic_deviation", "GeodesicDeviationResult"),
    "GeodesicDeviationAnalyzer": ("geodesic_deviation", "GeodesicDeviationAnalyzer"),
    "compute_geodesic_deviation": ("geodesic_deviation", "compute_geodesic_deviation"),
    # Rotation Flow (Priority 2A) - added to generalized_procrustes
    "RotationFlowResult": ("generalized_procrustes", "RotationFlowResult"),
    "RotationFlowAnalyzer": ("generalized_procrustes", "RotationFlowAnalyzer"),
    "compute_rotation_flow": ("generalized_procrustes", "compute_rotation_flow"),
    # Grassmann Distance (Priority 2B) - added to subspace
    "GrassmannDistanceResult": ("subspace", "GrassmannDistanceResult"),
    "compute_grassmann_distance": ("subspace", "compute_grassmann_distance"),
    "grassmann_log": ("subspace", "grassmann_log"),
    # Heat Kernel Signature (Priority 3A) - added to spectral_signature
    "HeatKernelSignatureResult": ("spectral_signature", "HeatKernelSignatureResult"),
    "HeatKernelSignature": ("spectral_signature", "HeatKernelSignature"),
    "compute_heat_kernel_signature": ("spectral_signature", "compute_heat_kernel_signature"),
    "compare_hks_profiles": ("spectral_signature", "compare_hks_profiles"),
    # Manifold Entropy (Geometric Self-Alignment Phase 1)
    "ManifoldEntropy": ("manifold_entropy", "ManifoldEntropy"),
    "ManifoldEntropyResult": ("manifold_entropy", "ManifoldEntropyResult"),
    "LayerEntropyResult": ("manifold_entropy", "LayerEntropyResult"),
    "SVDSignatureResult": ("manifold_entropy", "SVDSignatureResult"),
    "ComplexityLawResult": ("manifold_entropy", "ComplexityLawResult"),
    "compute_manifold_entropy": ("manifold_entropy", "compute_manifold_entropy"),
    # Trajectory complexity (looping vs feedforward dynamics)
    "TrajectoryComplexity": ("trajectory_complexity", "TrajectoryComplexity"),
    "TrajectoryComplexityResult": ("trajectory_complexity", "TrajectoryComplexityResult"),
    "compute_trajectory_complexity": ("trajectory_complexity", "compute_trajectory_complexity"),
    # Affine bridge (cross-modal alignment with vocabulary constraints)
    "AffineBridge": ("affine_bridge", "AffineBridge"),
    "AffineBridgeResult": ("affine_bridge", "AffineBridgeResult"),
    "VocabConstrainedProjection": ("affine_bridge", "VocabConstrainedProjection"),
    "VocabConstrainedResult": ("affine_bridge", "VocabConstrainedResult"),
    "HybridBridge": ("affine_bridge", "HybridBridge"),
    # Mode connectivity (merge compatibility via loss barriers)
    "ModeConnectivityResult": ("mode_connectivity", "ModeConnectivityResult"),
    "LossBarrierProfile": ("mode_connectivity", "LossBarrierProfile"),
    "analyze_mode_connectivity": ("mode_connectivity", "analyze_mode_connectivity"),
    "compute_loss_barrier_profile": ("mode_connectivity", "compute_loss_barrier_profile"),
    # Direct LoRA synthesis (training-free LoRA from manifold points)
    "GeometricLoRA": ("direct_lora_synthesis", "GeometricLoRA"),
    "LayerLoRAWeights": ("direct_lora_synthesis", "LayerLoRAWeights"),
    "generate_geometric_lora": ("direct_lora_synthesis", "generate_geometric_lora"),
    "save_geometric_lora": ("direct_lora_synthesis", "save_geometric_lora"),
    # Probe calibration (probe quality control via CKA)
    "ProbeCalibrator": ("probe_calibration", "ProbeCalibrator"),
    "ProbeCalibrationResult": ("probe_calibration", "ProbeCalibrationResult"),
    "CalibrationReport": ("probe_calibration", "CalibrationReport"),
    # Universal LoRA transfer (cross-model adapter projection)
    "UniversalLoRAProjector": ("universal_lora_projector", "UniversalLoRAProjector"),
    "LoRATransferResult": ("universal_lora_projector", "LoRATransferResult"),
    "LayerTransferResult": ("universal_lora_projector", "LayerTransferResult"),
    "SVDComponents": ("universal_lora_projector", "SVDComponents"),
    "compute_lora_delta": ("universal_lora_projector", "compute_lora_delta"),
    "decompose_to_lora": ("universal_lora_projector", "decompose_to_lora"),
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
