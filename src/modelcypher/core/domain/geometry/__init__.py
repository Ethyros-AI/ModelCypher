"""
Geometry domain - geometric analysis and transformation of model representations.

Core modules for analyzing manifold structure, computing alignments, and
understanding the geometric properties of language model weight spaces.

Note: This module imports submodules directly instead of using star imports
to avoid namespace pollution that can interfere with dataclass field definitions.
"""
from __future__ import annotations

# =============================================================================
# Geometry Submodules - import modules (not contents) to avoid namespace pollution
# =============================================================================

from . import types
from . import exceptions
from . import vector_math
from . import affine_stitching_layer
from . import alpha_smoothing
from . import anchor_invariance_analyzer
from . import cka
from . import compositional_probes
from . import concept_detector
from . import concept_response_matrix
from . import cross_architecture_layer_matcher
from . import cross_cultural_geometry
from . import dare_sparsity
from . import domain_signal_profile
from . import dora_decomposition
from . import fingerprints
from . import gate_detector
from . import generalized_procrustes
from . import geometry_fingerprint
from . import geometry_validation_suite
from . import gromov_wasserstein
from . import intersection_map_analysis
from . import intrinsic_dimension
from . import intrinsic_dimension_estimator
from . import invariant_convergence_analyzer
from . import invariant_layer_mapper
from . import manifold_clusterer
from . import manifold_dimensionality
from . import manifold_fidelity_sweep
from . import manifold_profile
from . import manifold_stitcher
from . import metaphor_convergence_analyzer
from . import model_fingerprints_projection
from . import path_geometry
from . import permutation_aligner
from . import persona_vector_monitor
from . import probe_corpus
from . import probes
from . import refinement_density
from . import refusal_direction_cache
from . import refusal_direction_detector
from . import shared_subspace_projector
from . import sparse_region_domains
from . import sparse_region_locator
from . import sparse_region_prober
from . import sparse_region_validator
from . import spectral_analysis
from . import tangent_space_alignment
from . import task_singular_vectors
from . import thermo_path_integration
from . import topological_fingerprint
from . import transfer_fidelity
from . import transport_guided_merger
from . import traversal_coherence
from . import verb_noun_classifier
from . import dimension_blender
from . import relative_representation

# Backward compatibility: explicitly re-export commonly used classes
from .dora_decomposition import DoRAConfig as DoRAConfiguration
from .dora_decomposition import ChangeType, DoRADecomposition

# VectorMath is commonly imported directly
from .vector_math import VectorMath

# PathNode and PathSignature for path geometry analysis
from .path_geometry import PathNode, PathSignature

# PermutationAligner for weight permutation alignment
from .permutation_aligner import PermutationAligner

# DimensionBlender for dimension correlation analysis
from .dimension_blender import DimensionBlender
