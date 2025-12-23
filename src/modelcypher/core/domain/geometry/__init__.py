"""
Geometry domain - geometric analysis and transformation of model representations.

Core modules for analyzing manifold structure, computing alignments, and
understanding the geometric properties of language model weight spaces.
"""
from __future__ import annotations

# =============================================================================
# All geometry modules - star-imported for maximum connectivity
# =============================================================================

from .types import *  # noqa: F401,F403
from .vector_math import *  # noqa: F401,F403
from .affine_stitching_layer import *  # noqa: F401,F403
from .alpha_smoothing import *  # noqa: F401,F403
from .anchor_invariance_analyzer import *  # noqa: F401,F403
from .cka import *  # noqa: F401,F403
from .compositional_probes import *  # noqa: F401,F403
from .concept_detector import *  # noqa: F401,F403
from .concept_response_matrix import *  # noqa: F401,F403
from .cross_architecture_layer_matcher import *  # noqa: F401,F403
from .cross_cultural_geometry import *  # noqa: F401,F403
from .dare_sparsity import *  # noqa: F401,F403
from .domain_signal_profile import *  # noqa: F401,F403
from .dora_decomposition import *  # noqa: F401,F403
from .dora_decomposition import DoRAConfig as DoRAConfiguration  # alias for backward compat
from .fingerprints import *  # noqa: F401,F403
from .gate_detector import *  # noqa: F401,F403  # uses TYPE_CHECKING + lazy import
from .generalized_procrustes import *  # noqa: F401,F403
from .geometry_fingerprint import *  # noqa: F401,F403
from .geometry_validation_suite import *  # noqa: F401,F403
from .gromov_wasserstein import *  # noqa: F401,F403
from .intersection_map_analysis import *  # noqa: F401,F403
from .intrinsic_dimension import *  # noqa: F401,F403
from .intrinsic_dimension_estimator import *  # noqa: F401,F403
from .invariant_convergence_analyzer import *  # noqa: F401,F403
from .invariant_layer_mapper import *  # noqa: F401,F403  # uses TYPE_CHECKING + lazy imports
from .manifold_clusterer import *  # noqa: F401,F403
from .manifold_dimensionality import *  # noqa: F401,F403
from .manifold_fidelity_sweep import *  # noqa: F401,F403
from .manifold_profile import *  # noqa: F401,F403
from .manifold_stitcher import *  # noqa: F401,F403
from .metaphor_convergence_analyzer import *  # noqa: F401,F403
from .model_fingerprints_projection import *  # noqa: F401,F403
from .path_geometry import *  # noqa: F401,F403
from .permutation_aligner import *  # noqa: F401,F403
from .persona_vector_monitor import *  # noqa: F401,F403
from .probe_corpus import *  # noqa: F401,F403
from .probes import *  # noqa: F401,F403
from .refinement_density import *  # noqa: F401,F403
from .refusal_direction_cache import *  # noqa: F401,F403
from .refusal_direction_detector import *  # noqa: F401,F403
from .shared_subspace_projector import *  # noqa: F401,F403
from .sparse_region_domains import *  # noqa: F401,F403
from .sparse_region_locator import *  # noqa: F401,F403
from .sparse_region_prober import *  # noqa: F401,F403
from .sparse_region_validator import *  # noqa: F401,F403
from .spectral_analysis import *  # noqa: F401,F403
from .tangent_space_alignment import *  # noqa: F401,F403
from .task_singular_vectors import *  # noqa: F401,F403
from .thermo_path_integration import *  # noqa: F401,F403
from .topological_fingerprint import *  # noqa: F401,F403
from .transfer_fidelity import *  # noqa: F401,F403
from .transport_guided_merger import *  # noqa: F401,F403
from .traversal_coherence import *  # noqa: F401,F403
from .verb_noun_classifier import *  # noqa: F401,F403  # uses lazy import in helper functions
from .dimension_blender import *  # noqa: F401,F403  # uses TYPE_CHECKING + lazy import

# dimension_blender excluded due to cross-package dependency on agents
# Can be imported directly: from modelcypher.core.domain.geometry.dimension_blender import ...
