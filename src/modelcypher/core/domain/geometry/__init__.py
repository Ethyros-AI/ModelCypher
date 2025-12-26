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

Uses lazy imports to avoid loading all 55+ submodules at package import time.
Import specific modules directly when needed: `from .cka import CKA`
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Module registry for lazy loading
_SUBMODULES = {
    "types",
    "exceptions",
    "vector_math",
    "backend_matrix_utils",
    "matrix_utils",
    "signature_base",
    "affine_stitching_layer",
    "alpha_smoothing",
    "anchor_invariance_analyzer",
    "cka",
    "compositional_probes",
    "concept_detector",
    "concept_dimensionality",
    "concept_response_matrix",
    "cross_architecture_layer_matcher",
    "cross_cultural_geometry",
    "dare_sparsity",
    "domain_signal_profile",
    "dora_decomposition",
    "fingerprints",
    "gate_detector",
    "generalized_procrustes",
    "geometry_fingerprint",
    "geometry_validation_suite",
    "gromov_wasserstein",
    "intersection_map_analysis",
    "intrinsic_dimension",
    "intrinsic_dimension_estimator",
    "invariant_convergence_analyzer",
    "invariant_layer_mapper",
    "manifold_clusterer",
    "manifold_dimensionality",
    "manifold_fidelity_sweep",
    "manifold_profile",
    "manifold_stitcher",
    "metaphor_convergence_analyzer",
    "model_fingerprints_projection",
    "path_geometry",
    "permutation_aligner",
    "persona_vector_monitor",
    "probes",
    "refinement_density",
    "refusal_direction_cache",
    "refusal_direction_detector",
    "shared_subspace_projector",
    "sparse_region_domains",
    "sparse_region_locator",
    "sparse_region_prober",
    "sparse_region_validator",
    "spectral_analysis",
    "tangent_space_alignment",
    "task_singular_vectors",
    "thermo_path_integration",
    "topological_fingerprint",
    "transfer_fidelity",
    "transport_guided_merger",
    "traversal_coherence",
    "verb_noun_classifier",
    "dimension_blender",
    "relative_representation",
}

# Attribute to submodule mapping for commonly used classes
# Format: "ExportedName": ("module_name", "actual_attr_name")
_ATTR_TO_MODULE = {
    "DoRAConfiguration": ("dora_decomposition", "DoRAConfig"),  # Alias for backward compat
    "DoRAConfig": ("dora_decomposition", "DoRAConfig"),
    "ChangeType": ("dora_decomposition", "ChangeType"),
    "DoRADecomposition": ("dora_decomposition", "DoRADecomposition"),
    "VectorMath": ("vector_math", "VectorMath"),
    "PathNode": ("path_geometry", "PathNode"),
    "PathSignature": ("path_geometry", "PathSignature"),
    "PermutationAligner": ("permutation_aligner", "PermutationAligner"),
    "DimensionBlender": ("dimension_blender", "DimensionBlender"),
    # Backend-aware matrix utilities
    "BackendMatrixUtils": ("backend_matrix_utils", "BackendMatrixUtils"),
    "ProcrustesResult": ("backend_matrix_utils", "ProcrustesResult"),
    # Signature base classes
    "SignatureMixin": ("signature_base", "SignatureMixin"),
    "LabeledSignatureMixin": ("signature_base", "LabeledSignatureMixin"),
}


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
    from .backend_matrix_utils import BackendMatrixUtils, ProcrustesResult
    from .dimension_blender import DimensionBlender
    from .dora_decomposition import ChangeType, DoRADecomposition
    from .dora_decomposition import DoRAConfig as DoRAConfiguration
    from .path_geometry import PathNode, PathSignature
    from .permutation_aligner import PermutationAligner
    from .signature_base import LabeledSignatureMixin, SignatureMixin
    from .vector_math import VectorMath
