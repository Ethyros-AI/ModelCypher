"""
Domain layer - core business logic and models.

This module aggregates all domain subpackages and provides
a unified import interface for CLI and MCP tools.
"""
from __future__ import annotations

# =============================================================================
# Core Subdomain Packages
# =============================================================================

# Import all subdomain packages to ensure they're loaded
from modelcypher.core.domain import (
    adapters,
    agents,
    dataset,
    dynamics,
    entropy,
    evaluation,
    geometry,
    inference,
    merging,
    research,
    safety,
    semantics,
    thermo,
    thermodynamics,
    training,
    validation,
)

# =============================================================================
# Re-exported Stubs (backward compatibility for legacy import paths)
# =============================================================================

# These import from their canonical locations in subdirectories
from .geometry.affine_stitching_layer import *  # noqa: F401,F403
from .geometry.compositional_probes import *  # noqa: F401,F403
from .geometry.cross_architecture_layer_matcher import *  # noqa: F401,F403
from .geometry.dare_sparsity import *  # noqa: F401,F403
from .geometry.gate_detector import *  # noqa: F401,F403
from .geometry.generalized_procrustes import *  # noqa: F401,F403
from .geometry.geometry_validation_suite import *  # noqa: F401,F403
from .geometry.intrinsic_dimension_estimator import *  # noqa: F401,F403
from .geometry.manifold_clusterer import *  # noqa: F401,F403
from .geometry.manifold_dimensionality import *  # noqa: F401,F403
from .geometry.manifold_profile import *  # noqa: F401,F403
from .geometry.persona_vector_monitor import *  # noqa: F401,F403
from .geometry.refusal_direction_cache import *  # noqa: F401,F403
from .geometry.refusal_direction_detector import *  # noqa: F401,F403
from .geometry.sparse_region_domains import *  # noqa: F401,F403
from .geometry.sparse_region_locator import *  # noqa: F401,F403
from .geometry.sparse_region_prober import *  # noqa: F401,F403
from .geometry.sparse_region_validator import *  # noqa: F401,F403
from .geometry.thermo_path_integration import *  # noqa: F401,F403
from .geometry.topological_fingerprint import *  # noqa: F401,F403
from .geometry.transfer_fidelity import *  # noqa: F401,F403
from .training.geometric_training_metrics import *  # noqa: F401,F403

# =============================================================================
# Root-level modules
# =============================================================================

from .chat_template import *  # noqa: F401,F403
from .dataset_export_formatter import *  # noqa: F401,F403
from .dataset_file_enumerator import *  # noqa: F401,F403
from .dataset_validation import *  # noqa: F401,F403
from .dataset_validator import *  # noqa: F401,F403
from .model_search import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .settings import *  # noqa: F401,F403
from .storage_usage import *  # noqa: F401,F403
from .training import *  # noqa: F401,F403
