"""Geometry MCP tools.

This module contains geometry-related MCP tools extracted from server.py.
TODO: Migrate remaining geometry tools from server.py to reduce file size.

Current geometry tools in server.py (~2000+ lines):
- mc_geometry_validate
- mc_geometry_path_detect/compare
- mc_geometry_gromov_wasserstein
- mc_geometry_intrinsic_dimension
- mc_geometry_topological_fingerprint
- mc_geometry_sparse_domains/locate
- mc_geometry_refusal_pairs/detect
- mc_geometry_persona_traits/extract/drift
- mc_geometry_manifold_cluster/dimension/query
- mc_geometry_transport_merge/synthesize
- mc_geometry_invariant_map_layers/collapse_risk
- mc_geometry_atlas_inventory
- mc_geometry_training_status/history
- mc_geometry_safety_jailbreak_test
- mc_geometry_dare_sparsity
- mc_geometry_dora_decomposition
- mc_geometry_primes_list/probe/compare
- mc_geometry_crm_build/compare/sequence_inventory
- mc_geometry_stitch_analyze/apply
- mc_geometry_refinement_analyze
- mc_geometry_stitch_train
- mc_geometry_domain_profile
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .common import (
    READ_ONLY_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)

if TYPE_CHECKING:
    pass


def register_geometry_tools(ctx: ServiceContext) -> None:
    """Register geometry-related MCP tools.

    NOTE: Geometry tools are currently still in server.py.
    This function is a placeholder for future migration.
    """
    # TODO: Migrate geometry tools from server.py to this module
    # This will reduce server.py from ~4900 lines to ~2900 lines
    pass
