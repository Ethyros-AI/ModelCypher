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
Domain layer - core business logic and models.

This module uses lazy imports to avoid loading all subpackages at import time.
Subpackages are loaded on first access.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# =============================================================================
# Subdomain package registry for lazy loading
# =============================================================================

_SUBPACKAGES = {
    "adapters",
    "agents",
    "dataset",
    "dynamics",
    "entropy",
    "evaluation",
    "geometry",
    "inference",
    "merging",
    "research",
    "safety",
    "semantics",
    "thermo",
    "thermodynamics",
    "training",
    "validation",
}

# =============================================================================
# Root-level module registry
# =============================================================================

_ROOT_MODULES = {
    "chat_template",
    "dataset_export_formatter",
    "dataset_file_enumerator",
    "dataset_validation",
    "dataset_validator",
    "model_search",
    "models",
    "settings",
    "storage_usage",
}

# =============================================================================
# Backward compatibility: attribute to module mapping
# Maps class/function names to their source modules for lazy loading
# =============================================================================

_COMPAT_ATTRS = {
    # From geometry subpackage - commonly used classes
    "AffineStitchingLayer": ("geometry.affine_stitching_layer", "AffineStitchingLayer"),
    "CompositionalProbes": ("geometry.compositional_probes", "CompositionalProbes"),
    "CrossArchitectureLayerMatcher": ("geometry.cross_architecture_layer_matcher", "CrossArchitectureLayerMatcher"),
    "DareSparsity": ("geometry.dare_sparsity", "DareSparsity"),
    "GateDetector": ("geometry.gate_detector", "GateDetector"),
    "GeneralizedProcrustes": ("geometry.generalized_procrustes", "GeneralizedProcrustes"),
    "GeometryValidationSuite": ("geometry.geometry_validation_suite", "GeometryValidationSuite"),
    "IntrinsicDimensionEstimator": ("geometry.intrinsic_dimension_estimator", "IntrinsicDimensionEstimator"),
    "ManifoldClusterer": ("geometry.manifold_clusterer", "ManifoldClusterer"),
    "ManifoldDimensionality": ("geometry.manifold_dimensionality", "ManifoldDimensionality"),
    "ManifoldProfile": ("geometry.manifold_profile", "ManifoldProfile"),
    "PersonaVectorMonitor": ("geometry.persona_vector_monitor", "PersonaVectorMonitor"),
    "RefusalDirectionCache": ("geometry.refusal_direction_cache", "RefusalDirectionCache"),
    "RefusalDirectionDetector": ("geometry.refusal_direction_detector", "RefusalDirectionDetector"),
    "SparseRegionDomains": ("geometry.sparse_region_domains", "SparseRegionDomains"),
    "SparseRegionLocator": ("geometry.sparse_region_locator", "SparseRegionLocator"),
    "SparseRegionProber": ("geometry.sparse_region_prober", "SparseRegionProber"),
    "SparseRegionValidator": ("geometry.sparse_region_validator", "SparseRegionValidator"),
    "ThermoPathIntegration": ("geometry.thermo_path_integration", "ThermoPathIntegration"),
    "TopologicalFingerprint": ("geometry.topological_fingerprint", "TopologicalFingerprint"),
    "TransferFidelity": ("geometry.transfer_fidelity", "TransferFidelity"),
    # From training subpackage
    "GeometricTrainingMetrics": ("training.geometric_training_metrics", "GeometricTrainingMetrics"),
}


def __getattr__(name: str):
    """Lazy load subpackages, modules, and backward-compatible attributes."""
    # Check subpackages first
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)

    # Check root-level modules
    if name in _ROOT_MODULES:
        return importlib.import_module(f".{name}", __name__)

    # Check backward compatibility attributes
    if name in _COMPAT_ATTRS:
        module_path, attr_name = _COMPAT_ATTRS[name]
        module = importlib.import_module(f".{module_path}", __name__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available subpackages, modules, and attributes."""
    return list(_SUBPACKAGES) + list(_ROOT_MODULES) + list(_COMPAT_ATTRS.keys())


# TYPE_CHECKING for static analysis only
if TYPE_CHECKING:
    from . import (
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
