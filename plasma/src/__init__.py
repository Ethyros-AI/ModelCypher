# Plasma dynamics geometry analysis

from .data_loader import PlasmaShot, create_synthetic_shot
from .geometry_tools import (
    GeometricProfile,
    PCAManifold,
    ManifoldGradient,
    compute_expansion_ratio,
    compute_local_dimension,
    compute_spectral_entropy,
    compute_jacobian_approximation,
    compute_pca_manifold,
    compute_gradient_to_manifold,
    compute_trajectory_manifold_analysis,
    analyze_shot,
    compare_profiles,
)

__all__ = [
    "PlasmaShot",
    "create_synthetic_shot",
    "GeometricProfile",
    "PCAManifold",
    "ManifoldGradient",
    "compute_expansion_ratio",
    "compute_local_dimension",
    "compute_spectral_entropy",
    "compute_jacobian_approximation",
    "compute_pca_manifold",
    "compute_gradient_to_manifold",
    "compute_trajectory_manifold_analysis",
    "analyze_shot",
    "compare_profiles",
]
