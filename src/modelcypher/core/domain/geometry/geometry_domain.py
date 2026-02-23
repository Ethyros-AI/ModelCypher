"""Geometry domain classification.

Weight space is Euclidean (flat). Activation space is curved (Riemannian).
Cross-family falsification (2026-02-23, LFM2-350M + Qwen2.5-Coder-0.5B):
weight distortion below Gaussian random at all ranks; activation distortion
0.37-0.54. Evidence: results/weight_geometry/ artifacts.

This enum provides fail-fast enforcement at API boundaries.
"""

from enum import Enum


class GeometryDomain(str, Enum):
    """Domain classification for geometric operations.

    ACTIVATION: Curved manifold. Geodesic distances, curvature estimation,
        Frechet mean, intrinsic dimension via TwoNN.
    WEIGHT: Euclidean space. SVD, Procrustes, Frobenius norm, linear
        interpolation, spectral capacity analysis.
    """

    ACTIVATION = "activation"
    WEIGHT = "weight"
