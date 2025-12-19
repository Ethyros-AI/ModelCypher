from __future__ import annotations

import math

import numpy as np

from modelcypher.core.domain.geometry_fingerprint import GeometricFingerprint


def test_gram_statistics_identity():
    gram = [1.0, 0.0, 0.0, 1.0]
    mean, std, gram_hash = GeometricFingerprint.gram_statistics(gram, n=2)
    assert mean == 0.0
    assert std == 0.0
    assert len(gram_hash) == 64


def test_spectral_radius_identity():
    gram = [1.0, 0.0, 0.0, 1.0]
    radius = GeometricFingerprint.estimate_spectral_radius(gram, n=2, iterations=8)
    assert math.isfinite(radius)
    assert abs(radius - 1.0) < 0.2


def test_condition_number_identity():
    gram = [1.0, 0.0, 0.0, 1.0]
    condition = GeometricFingerprint.estimate_condition_number(gram, n=2, iterations=12)
    assert abs(condition - 1.0) < 1e-3


def test_effective_dimensionality_identity():
    gram = np.eye(3, dtype=np.float32).reshape(-1).tolist()
    dim = GeometricFingerprint.estimate_effective_dimensionality(gram, n=3)
    assert abs(dim - 3.0) < 1e-3
