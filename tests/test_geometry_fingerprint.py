from __future__ import annotations

import math

import numpy as np

from modelcypher.core.domain.geometry.geometry_fingerprint import (
    AnchorSet,
    CompositionStrategy,
    FitPrediction,
    GeometricFingerprint,
)


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


def test_gram_statistics_invalid_size():
    """Invalid gram size returns zeros."""
    mean, std, gram_hash = GeometricFingerprint.gram_statistics([1.0], n=2)
    assert mean == 0.0
    assert std == 0.0
    assert gram_hash == ""


def test_predict_fit_identical_fingerprints():
    """Identical fingerprints have high fit score."""
    fp = GeometricFingerprint.placeholder
    prediction = fp.predict_fit(fp)
    assert isinstance(prediction, FitPrediction)
    assert prediction.fit_score > 0


def test_fit_prediction_properties():
    """FitPrediction properties work correctly."""
    prediction = FitPrediction(
        fit_score=0.75,
        location_score=0.8,
        direction_score=0.7,
        rotation_penalty=0.1,
    )
    assert prediction.is_compatible
    assert prediction.assessment in ["excellent", "good", "moderate", "poor", "incompatible"]


def test_suggest_composition_strategy_single_fingerprint():
    """Single fingerprint suggests automatic strategy."""
    strategy = GeometricFingerprint.suggest_composition_strategy([GeometricFingerprint.placeholder])
    assert strategy == CompositionStrategy.automatic


def test_symmetric_eigenvalues_identity():
    """Identity matrix has all eigenvalues = 1."""
    gram = [1.0, 0.0, 0.0, 1.0]
    eigenvalues = GeometricFingerprint.symmetric_eigenvalues(gram, n=2)
    assert eigenvalues is not None
    assert all(abs(ev - 1.0) < 1e-6 for ev in eigenvalues)


def test_anchor_set_enum_values():
    """All anchor set values exist."""
    values = [a.value for a in AnchorSet]
    assert "semanticPrimes" in values
    assert "computationalGates" in values


def test_composition_strategy_enum_values():
    """All composition strategy values exist."""
    values = [s.value for s in CompositionStrategy]
    assert "weightBlending" in values
    assert "attentionRouting" in values
    assert "sequential" in values
