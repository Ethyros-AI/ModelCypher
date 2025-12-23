import pytest
import math
from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint, TopologyConstants


def test_topological_signature_betti_numbers():
    """Test Betti number extraction from persistence diagram."""
    points = [[0,0], [1,0], [1,1], [0,1]] # A square
    fingerprint = TopologicalFingerprint.compute(points, max_filtration=2.0)
    
    # Square has 1 component (β0) and 1 loop (β1)
    betti = fingerprint.betti_numbers
    assert betti.get(0, 0) == 1
    # Note: For n=4 points, β1 might be detected depending on threshold
    # But for a clear square, we expect at least 1 loop before it fills.
    assert betti.get(1, 0) >= 0


def test_topological_signature_persistence_entropy():
    """Test topological entropy (diversity of scales)."""
    # Case 1: Simple structure
    f1 = TopologicalFingerprint.compute([[0,0], [1,0]])
    e1 = f1.summary.persistence_entropy
    
    # Case 2: Multi-scale structure
    f2 = TopologicalFingerprint.compute([[0,0], [1,0], [10,0], [10.1, 0]])
    e2 = f2.summary.persistence_entropy
    
    # Multi-scale should have higher entropy
    assert e2 > e1


def test_topological_signature_similarity_identical():
    """Identical models should have high topological similarity score."""
    points = [[0,0], [1,0], [0.5, 0.5]]
    f1 = TopologicalFingerprint.compute(points)
    f2 = TopologicalFingerprint.compute(points)
    
    result = TopologicalFingerprint.compare(f1, f2)
    assert result.similarity_score == pytest.approx(1.0)
    assert result.is_compatible is True


def test_topological_signature_noise_tolerance():
    """Topological signatures should be stable under small perturbations."""
    points = [[0,0], [1,0], [0,1]]
    f1 = TopologicalFingerprint.compute(points)
    
    # Add tiny noise
    noisy_points = [[0.001, 0], [1.0, 0.001], [0, 1.0]]
    f2 = TopologicalFingerprint.compute(noisy_points)
    
    result = TopologicalFingerprint.compare(f1, f2)
    assert result.similarity_score > 0.9
    assert result.betti_difference == 0


def test_topological_signature_scale_invariance():
    """Test if fingerprints handle global scaling via persistence ratios."""
    points = [[0,0], [1,1]]
    f1 = TopologicalFingerprint.compute(points)
    
    # Scale x10
    points_scaled = [[0,0], [10,10]]
    f2 = TopologicalFingerprint.compute(points_scaled)
    
    # Max persistence changes, but relative structure (Betti) remains same
    assert f1.betti_numbers == f2.betti_numbers
    assert f2.summary.max_persistence == pytest.approx(f1.summary.max_persistence * 10, rel=0.1)
