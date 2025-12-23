import pytest
from modelcypher.core.domain.geometry.topological_fingerprint import TopologicalFingerprint, PersistencePoint


def test_persistent_homology_point_persistence():
    """Test persistence calculation (death - birth)."""
    p = PersistencePoint(birth=1.2, death=3.5, dimension=0)
    assert p.persistence == pytest.approx(2.3)


def test_persistent_homology_filtration_death_clamping():
    """Test that points persist to max_filtration if they don't die."""
    points = [[0,0], [1,0]]
    max_filt = 5.0
    fingerprint = TopologicalFingerprint.compute(points, max_filtration=max_filt)
    
    # One component must survive until max_filtration
    points0 = fingerprint.diagram.points
    assert any(p.death == max_filt for p in points0)


def test_persistent_homology_bottleneck_stability():
    """Bottleneck distance should satisfy the stable property."""
    # Stability: d_B(diag(X), diag(Y)) <= L_inf(X, Y)
    # Not strictly true for Rips but a good heuristic
    x = [[0.0, 0.0], [1.0, 0.0]]
    y = [[0.0, 0.0], [1.1, 0.0]] # L_inf noise = 0.1
    
    f1 = TopologicalFingerprint.compute(x)
    f2 = TopologicalFingerprint.compute(y)
    
    result = TopologicalFingerprint.compare(f1, f2)
    # Bottleneck distance should be bounded by noise (roughly)
    assert result.bottleneck_distance <= 0.2 # Allow some slack for D_B


def test_persistent_homology_1dim_void_detection():
    """Test detection of 1D persistence (loops)."""
    # 6 points forming a hexagon
    points = [
        [1, 0], [0.5, 0.866], [-0.5, 0.866],
        [-1, 0], [-0.5, -0.866], [0.5, -0.866]
    ]
    fingerprint = TopologicalFingerprint.compute(points, max_dimension=1, num_steps=100)
    
    # Should find at least one significant cycle
    assert fingerprint.summary.cycle_count >= 1


def test_persistent_homology_betti_thresholding():
    """Test that Betti numbers respect persistence threshold."""
    # Point cloud with one small noise component and one major component
    # Diagram: (0, 0.1), (0, 10.0)
    from modelcypher.core.domain.geometry.topological_fingerprint import PersistenceDiagram, PersistencePoint
    
    diag = PersistenceDiagram([
        PersistencePoint(0.0, 0.1, 0),
        PersistencePoint(0.0, 10.0, 0)
    ])
    
    betti_none = diag.betti_numbers(persistence_threshold=0.0)
    betti_strict = diag.betti_numbers(persistence_threshold=0.5)
    
    assert betti_none[0] == 2
    assert betti_strict[0] == 1
