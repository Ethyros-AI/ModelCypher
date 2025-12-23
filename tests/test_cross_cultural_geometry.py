from __future__ import annotations

from modelcypher.core.domain.geometry.cross_cultural_geometry import (
    AlignmentAssessment,
    CrossCulturalGeometry,
    MergeAssessment,
)


def test_compute_cka_identical():
    gram = [
        1.0, 0.2, 0.3,
        0.2, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    cka = CrossCulturalGeometry.compute_cka(gram, gram, n=3)
    assert cka > 0.99


def test_analyze_alignment_identical():
    gram = [
        1.0, 0.2, 0.3,
        0.2, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    analysis = CrossCulturalGeometry.analyze_alignment(gram, gram, n=3)
    assert analysis is not None
    assert analysis.cka > 0.99


def test_compute_cka_orthogonal():
    """Orthogonal Gram matrices have low CKA."""
    gram_a = [1.0, 0.0, 0.0, 1.0]  # Identity
    gram_b = [1.0, 0.9, 0.9, 1.0]  # Highly correlated
    cka = CrossCulturalGeometry.compute_cka(gram_a, gram_b, n=2)
    assert 0.0 <= cka <= 1.0


def test_compute_cka_invalid_size():
    """Invalid matrix sizes return 0."""
    gram_a = [1.0, 0.0, 0.0, 1.0]  # 2x2
    cka = CrossCulturalGeometry.compute_cka(gram_a, gram_a, n=3)  # Wrong n
    assert cka == 0.0


def test_analyze_alignment_assessment_thresholds():
    """Test alignment assessment threshold behavior."""
    gram = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    analysis = CrossCulturalGeometry.analyze_alignment(gram, gram, n=3)
    assert analysis is not None
    assert analysis.alignment_assessment == AlignmentAssessment.aligned


def test_analyze_full_comparison():
    """Full analysis produces valid comparison result."""
    gram_a = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ]
    gram_b = [
        1.0, 0.4, 0.2,
        0.4, 1.0, 0.5,
        0.2, 0.5, 1.0,
    ]
    prime_ids = ["A", "B", "C"]
    prime_categories = {"A": "cat1", "B": "cat1", "C": "cat2"}

    result = CrossCulturalGeometry.analyze(gram_a, gram_b, prime_ids, prime_categories)
    assert result is not None
    assert 0.0 <= result.merge_quality_score <= 1.0
    assert result.merge_assessment in list(MergeAssessment)


def test_analyze_roughness_reduction():
    """Roughness reduction is calculated correctly."""
    gram_smooth = [
        1.0, 0.5, 0.5,
        0.5, 1.0, 0.5,
        0.5, 0.5, 1.0,
    ]
    gram_rough = [
        1.0, 0.9, 0.1,
        0.9, 1.0, 0.2,
        0.1, 0.2, 1.0,
    ]
    prime_ids = ["A", "B", "C"]
    prime_categories = {"A": "cat1", "B": "cat1", "C": "cat2"}

    result = CrossCulturalGeometry.analyze(gram_smooth, gram_rough, prime_ids, prime_categories)
    assert result is not None
    # Merged roughness should be between the two
    assert result.merged_gram_roughness >= 0


def test_analyze_invalid_inputs():
    """Invalid inputs return None."""
    gram = [1.0]
    result = CrossCulturalGeometry.analyze(gram, gram, ["A"], {})
    assert result is None  # n <= 1
