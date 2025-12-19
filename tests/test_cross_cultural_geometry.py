from __future__ import annotations

from modelcypher.core.domain.cross_cultural_geometry import CrossCulturalGeometry


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
