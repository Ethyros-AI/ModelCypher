from __future__ import annotations

import pytest

from modelcypher.core.domain.compositional_probes import (
    CompositionCategory,
    CompositionProbe,
    CompositionalProbes,
)


def test_analyze_composition_basic() -> None:
    probe = CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.mental_predicate)
    components = [[1.0, 0.0], [0.0, 1.0]]
    composition = [0.5, 0.5]
    analysis = CompositionalProbes.analyze_composition(composition, components, probe)

    assert analysis.barycentric_weights[0] == pytest.approx(0.5, abs=1e-3)
    assert analysis.barycentric_weights[1] == pytest.approx(0.5, abs=1e-3)
    assert analysis.residual_norm == pytest.approx(0.0, abs=1e-6)
    assert analysis.is_compositional is True


def test_check_consistency_identical() -> None:
    probe = CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.mental_predicate)
    analysis = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe)
    result = CompositionalProbes.check_consistency([analysis], [analysis])

    assert result.barycentric_correlation == pytest.approx(1.0, abs=1e-6)
    assert result.angular_correlation == pytest.approx(1.0, abs=1e-6)
    assert result.is_compatible is True


def test_analyze_all_probes_custom() -> None:
    probe = CompositionProbe("TEST", ["A", "B"], CompositionCategory.relational)
    prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    composition_embeddings = {"TEST": [0.5, 0.5]}
    analyses = CompositionalProbes.analyze_all_probes(
        prime_embeddings=prime_embeddings,
        composition_embeddings=composition_embeddings,
        probes=[probe],
    )
    assert len(analyses) == 1
