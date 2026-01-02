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

from __future__ import annotations

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.compositional_probes import (
    CompositionalProbes,
    CompositionCategory,
    CompositionProbe,
)


def _scalar_tol():
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


def test_analyze_composition_basic() -> None:
    probe = CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE)
    components = [[1.0, 0.0], [0.0, 1.0]]
    composition = [0.5, 0.5]
    analysis = CompositionalProbes.analyze_composition(composition, components, probe)

    tol = _scalar_tol()
    assert abs(analysis.barycentric_weights[0] - 0.5) <= tol
    assert abs(analysis.barycentric_weights[1] - 0.5) <= tol
    assert abs(analysis.residual_norm) <= tol
    assert analysis.is_compositional is True


def test_check_consistency_identical() -> None:
    """Check consistency returns raw measurements."""
    probe = CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.MENTAL_PREDICATE)
    analysis = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe)
    result = CompositionalProbes.check_consistency([analysis], [analysis])

    # Raw measurements. The numbers ARE the answer.
    tol = _scalar_tol()
    assert abs(result.barycentric_correlation - 1.0) <= tol
    assert abs(result.angular_correlation - 1.0) <= tol
    assert abs(result.consistency_score - 1.0) <= tol


def test_analyze_all_probes_custom() -> None:
    probe = CompositionProbe("TEST", ["A", "B"], CompositionCategory.RELATIONAL)
    prime_embeddings = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    composition_embeddings = {"TEST": [0.5, 0.5]}
    analyses = CompositionalProbes.analyze_all_probes(
        prime_embeddings=prime_embeddings,
        composition_embeddings=composition_embeddings,
        probes=[probe],
    )
    assert len(analyses) == 1
