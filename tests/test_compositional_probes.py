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

import pytest

from modelcypher.core.domain.geometry.compositional_probes import (
    CompositionalProbes,
    CompositionCategory,
    CompositionProbe,
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
    """Check consistency returns raw measurements."""
    probe = CompositionProbe("I WANT", ["I", "WANT"], CompositionCategory.mental_predicate)
    analysis = CompositionalProbes.analyze_composition([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], probe)
    result = CompositionalProbes.check_consistency([analysis], [analysis])

    # Raw measurements. The numbers ARE the answer.
    assert result.barycentric_correlation == pytest.approx(1.0, abs=1e-6)
    assert result.angular_correlation == pytest.approx(1.0, abs=1e-6)
    assert result.consistency_score == pytest.approx(1.0, abs=1e-6)


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
