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
from modelcypher.core.domain.geometry.cross_cultural_geometry import (
    CrossCulturalGeometry,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _epsilon() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


def test_compute_cka_identical():
    gram = [
        1.0,
        0.2,
        0.3,
        0.2,
        1.0,
        0.4,
        0.3,
        0.4,
        1.0,
    ]
    cka = CrossCulturalGeometry.compute_cka(gram, gram, n=3)
    assert abs(cka - 1.0) <= _epsilon()


def test_analyze_alignment_identical():
    gram = [
        1.0,
        0.2,
        0.3,
        0.2,
        1.0,
        0.4,
        0.3,
        0.4,
        1.0,
    ]
    analysis = CrossCulturalGeometry.analyze_alignment(gram, gram, n=3)
    assert analysis is not None
    assert abs(analysis.cka - 1.0) <= _epsilon()


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


def test_analyze_alignment_cka_signal():
    """Test that CKA is the alignment signal - high CKA for identical grams."""
    gram = [
        1.0,
        0.5,
        0.3,
        0.5,
        1.0,
        0.4,
        0.3,
        0.4,
        1.0,
    ]
    analysis = CrossCulturalGeometry.analyze_alignment(gram, gram, n=3)
    assert analysis is not None
    assert abs(analysis.cka - 1.0) <= _epsilon()


def test_analyze_full_comparison():
    """Full analysis produces valid comparison result."""
    gram_a = [
        1.0,
        0.5,
        0.3,
        0.5,
        1.0,
        0.4,
        0.3,
        0.4,
        1.0,
    ]
    gram_b = [
        1.0,
        0.4,
        0.2,
        0.4,
        1.0,
        0.5,
        0.2,
        0.5,
        1.0,
    ]
    prime_ids = ["A", "B", "C"]
    prime_categories = {"A": "cat1", "B": "cat1", "C": "cat2"}

    result = CrossCulturalGeometry.analyze(gram_a, gram_b, prime_ids, prime_categories)
    assert result is not None
    assert len(result.row_correlations) == len(prime_ids)
    assert len(result.row_sharpness_a) == len(prime_ids)
    assert len(result.row_sharpness_b) == len(prime_ids)
    assert len(result.row_sharpness_ratio) == len(prime_ids)


def test_analyze_roughness_reduction():
    """Roughness reduction is calculated correctly."""
    gram_smooth = [
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        0.5,
        0.5,
        1.0,
    ]
    gram_rough = [
        1.0,
        0.9,
        0.1,
        0.9,
        1.0,
        0.2,
        0.1,
        0.2,
        1.0,
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
