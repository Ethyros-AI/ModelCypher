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

"""Tests for outer similarity metrics (Kucukahmetler et al. 2026).

Tests focus on mathematical identities and known-value verification,
not heuristic thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.relative_representation import (
    OuterSimilarityResult,
    compute_outer_similarity,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    return get_default_backend()


class TestIdenticalInputs:
    """When both relative representations are identical, all metrics = 1.0."""

    def test_perfect_agreement(self, backend: "Backend") -> None:
        rel = backend.array([
            [0.9, 0.3, 0.1, 0.5],
            [0.2, 0.8, 0.4, 0.6],
            [0.1, 0.1, 0.9, 0.3],
        ])
        result = compute_outer_similarity(rel, rel, backend)

        assert isinstance(result, OuterSimilarityResult)
        assert result.cosine_rss == pytest.approx(1.0, abs=1e-5)
        assert result.spearman_rank == pytest.approx(1.0, abs=1e-5)
        assert result.top1_agreement == pytest.approx(1.0, abs=1e-5)
        assert result.n_samples == 3
        assert result.n_anchors == 4


class TestOrthogonalInputs:
    """Orthogonal rows should give cosine_rss = 0.0."""

    def test_orthogonal_cosine_zero(self, backend: "Backend") -> None:
        # Construct two [2, 4] matrices where corresponding rows are orthogonal
        rel_1 = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        rel_2 = backend.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        result = compute_outer_similarity(rel_1, rel_2, backend)
        assert result.cosine_rss == pytest.approx(0.0, abs=1e-5)


class TestReversedRanks:
    """Negated inputs reverse rank order: spearman = -1, top1 = 0."""

    def test_negation_reverses_everything(self, backend: "Backend") -> None:
        rel = backend.array([
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.9, 0.7, 0.5, 0.3, 0.1],
            [0.2, 0.4, 0.6, 0.8, 1.0],
        ])
        neg_rel = rel * -1.0
        result = compute_outer_similarity(rel, neg_rel, backend)

        # Negation reverses all ranks
        assert result.spearman_rank == pytest.approx(-1.0, abs=1e-5)
        # argmax(rel) vs argmax(-rel): max becomes min, never agree
        assert result.top1_agreement == pytest.approx(0.0, abs=1e-5)
        # Cosine of negated vectors = -1
        assert result.cosine_rss == pytest.approx(-1.0, abs=1e-5)


class TestSmallAnchors:
    """Spearman is undefined (returns 0.0) for n_anchors < 3."""

    def test_two_anchors_spearman_zero(self, backend: "Backend") -> None:
        rel_1 = backend.array([[0.5, 0.8], [0.3, 0.9]])
        rel_2 = backend.array([[0.6, 0.7], [0.4, 0.8]])
        result = compute_outer_similarity(rel_1, rel_2, backend)
        assert result.spearman_rank == 0.0
        # Cosine and top-1 should still work
        assert result.n_anchors == 2


class TestShapeMismatch:
    """Mismatched shapes should raise ValueError."""

    def test_different_n_samples(self, backend: "Backend") -> None:
        rel_1 = backend.array([[0.1, 0.2, 0.3]])
        rel_2 = backend.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(ValueError, match="matching shapes"):
            compute_outer_similarity(rel_1, rel_2, backend)

    def test_different_n_anchors(self, backend: "Backend") -> None:
        rel_1 = backend.array([[0.1, 0.2, 0.3]])
        rel_2 = backend.array([[0.1, 0.2]])
        with pytest.raises(ValueError, match="matching shapes"):
            compute_outer_similarity(rel_1, rel_2, backend)

    def test_1d_input(self, backend: "Backend") -> None:
        rel = backend.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="2D"):
            compute_outer_similarity(rel, rel, backend)


class TestNearZeroNorms:
    """Rows with near-zero norms should be stable via epsilon floor."""

    def test_zero_row_stable(self, backend: "Backend") -> None:
        rel_1 = backend.array([
            [1e-30, 1e-30, 1e-30, 1e-30],
            [0.5, 0.3, 0.7, 0.1],
        ])
        rel_2 = backend.array([
            [0.5, 0.3, 0.7, 0.1],
            [0.5, 0.3, 0.7, 0.1],
        ])
        result = compute_outer_similarity(rel_1, rel_2, backend)
        # Should not NaN or raise
        assert -1.0 <= result.cosine_rss <= 1.0
        assert -1.0 <= result.spearman_rank <= 1.0
        assert 0.0 <= result.top1_agreement <= 1.0


class TestKnownValues:
    """Handcrafted [3, 4] matrices with manually computed expected values."""

    def test_known_3x4(self, backend: "Backend") -> None:
        # rel_1:
        #   row 0: [1, 2, 3, 4]  -> rank [0,1,2,3], argmax=3
        #   row 1: [4, 3, 2, 1]  -> rank [3,2,1,0], argmax=0
        #   row 2: [1, 1, 1, 5]  -> rank [0,1,2,3] (argsort stable), argmax=3
        rel_1 = backend.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 5.0],
        ])
        # rel_2:
        #   row 0: [2, 1, 4, 3]  -> rank [1,0,3,2], argmax=2
        #   row 1: [3, 4, 1, 2]  -> rank [2,3,0,1], argmax=1
        #   row 2: [1, 1, 1, 5]  -> rank [0,1,2,3], argmax=3
        rel_2 = backend.array([
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 1.0, 1.0, 5.0],
        ])

        result = compute_outer_similarity(rel_1, rel_2, backend)

        # Top-1 agreement: row 0 (3 vs 2: no), row 1 (0 vs 1: no), row 2 (3 vs 3: yes)
        assert result.top1_agreement == pytest.approx(1.0 / 3.0, abs=1e-5)

        # Cosine RSS: compute manually
        # row 0: dot=1*2 + 2*1 + 3*4 + 4*3 = 28, |a|=sqrt(30), |b|=sqrt(30)
        #         cos = 28/30 = 0.9333...
        # row 1: dot=4*3 + 3*4 + 2*1 + 1*2 = 28, |a|=sqrt(30), |b|=sqrt(30)
        #         cos = 28/30 = 0.9333...
        # row 2: dot=1*1 + 1*1 + 1*1 + 5*5 = 28, |a|=sqrt(28), |b|=sqrt(28)
        #         cos = 28/28 = 1.0
        # mean = (0.9333 + 0.9333 + 1.0) / 3 = 0.9555...
        expected_cos = (28.0 / 30.0 + 28.0 / 30.0 + 1.0) / 3.0
        assert result.cosine_rss == pytest.approx(expected_cos, abs=1e-4)

        # Spearman: d^2 formula with n=4
        # row 0: ranks_1=[0,1,2,3], ranks_2=[1,0,3,2], d=[−1,1,−1,1], d^2=4
        #         rho = 1 - 6*4/(4*15) = 1 - 24/60 = 0.6
        # row 1: ranks_1=[3,2,1,0], ranks_2=[2,3,0,1], d=[1,−1,1,−1], d^2=4
        #         rho = 1 - 24/60 = 0.6
        # row 2: ranks_1=[0,1,2,3], ranks_2=[0,1,2,3], d=[0,0,0,0], d^2=0
        #         rho = 1 - 0 = 1.0
        # mean = (0.6 + 0.6 + 1.0) / 3 = 0.7333...
        expected_spearman = (0.6 + 0.6 + 1.0) / 3.0
        assert result.spearman_rank == pytest.approx(expected_spearman, abs=1e-4)


class TestSingleSample:
    """Single sample should work for all metrics."""

    def test_single_sample_identical(self, backend: "Backend") -> None:
        rel = backend.array([[0.3, 0.7, 0.1, 0.9]])
        result = compute_outer_similarity(rel, rel, backend)
        assert result.cosine_rss == pytest.approx(1.0, abs=1e-5)
        assert result.top1_agreement == pytest.approx(1.0, abs=1e-5)
        assert result.n_samples == 1
