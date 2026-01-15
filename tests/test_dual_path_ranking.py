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

"""Tests for rank geometry metrics in dual_path.py.

These tests verify rank geometry in logit space (ordering and frontier),
which is derived directly from score geometry rather than probability.
"""

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.infrastructure.dual_path_mlx import compute_token_rank_metrics


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


class TestTokenRankMetrics:
    """Tests for compute_token_rank_metrics function."""

    def test_top_token_has_rank_zero(self):
        """The highest score token should have rank 0."""
        backend = get_default_backend()
        scores = backend.array([0.5, 0.3, 0.15, 0.05])

        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, token_id=0)

        assert rank == 0
        assert abs(rank_fraction - 1.0) <= _eps(backend, rank_fraction, 1.0)
        assert frontier_hit is True

    def test_second_token_has_rank_one(self):
        """The second highest score token should have rank 1."""
        backend = get_default_backend()
        scores = backend.array([0.5, 0.3, 0.15, 0.05])
        eps = machine_epsilon(backend, scores)

        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, token_id=1)

        assert rank == 1
        assert abs(rank_fraction - (2 / 3)) < eps
        assert frontier_hit is False

    def test_lowest_token_has_max_rank(self):
        """The lowest score token should have rank vocab_size-1."""
        backend = get_default_backend()
        scores = backend.array([0.5, 0.3, 0.15, 0.05])

        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, token_id=3)

        assert rank == 3  # vocab_size - 1
        assert abs(rank_fraction) <= _eps(backend, rank_fraction)
        assert frontier_hit is False

    def test_rank_fraction_range(self):
        """Rank fraction should be in [0, 1]."""
        backend = get_default_backend()
        # Test with various distributions
        for seed in range(10):
            backend.random_seed(seed)
            scores = backend.random_uniform(shape=(100,))
            backend.eval(scores)
            max_arr = backend.max(scores)
            backend.eval(max_arr)
            eps = _eps(backend, float(backend.to_scalar(max_arr)))

            for token_id in range(100):
                _, rank_fraction, _ = compute_token_rank_metrics(scores, token_id)
                assert rank_fraction >= -eps
                assert rank_fraction <= 1.0 + eps

    def test_frontier_gap_derivation(self):
        """Frontier hit should follow the largest relative gap."""
        backend = get_default_backend()
        scores = backend.array([10.0, 9.5, 2.0, 1.9])

        rank0, _, hit0 = compute_token_rank_metrics(scores, token_id=0)
        rank1, _, hit1 = compute_token_rank_metrics(scores, token_id=1)
        rank2, _, hit2 = compute_token_rank_metrics(scores, token_id=2)

        assert rank0 == 0
        assert rank1 == 1
        assert rank2 == 2
        assert hit0 is True
        assert hit1 is True
        assert hit2 is False

    def test_uniform_distribution(self):
        """Uniform scores should give equal ranks and full frontier."""
        backend = get_default_backend()
        n = 100
        scores = backend.ones((n,))
        backend.eval(scores)

        # All tokens have same score, so all should have rank 0
        # (no tokens have strictly higher score)
        for i in range(n):
            rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, token_id=i)
            assert rank == 0  # All equal = all are "top"
            assert abs(rank_fraction - 1.0) <= _eps(backend, rank_fraction, 1.0)
            assert frontier_hit is True

    def test_large_vocabulary(self):
        """Should handle large vocabularies correctly."""
        backend = get_default_backend()
        vocab_size = 32000
        backend.random_seed(42)
        scores = backend.random_uniform(shape=(vocab_size,))
        backend.eval(scores)

        # Find the actual top token
        top_id = int(backend.to_scalar(backend.argmax(scores)))
        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, top_id)

        assert rank == 0
        assert abs(rank_fraction - 1.0) <= _eps(backend, rank_fraction, 1.0)
        assert frontier_hit is True

        # Find the bottom token
        bottom_id = int(backend.to_scalar(backend.argmin(scores)))
        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, bottom_id)

        assert rank == vocab_size - 1
        assert abs(rank_fraction) <= _eps(backend, rank_fraction)
        assert frontier_hit is False

    def test_single_token_vocab(self):
        """Single token vocabulary should have rank fraction 1.0."""
        backend = get_default_backend()
        scores = backend.array([1.0])

        rank, rank_fraction, frontier_hit = compute_token_rank_metrics(scores, token_id=0)

        assert rank == 0
        assert abs(rank_fraction - 1.0) <= _eps(backend, rank_fraction, 1.0)
        assert frontier_hit is True

    def test_two_token_vocab(self):
        """Two token vocabulary should work correctly."""
        backend = get_default_backend()
        scores = backend.array([0.7, 0.3])

        rank0, rank_fraction0, hit0 = compute_token_rank_metrics(scores, token_id=0)
        rank1, rank_fraction1, hit1 = compute_token_rank_metrics(scores, token_id=1)

        assert rank0 == 0
        assert abs(rank_fraction0 - 1.0) <= _eps(backend, rank_fraction0, 1.0)
        assert hit0 is True

        assert rank1 == 1
        assert abs(rank_fraction1) <= _eps(backend, rank_fraction1)
        assert hit1 is False

    def test_ties_in_probability(self):
        """Tokens with equal scores should have same rank."""
        backend = get_default_backend()
        scores = backend.array([0.4, 0.3, 0.3])  # Token 1 and 2 are tied

        rank1, rank_fraction1, hit1 = compute_token_rank_metrics(scores, token_id=1)
        rank2, rank_fraction2, hit2 = compute_token_rank_metrics(scores, token_id=2)

        # Both have same score, so same rank (1 token has higher score)
        assert rank1 == rank2 == 1
        assert abs(rank_fraction1 - rank_fraction2) <= _eps(
            backend, rank_fraction1, rank_fraction2
        )
        assert hit1 is hit2 is False

    def test_monotonicity(self):
        """Higher score tokens should have higher rank fractions."""
        backend = get_default_backend()
        scores = backend.array([0.4, 0.3, 0.2, 0.1])

        rank_fractions = []
        for token_id in range(4):
            _, rank_fraction, _ = compute_token_rank_metrics(scores, token_id)
            rank_fractions.append(rank_fraction)

        # Should be monotonically decreasing
        eps = _eps(backend, *rank_fractions)
        for i in range(len(rank_fractions) - 1):
            assert rank_fractions[i] + eps >= rank_fractions[i + 1]
