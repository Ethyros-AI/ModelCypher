"""Tests for ranking-based approval metrics in dual_path.py.

These tests verify that the normalized approval metric correctly computes
ranking-based approval scores, which are more accurate than raw probability
for measuring whether the base model "approves" of tokens.
"""

import numpy as np
import pytest

from modelcypher.core.domain.inference.dual_path import compute_token_rank_metrics


class TestTokenRankMetrics:
    """Tests for compute_token_rank_metrics function."""

    def test_top_token_has_rank_zero(self):
        """The highest probability token should have rank 0."""
        probs = np.array([0.5, 0.3, 0.15, 0.05])

        rank, approval, hit = compute_token_rank_metrics(probs, token_id=0)

        assert rank == 0
        assert approval == 1.0  # Top token = max approval
        assert hit is True

    def test_second_token_has_rank_one(self):
        """The second highest probability token should have rank 1."""
        probs = np.array([0.5, 0.3, 0.15, 0.05])

        rank, approval, hit = compute_token_rank_metrics(probs, token_id=1)

        assert rank == 1
        # normalized = 1 - (1 / 3) = 0.667
        assert approval == pytest.approx(2 / 3, abs=0.01)
        assert hit is True

    def test_lowest_token_has_max_rank(self):
        """The lowest probability token should have rank vocab_size-1."""
        probs = np.array([0.5, 0.3, 0.15, 0.05])

        rank, approval, hit = compute_token_rank_metrics(probs, token_id=3)

        assert rank == 3  # vocab_size - 1
        assert approval == 0.0  # Bottom token = min approval
        assert hit is True  # rank 3 < top_k=10

    def test_normalized_approval_range(self):
        """Normalized approval should be in [0, 1]."""
        # Test with various distributions
        for _ in range(10):
            probs = np.random.rand(100)
            probs /= probs.sum()

            for token_id in range(100):
                _, approval, _ = compute_token_rank_metrics(probs, token_id)
                assert 0.0 <= approval <= 1.0

    def test_top_k_hit_threshold(self):
        """Top-K hit should respect the threshold parameter."""
        # Create a distribution with clear ranking
        probs = np.array([0.5, 0.25, 0.1, 0.08, 0.04, 0.02, 0.01])
        probs /= probs.sum()

        # Token 0 should always be in top-K
        rank0, _, hit0 = compute_token_rank_metrics(probs, token_id=0, top_k=1)
        assert rank0 == 0
        assert hit0 is True

        # Token 6 (lowest) with k=3 should be out
        rank6, _, hit6 = compute_token_rank_metrics(probs, token_id=6, top_k=3)
        assert rank6 == 6  # Lowest ranked
        assert hit6 is False

    def test_uniform_distribution(self):
        """Uniform distribution should give middle ranks."""
        n = 100
        probs = np.ones(n) / n  # Uniform

        # All tokens have same probability, so all should have rank 0
        # (no tokens have strictly higher probability)
        for i in range(n):
            rank, approval, _ = compute_token_rank_metrics(probs, token_id=i)
            assert rank == 0  # All equal = all are "top"
            assert approval == 1.0

    def test_large_vocabulary(self):
        """Should handle large vocabularies correctly."""
        vocab_size = 32000
        probs = np.random.rand(vocab_size)
        probs /= probs.sum()

        # Find the actual top token
        top_id = np.argmax(probs)
        rank, approval, hit = compute_token_rank_metrics(probs, top_id)

        assert rank == 0
        assert approval == 1.0
        assert hit is True

        # Find the bottom token
        bottom_id = np.argmin(probs)
        rank, approval, hit = compute_token_rank_metrics(probs, bottom_id)

        assert rank == vocab_size - 1
        assert approval == 0.0
        assert hit is False  # rank 31999 > top_k=10

    def test_single_token_vocab(self):
        """Single token vocabulary should have approval 1.0."""
        probs = np.array([1.0])

        rank, approval, hit = compute_token_rank_metrics(probs, token_id=0)

        assert rank == 0
        assert approval == 1.0
        assert hit is True

    def test_two_token_vocab(self):
        """Two token vocabulary should work correctly."""
        probs = np.array([0.7, 0.3])

        rank0, approval0, _ = compute_token_rank_metrics(probs, token_id=0)
        rank1, approval1, _ = compute_token_rank_metrics(probs, token_id=1)

        assert rank0 == 0
        assert approval0 == 1.0

        assert rank1 == 1
        assert approval1 == 0.0  # 1 - (1 / 1) = 0

    def test_ties_in_probability(self):
        """Tokens with equal probability should have same rank."""
        probs = np.array([0.4, 0.3, 0.3])  # Token 1 and 2 are tied

        rank1, approval1, _ = compute_token_rank_metrics(probs, token_id=1)
        rank2, approval2, _ = compute_token_rank_metrics(probs, token_id=2)

        # Both have same probability, so same rank (1 token has higher prob)
        assert rank1 == rank2 == 1
        assert approval1 == approval2

    def test_custom_top_k(self):
        """Custom top_k parameter should be respected."""
        probs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])

        # With k=2, only ranks 0 and 1 should be hits
        _, _, hit0 = compute_token_rank_metrics(probs, token_id=0, top_k=2)
        _, _, hit1 = compute_token_rank_metrics(probs, token_id=1, top_k=2)
        _, _, hit2 = compute_token_rank_metrics(probs, token_id=2, top_k=2)

        assert hit0 is True
        assert hit1 is True
        assert hit2 is False

    def test_monotonicity(self):
        """Higher probability tokens should have higher approval scores."""
        probs = np.array([0.4, 0.3, 0.2, 0.1])

        approvals = []
        for token_id in range(4):
            _, approval, _ = compute_token_rank_metrics(probs, token_id)
            approvals.append(approval)

        # Should be monotonically decreasing
        for i in range(len(approvals) - 1):
            assert approvals[i] >= approvals[i + 1]
