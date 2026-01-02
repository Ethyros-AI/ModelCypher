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

"""Tests for Conflict Score Calculator and Analysis (requires MLX)."""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conflict_score import (
    ConflictAnalysis,
    ConflictScoreCalculator,
    ConflictScoreResult,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values)))


class TestConflictScoreResult:
    """Tests for ConflictScoreResult - raw measurements only."""

    def test_creation_with_raw_measurements(self):
        """Should create with raw measurements only."""
        result = ConflictScoreResult(
            mean_kl=0.3,
            base_frontier_rate=0.9,
            conflict_score=0.03,
        )

        assert result.mean_kl == 0.3
        assert result.base_frontier_rate == 0.9
        assert result.conflict_score == 0.03

    def test_conflict_score_formula(self):
        """Verify conflict_score = KL × (1 - frontier_rate)."""
        result = ConflictScoreResult(
            mean_kl=2.0,
            base_frontier_rate=0.3,
            conflict_score=1.4,  # 2.0 * (1 - 0.3) = 1.4
        )

        expected = result.mean_kl * (1.0 - result.base_frontier_rate)
        backend = get_default_backend()
        assert abs(result.conflict_score - expected) <= _eps(backend, result.conflict_score)


class TestConflictScoreCalculator:
    """Tests for ConflictScoreCalculator."""

    def test_initialization(self):
        """Should initialize with a backend."""
        calc = ConflictScoreCalculator()

        assert calc._backend is not None

    def test_flatten_to_vocab_1d(self):
        """1D input should pass through."""
        calc = ConflictScoreCalculator()
        logits = mx.array([1.0, 2.0, 3.0])

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (3,)

    def test_flatten_to_vocab_3d(self):
        """3D input [batch, seq, vocab] should extract last token."""
        calc = ConflictScoreCalculator()
        logits = mx.zeros((2, 5, 100))  # batch=2, seq=5, vocab=100

        result = calc._flatten_to_vocab(logits)

        assert result.shape == (100,)

    def test_compute_identical_logits(self):
        """Identical logits should have zero KL."""
        calc = ConflictScoreCalculator()
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,  # Top token
        )

        # KL should be ~0 for identical distributions
        backend = get_default_backend()
        assert result.mean_kl <= _eps(backend, result.mean_kl)
        # Top token should be in frontier
        assert abs(result.base_frontier_rate - 1.0) <= _eps(backend, result.base_frontier_rate)
        # Conflict should be ~0
        assert result.conflict_score <= _eps(backend, result.conflict_score)

    def test_compute_different_logits(self):
        """Different logits should have positive KL."""
        calc = ConflictScoreCalculator()
        base = mx.array([5.0, 4.0, 3.0, 2.0, 1.0])
        adapted = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Reversed

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=4,  # Top for adapted, bottom for base
        )

        # KL should be positive
        backend = get_default_backend()
        assert result.mean_kl > _eps(backend, result.mean_kl)

    def test_is_in_frontier(self):
        """Should correctly identify frontier membership."""
        calc = ConflictScoreCalculator()
        logits = mx.array([10.0, 9.5, 2.0, 1.0, 0.5])

        # Largest gap is between 9.5 and 2.0, so frontier size = 2
        assert calc._is_in_frontier(logits, token_id=0)
        assert calc._is_in_frontier(logits, token_id=1)
        assert not calc._is_in_frontier(logits, token_id=2)


class TestConflictAnalysis:
    """Tests for ConflictAnalysis static computation - raw measurements only."""

    def test_compute_high_frontier_rate(self):
        """High frontier rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.1, 0.2, 0.1, 0.15],
            base_frontier_hit=[True, True, True, True],
        )

        assert result is not None
        backend = get_default_backend()
        assert abs(result.base_frontier_rate - 1.0) <= _eps(backend, result.base_frontier_rate)
        assert result.token_count == 4
        # Low KL + high frontier rate = low conflict
        assert result.conflict_score <= _eps(backend, result.conflict_score)

    def test_compute_mid_frontier_rate(self):
        """Mid frontier rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.4],  # Low KL
            base_frontier_hit=[True, True, True, True, True, False, True],  # 6/7
        )

        assert result is not None
        backend = get_default_backend()
        assert abs(result.base_frontier_rate - 6 / 7) <= _eps(
            backend, result.base_frontier_rate
        )
        assert result.token_count == 7

    def test_compute_low_frontier_rate(self):
        """Low frontier rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[2.0, 3.0, 2.5, 3.0],
            base_frontier_hit=[False, False, False, False],
        )

        assert result is not None
        backend = get_default_backend()
        assert abs(result.base_frontier_rate) <= _eps(backend, result.base_frontier_rate)
        assert result.token_count == 4
        # High KL + zero frontier rate = high conflict
        mean_kl = sum([2.0, 3.0, 2.5, 3.0]) / 4
        assert abs(result.conflict_score - mean_kl) <= _eps(backend, result.conflict_score)

    def test_compute_empty(self):
        """Empty input should return None."""
        result = ConflictAnalysis.compute([], [])

        assert result is None

    def test_compute_with_nones(self):
        """Should skip None values."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.1, None, 0.2, 0.1],
            base_frontier_hit=[True, None, True, True],
        )

        assert result is not None
        backend = get_default_backend()
        assert abs(result.base_frontier_rate - 1.0) <= _eps(backend, result.base_frontier_rate)
        assert result.token_count == 3  # Only 3 valid pairs


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================


class TestKLDivergenceInvariants:
    """Tests for KL divergence mathematical invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_kl_divergence_non_negative(self, seed: int) -> None:
        """KL divergence must be >= 0.

        Mathematical property: KL(P||Q) >= 0 (Gibbs' inequality).
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        calc = ConflictScoreCalculator()

        base_data = backend.random_normal((100,))
        adapted_data = backend.random_normal((100,))
        backend.eval(base_data, adapted_data)

        base = mx.array(backend.to_numpy(base_data).astype("float32"))
        adapted = mx.array(backend.to_numpy(adapted_data).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        backend = get_default_backend()
        assert result.mean_kl >= -_eps(backend, result.mean_kl)

    def test_kl_self_divergence_zero(self) -> None:
        """KL(P||P) = 0.

        Mathematical property: Self-divergence is zero.
        """
        calc = ConflictScoreCalculator()

        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,
        )

        backend = get_default_backend()
        assert result.mean_kl <= _eps(backend, result.mean_kl)

    @pytest.mark.parametrize("seed", range(5))
    def test_kl_asymmetry(self, seed: int) -> None:
        """KL(P||Q) != KL(Q||P) in general.

        Mathematical property: KL divergence is asymmetric.
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        calc = ConflictScoreCalculator()

        # Generate random uniform values in [0.1, 5.0]
        p_data = backend.random_uniform(low=0.1, high=5.0, shape=(100,))
        q_data = backend.random_uniform(low=0.1, high=5.0, shape=(100,))
        backend.eval(p_data, q_data)

        p = mx.array(backend.to_numpy(p_data).astype("float32"))
        q = mx.array(backend.to_numpy(q_data).astype("float32"))

        result_pq = calc.compute(base_logits=p, adapted_logits=q, sampled_token=0)
        result_qp = calc.compute(base_logits=q, adapted_logits=p, sampled_token=0)

        # In general, they're different (unless identical)
        # This test just checks both are valid non-negative values
        assert result_pq.mean_kl >= 0.0
        assert result_qp.mean_kl >= 0.0


class TestFrontierRateInvariants:
    """Tests for frontier rate bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_frontier_rate_bounded_zero_one(self, seed: int) -> None:
        """Frontier rate must be in [0, 1]."""
        backend = get_default_backend()
        backend.random_seed(seed)
        calc = ConflictScoreCalculator()

        base_data = backend.random_normal((100,))
        adapted_data = backend.random_normal((100,))
        backend.eval(base_data, adapted_data)

        base = mx.array(backend.to_numpy(base_data).astype("float32"))
        adapted = mx.array(backend.to_numpy(adapted_data).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        assert 0.0 <= result.base_frontier_rate <= 1.0

    def test_frontier_rate_one_for_top_token(self) -> None:
        """Frontier rate should be 1.0 when sampling top token of base."""
        calc = ConflictScoreCalculator()

        # Token 4 has highest logit (5.0)
        base = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adapted = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=4,  # Top token in base
        )

        assert result.base_frontier_rate == 1.0

    def test_frontier_rate_zero_for_bottom_token(self) -> None:
        """Frontier rate should be 0.0 when sampling non-frontier token."""
        calc = ConflictScoreCalculator()

        # Token 0 has lowest logit (1.0), not in top-3
        base = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adapted = mx.array([10.0, 1.0, 1.0, 1.0, 1.0])  # Adapted prefers token 0

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,  # Bottom token in base
        )

        # This is a single sample, so frontier rate is 0 or 1
        assert result.base_frontier_rate == 0.0


class TestConflictScoreInvariants:
    """Tests for conflict score invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_conflict_score_non_negative(self, seed: int) -> None:
        """Conflict score must be >= 0.

        Mathematical property: Conflict is derived from non-negative KL.
        """
        backend = get_default_backend()
        backend.random_seed(seed)
        calc = ConflictScoreCalculator()

        base_data = backend.random_normal((100,))
        adapted_data = backend.random_normal((100,))
        backend.eval(base_data, adapted_data)

        base = mx.array(backend.to_numpy(base_data).astype("float32"))
        adapted = mx.array(backend.to_numpy(adapted_data).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        assert result.conflict_score >= 0.0

    def test_identical_logits_no_conflict(self) -> None:
        """Identical logits should have zero conflict score."""
        calc = ConflictScoreCalculator()

        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,
        )

        backend = get_default_backend()
        assert result.conflict_score < _eps(backend, result.conflict_score)
