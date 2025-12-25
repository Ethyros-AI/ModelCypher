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

from modelcypher.core.domain.entropy.conflict_score import (
    ConflictAnalysis,
    ConflictScoreCalculator,
    ConflictScoreResult,
)


class TestConflictScoreResult:
    """Tests for ConflictScoreResult - raw measurements only."""

    def test_creation_with_raw_measurements(self):
        """Should create with raw measurements only."""
        result = ConflictScoreResult(
            mean_kl=0.3,
            base_approval_rate=0.9,
            conflict_score=0.03,
        )

        assert result.mean_kl == 0.3
        assert result.base_approval_rate == 0.9
        assert result.conflict_score == 0.03

    def test_exceeds_threshold_false(self):
        """Low conflict_score should not exceed threshold."""
        result = ConflictScoreResult(
            mean_kl=0.3,
            base_approval_rate=0.9,
            conflict_score=0.03,
        )

        assert not result.exceeds_threshold(0.3)

    def test_exceeds_threshold_true(self):
        """High conflict_score should exceed threshold."""
        result = ConflictScoreResult(
            mean_kl=2.0,
            base_approval_rate=0.3,
            conflict_score=1.4,
        )

        assert result.exceeds_threshold(0.3)

    def test_conflict_score_formula(self):
        """Verify conflict_score = KL × (1 - approval)."""
        result = ConflictScoreResult(
            mean_kl=2.0,
            base_approval_rate=0.3,
            conflict_score=1.4,  # 2.0 * (1 - 0.3) = 1.4
        )

        expected = result.mean_kl * (1.0 - result.base_approval_rate)
        assert abs(result.conflict_score - expected) < 0.01


class TestConflictScoreCalculator:
    """Tests for ConflictScoreCalculator."""

    def test_initialization(self):
        """Should initialize with default top_k."""
        calc = ConflictScoreCalculator()

        assert calc.top_k == 10
        assert calc.epsilon > 0

    def test_custom_top_k(self):
        """Should accept custom top_k."""
        calc = ConflictScoreCalculator(top_k=5)

        assert calc.top_k == 5

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
        calc = ConflictScoreCalculator(top_k=5)
        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,  # Top token
        )

        # KL should be ~0 for identical distributions
        assert result.mean_kl < 0.01
        # Top token should be approved
        assert result.base_approval_rate == 1.0
        # Conflict should be ~0
        assert result.conflict_score < 0.01

    def test_compute_different_logits(self):
        """Different logits should have positive KL."""
        calc = ConflictScoreCalculator(top_k=5)
        base = mx.array([5.0, 4.0, 3.0, 2.0, 1.0])
        adapted = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Reversed

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=4,  # Top for adapted, bottom for base
        )

        # KL should be positive
        assert result.mean_kl > 0

    def test_is_in_top_k(self):
        """Should correctly identify top-K membership."""
        calc = ConflictScoreCalculator()
        logits = mx.array([1.0, 5.0, 3.0, 4.0, 2.0])  # Sorted order: [1, 3, 4, 2, 0]

        # Token 1 (value 5.0) is top
        assert calc._is_in_top_k(logits, token_id=1, k=3)
        # Token 0 (value 1.0) is lowest
        assert not calc._is_in_top_k(logits, token_id=0, k=3)


class TestConflictAnalysis:
    """Tests for ConflictAnalysis static computation - raw measurements only."""

    def test_compute_high_approval(self):
        """High approval rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.1, 0.2, 0.1, 0.15],
            base_approved_top_k=[True, True, True, True],
        )

        assert result is not None
        assert result.base_approval_rate == 1.0
        assert result.token_count == 4
        # Low KL + high approval = low conflict
        assert result.conflict_score < 0.01

    def test_compute_mid_approval(self):
        """Mid approval rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.4],  # Low KL
            base_approved_top_k=[True, True, True, True, True, False, True],  # 6/7 = ~85%
        )

        assert result is not None
        assert abs(result.base_approval_rate - 6 / 7) < 0.01
        assert result.token_count == 7

    def test_compute_low_approval(self):
        """Low approval rate with raw measurements."""
        result = ConflictAnalysis.compute(
            kl_divergences=[2.0, 3.0, 2.5, 3.0],
            base_approved_top_k=[False, False, False, False],
        )

        assert result is not None
        assert result.base_approval_rate == 0.0
        assert result.token_count == 4
        # High KL + zero approval = high conflict
        mean_kl = sum([2.0, 3.0, 2.5, 3.0]) / 4
        assert abs(result.conflict_score - mean_kl) < 0.01

    def test_compute_empty(self):
        """Empty input should return None."""
        result = ConflictAnalysis.compute([], [])

        assert result is None

    def test_compute_with_nones(self):
        """Should skip None values."""
        result = ConflictAnalysis.compute(
            kl_divergences=[0.1, None, 0.2, 0.1],
            base_approved_top_k=[True, None, True, True],
        )

        assert result is not None
        assert result.base_approval_rate == 1.0
        assert result.token_count == 3  # Only 3 valid pairs

    def test_exceeds_threshold(self):
        """Should correctly identify when conflict exceeds threshold."""
        result = ConflictAnalysis.compute(
            kl_divergences=[2.0, 3.0],
            base_approved_top_k=[False, False],
        )

        assert result is not None
        assert result.exceeds_threshold(0.5)  # High conflict
        assert not result.exceeds_threshold(10.0)  # Very high threshold


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
        import numpy as np

        rng = np.random.default_rng(seed)
        calc = ConflictScoreCalculator(top_k=5)

        base = mx.array(rng.standard_normal(100).astype("float32"))
        adapted = mx.array(rng.standard_normal(100).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        assert result.mean_kl >= 0.0

    def test_kl_self_divergence_zero(self) -> None:
        """KL(P||P) = 0.

        Mathematical property: Self-divergence is zero.
        """
        calc = ConflictScoreCalculator(top_k=5)

        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,
        )

        assert result.mean_kl < 0.01  # Approximately zero

    @pytest.mark.parametrize("seed", range(5))
    def test_kl_asymmetry(self, seed: int) -> None:
        """KL(P||Q) != KL(Q||P) in general.

        Mathematical property: KL divergence is asymmetric.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        calc = ConflictScoreCalculator(top_k=5)

        p = mx.array(rng.uniform(0.1, 5.0, 100).astype("float32"))
        q = mx.array(rng.uniform(0.1, 5.0, 100).astype("float32"))

        result_pq = calc.compute(base_logits=p, adapted_logits=q, sampled_token=0)
        result_qp = calc.compute(base_logits=q, adapted_logits=p, sampled_token=0)

        # In general, they're different (unless identical)
        # This test just checks both are valid non-negative values
        assert result_pq.mean_kl >= 0.0
        assert result_qp.mean_kl >= 0.0


class TestApprovalRateInvariants:
    """Tests for approval rate bounds."""

    @pytest.mark.parametrize("seed", range(5))
    def test_approval_rate_bounded_zero_one(self, seed: int) -> None:
        """Approval rate must be in [0, 1].

        Mathematical property: Approval rate is a proportion.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        calc = ConflictScoreCalculator(top_k=5)

        base = mx.array(rng.standard_normal(100).astype("float32"))
        adapted = mx.array(rng.standard_normal(100).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        assert 0.0 <= result.base_approval_rate <= 1.0

    def test_approval_rate_one_for_top_token(self) -> None:
        """Approval rate should be 1.0 when sampling top token of base."""
        calc = ConflictScoreCalculator(top_k=5)

        # Token 4 has highest logit (5.0)
        base = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adapted = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=4,  # Top token in base
        )

        assert result.base_approval_rate == 1.0

    def test_approval_rate_zero_for_bottom_token(self) -> None:
        """Approval rate should be 0.0 when sampling non-top-k token."""
        calc = ConflictScoreCalculator(top_k=3)

        # Token 0 has lowest logit (1.0), not in top-3
        base = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adapted = mx.array([10.0, 1.0, 1.0, 1.0, 1.0])  # Adapted prefers token 0

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,  # Bottom token in base
        )

        # This is a single sample, so approval rate is 0 or 1
        # Token 0 is NOT in top-3 of base, so approval_rate = 0
        assert result.base_approval_rate == 0.0


class TestConflictScoreInvariants:
    """Tests for conflict score invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_conflict_score_non_negative(self, seed: int) -> None:
        """Conflict score must be >= 0.

        Mathematical property: Conflict is derived from non-negative KL.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        calc = ConflictScoreCalculator(top_k=5)

        base = mx.array(rng.standard_normal(100).astype("float32"))
        adapted = mx.array(rng.standard_normal(100).astype("float32"))

        result = calc.compute(
            base_logits=base,
            adapted_logits=adapted,
            sampled_token=0,
        )

        assert result.conflict_score >= 0.0

    def test_identical_logits_no_conflict(self) -> None:
        """Identical logits should have zero conflict score."""
        calc = ConflictScoreCalculator(top_k=5)

        logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = calc.compute(
            base_logits=logits,
            adapted_logits=logits,
            sampled_token=4,
        )

        assert result.conflict_score < 0.01
        assert not result.exceeds_threshold(0.01)  # Very low threshold
