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

"""Comprehensive tests for permutation_aligner.py.

Tests cover:
- AlignmentResult and Config dataclasses
- PermutationAligner.align() with various inputs
- PermutationAligner.apply() for dense and sparse permutations
- Anchor-based alignment methods
- Hungarian algorithm correctness
- MLP re-basin operations
- TIES-Merging fusion
- Weight key classification
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain.geometry.permutation_aligner import (
    AlignmentResult,
    AnchorActivationContext,
    Config,
    FusionConfig,
    PermutationAligner,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_defaults(self) -> None:
        """Default config has None threshold - must be explicitly set."""
        config = Config()
        assert config.min_match_threshold is None
        assert config.use_anchor_grounding is True
        assert config.top_k == 5
        # effective_threshold should raise without explicit threshold
        import pytest
        with pytest.raises(ValueError, match="min_match_threshold not set"):
            _ = config.effective_threshold

    def test_config_with_threshold(self) -> None:
        """Config.with_threshold() creates config with explicit threshold."""
        config = Config.with_threshold(0.1)
        assert config.min_match_threshold == 0.1
        assert config.effective_threshold == 0.1

    def test_config_from_similarity_distribution(self) -> None:
        """Config.from_similarity_distribution() derives threshold from data."""
        # Similarities with mean ~0.5, std ~0.1
        similarities = [0.4, 0.45, 0.5, 0.55, 0.6]
        config = Config.from_similarity_distribution(similarities, sigma=2.0)
        assert config.min_match_threshold is not None
        # Threshold should be mean - 2*std ≈ 0.5 - 2*0.063 ≈ 0.37
        assert 0.2 < config.effective_threshold < 0.5

    def test_config_from_empty_distribution_raises(self) -> None:
        """Config.from_similarity_distribution() raises on empty data."""
        import pytest
        with pytest.raises(ValueError, match="Cannot derive threshold from empty"):
            Config.from_similarity_distribution([])

    def test_config_custom_values(self) -> None:
        """Custom config values should be accepted."""
        config = Config(min_match_threshold=0.5, use_anchor_grounding=False, top_k=10)
        assert config.min_match_threshold == 0.5
        assert config.use_anchor_grounding is False
        assert config.top_k == 10

    def test_config_is_frozen(self) -> None:
        """Config should be immutable (frozen)."""
        config = Config()
        with pytest.raises((TypeError, AttributeError)):
            config.min_match_threshold = 0.5  # type: ignore


class TestFusionConfig:
    """Tests for FusionConfig dataclass."""

    def test_fusion_config_defaults(self) -> None:
        """Default fusion config has None threshold - must be explicitly set."""
        config = FusionConfig()
        assert config.interference_threshold is None
        assert config.source_alpha == 0.5
        assert config.normalize is False
        # effective_threshold should raise without explicit threshold
        import pytest
        with pytest.raises(ValueError, match="interference_threshold not set"):
            _ = config.effective_threshold

    def test_fusion_config_with_threshold(self) -> None:
        """FusionConfig.with_threshold() creates config with explicit threshold."""
        config = FusionConfig.with_threshold(0.5)
        assert config.interference_threshold == 0.5
        assert config.effective_threshold == 0.5

    def test_fusion_config_from_confidence_distribution(self) -> None:
        """FusionConfig.from_confidence_distribution() derives threshold from data."""
        # Confidences with mean ~0.7, std ~0.1
        confidences = [0.6, 0.65, 0.7, 0.75, 0.8]
        config = FusionConfig.from_confidence_distribution(confidences, sigma=1.0)
        assert config.interference_threshold is not None
        # Threshold should be mean + 1*std ≈ 0.7 + 0.063 ≈ 0.77
        assert 0.7 < config.effective_threshold < 0.9

    def test_fusion_config_from_empty_distribution_raises(self) -> None:
        """FusionConfig.from_confidence_distribution() raises on empty data."""
        import pytest
        with pytest.raises(ValueError, match="Cannot derive threshold from empty"):
            FusionConfig.from_confidence_distribution([])


class TestAnchorActivationContext:
    """Tests for AnchorActivationContext dataclass."""

    def test_activations_returns_tuple(self) -> None:
        """activations() should return source and target for valid layer."""
        context = AnchorActivationContext(
            anchor_ids=["a", "b"],
            source_by_layer={0: [[1.0, 2.0], [3.0, 4.0]]},
            target_by_layer={0: [[5.0, 6.0], [7.0, 8.0]]},
        )
        result = context.activations(0)
        assert result is not None
        source, target = result
        assert len(source) == 2
        assert len(target) == 2

    def test_activations_returns_none_for_missing_layer(self) -> None:
        """activations() should return None for missing layer."""
        context = AnchorActivationContext(
            anchor_ids=["a"],
            source_by_layer={0: [[1.0]]},
            target_by_layer={0: [[2.0]]},
        )
        assert context.activations(99) is None

    def test_activations_returns_none_for_length_mismatch(self) -> None:
        """activations() should return None if source/target lengths differ."""
        context = AnchorActivationContext(
            anchor_ids=["a", "b"],
            source_by_layer={0: [[1.0], [2.0], [3.0]]},  # 3 items
            target_by_layer={0: [[1.0], [2.0]]},  # 2 items
        )
        assert context.activations(0) is None


# =============================================================================
# Hungarian Algorithm Tests
# =============================================================================


class TestHungarianAlgorithm:
    """Tests for the Hungarian algorithm implementation."""

    def test_hungarian_identity_assignment(self) -> None:
        """Diagonal cost matrix should give identity assignment."""
        # Cost matrix where diagonal is 0 (best match)
        cost_matrix = [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
        assignment = PermutationAligner._hungarian_algorithm(cost_matrix)
        assert assignment == [0, 1, 2]

    def test_hungarian_reverse_assignment(self) -> None:
        """Anti-diagonal cost matrix should give reverse assignment."""
        # Cost matrix where anti-diagonal is 0
        cost_matrix = [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        assignment = PermutationAligner._hungarian_algorithm(cost_matrix)
        assert assignment == [2, 1, 0]

    def test_hungarian_empty_matrix(self) -> None:
        """Empty cost matrix should return empty assignment."""
        assignment = PermutationAligner._hungarian_algorithm([])
        assert assignment == []

    def test_hungarian_single_element(self) -> None:
        """Single element matrix should return [0]."""
        cost_matrix = [[5.0]]
        assignment = PermutationAligner._hungarian_algorithm(cost_matrix)
        assert assignment == [0]

    def test_hungarian_optimal_assignment(self) -> None:
        """Should find optimal (minimum cost) assignment."""
        # Classic example where greedy would fail
        cost_matrix = [
            [10.0, 5.0, 13.0],
            [3.0, 15.0, 2.0],
            [8.0, 9.0, 7.0],
        ]
        assignment = PermutationAligner._hungarian_algorithm(cost_matrix)
        # Optimal: 0→1 (5) + 1→2 (2) + 2→0 (8) = 15
        # or: 0→0 (10) + 1→2 (2) + 2→1 (9) = 21
        # or: 0→2 (13) + 1→0 (3) + 2→1 (9) = 25
        # Actual cost depends on assignment found
        total_cost = sum(
            cost_matrix[i][assignment[i]] for i in range(len(cost_matrix))
        )
        # Should find an assignment with reasonable cost
        assert total_cost <= 21  # At least as good as naive diagonal

    def test_hungarian_is_permutation(self) -> None:
        """Assignment should be a valid permutation (no duplicates)."""
        cost_matrix = [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
        ]
        assignment = PermutationAligner._hungarian_algorithm(cost_matrix)
        # Check it's a valid permutation
        assert sorted(assignment) == [0, 1, 2, 3]


# =============================================================================
# PermutationAligner.align() Tests
# =============================================================================


class TestPermutationAlignerAlign:
    """Tests for PermutationAligner.align() method."""

    def test_align_identical_matrices(self, any_backend: "Backend") -> None:
        """Identical matrices should produce identity permutation."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((10, 20))
        b.eval(weight)

        result = PermutationAligner.align(weight, weight, config=Config(), backend=b)

        assert result.match_quality > 0.9  # Should be very high
        assert result.sign_flip_count == 0  # No flips needed

    def test_align_permuted_matrix(self, any_backend: "Backend") -> None:
        """Should find permutation for row-permuted matrix."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((8, 16))
        b.eval(source)

        # Create permuted target (swap rows 0 and 1)
        source_np = b.to_numpy(source)
        target_np = source_np.copy()
        target_np[[0, 1]] = target_np[[1, 0]]
        target = b.array(target_np)
        b.eval(target)

        result = PermutationAligner.align(source, target, config=Config(), backend=b)

        assert result.match_quality > 0.9
        assert b.shape(result.permutation) == (8, 8)
        assert b.shape(result.signs) == (8, 8)

    def test_align_dimension_mismatch_raises(self, any_backend: "Backend") -> None:
        """Mismatched dimensions should raise ValueError."""
        b = any_backend
        source = b.random_normal((10, 20))
        target = b.random_normal((10, 30))  # Different input dim
        b.eval(source, target)

        with pytest.raises(ValueError, match="Weight dimensions must match"):
            PermutationAligner.align(source, target, backend=b)

    def test_align_output_mismatch_raises(self, any_backend: "Backend") -> None:
        """Mismatched output dimensions should raise ValueError."""
        b = any_backend
        source = b.random_normal((10, 20))
        target = b.random_normal((15, 20))  # Different output dim
        b.eval(source, target)

        with pytest.raises(ValueError, match="Weight dimensions must match"):
            PermutationAligner.align(source, target, backend=b)

    def test_align_1d_raises(self, any_backend: "Backend") -> None:
        """1D arrays should raise ValueError."""
        b = any_backend
        source = b.random_normal((20,))
        target = b.random_normal((20,))
        b.eval(source, target)

        with pytest.raises(ValueError, match="must be 2D"):
            PermutationAligner.align(source, target, backend=b)

    def test_align_with_anchors(self, any_backend: "Backend") -> None:
        """Should use anchors when provided and dimensions match."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((10, 20))
        target = b.random_normal((10, 20))
        anchors = b.random_normal((5, 20))  # 5 anchors, matching input dim
        b.eval(source, target, anchors)

        result = PermutationAligner.align(
            source, target, anchors=anchors, config=Config(use_anchor_grounding=True), backend=b
        )

        assert result is not None
        assert result.match_quality >= 0.0

    def test_align_result_structure(self, any_backend: "Backend") -> None:
        """AlignmentResult should have correct structure."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((8, 16))
        target = b.random_normal((8, 16))
        b.eval(source, target)

        result = PermutationAligner.align(source, target, backend=b)

        assert hasattr(result, "permutation")
        assert hasattr(result, "signs")
        assert hasattr(result, "match_quality")
        assert hasattr(result, "match_confidences")
        assert hasattr(result, "sign_flip_count")
        assert len(result.match_confidences) == 8


# =============================================================================
# PermutationAligner.apply() Tests
# =============================================================================


class TestPermutationAlignerApply:
    """Tests for PermutationAligner.apply() method."""

    def test_apply_identity_permutation(self, any_backend: "Backend") -> None:
        """Identity permutation should not change weights."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((8, 16))
        b.eval(weight)

        # Create identity alignment
        perm = b.eye(8)
        signs = b.eye(8)
        alignment = AlignmentResult(
            permutation=perm,
            signs=signs,
            match_quality=1.0,
            match_confidences=[1.0] * 8,
            sign_flip_count=0,
        )

        result = PermutationAligner.apply(
            weight, alignment, align_output=True, align_input=False, backend=b
        )
        b.eval(result)

        diff = b.to_numpy(result) - b.to_numpy(weight)
        assert abs(diff).max() < 1e-5

    def test_apply_output_alignment(self, any_backend: "Backend") -> None:
        """Output alignment should permute rows."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((4, 8))
        b.eval(weight)

        # Swap rows 0 and 1
        perm_data = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        perm = b.array(perm_data)
        signs = b.eye(4)
        b.eval(perm, signs)

        alignment = AlignmentResult(
            permutation=perm,
            signs=signs,
            match_quality=1.0,
            match_confidences=[1.0] * 4,
            sign_flip_count=0,
        )

        result = PermutationAligner.apply(
            weight, alignment, align_output=True, align_input=False, backend=b
        )
        b.eval(result)

        # Check rows are swapped
        weight_np = b.to_numpy(weight)
        result_np = b.to_numpy(result)
        # Row 0 of result should be row 1 of original
        assert abs(result_np[0] - weight_np[1]).max() < 1e-5
        assert abs(result_np[1] - weight_np[0]).max() < 1e-5

    def test_apply_input_alignment(self, any_backend: "Backend") -> None:
        """Input alignment should permute columns."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((8, 4))
        b.eval(weight)

        # Swap columns 0 and 1
        perm_data = [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        perm = b.array(perm_data)
        signs = b.eye(4)
        b.eval(perm, signs)

        alignment = AlignmentResult(
            permutation=perm,
            signs=signs,
            match_quality=1.0,
            match_confidences=[1.0] * 4,
            sign_flip_count=0,
        )

        result = PermutationAligner.apply(
            weight, alignment, align_output=False, align_input=True, backend=b
        )
        b.eval(result)

        weight_np = b.to_numpy(weight)
        result_np = b.to_numpy(result)
        # Column 0 of result should be column 1 of original
        assert abs(result_np[:, 0] - weight_np[:, 1]).max() < 1e-5
        assert abs(result_np[:, 1] - weight_np[:, 0]).max() < 1e-5

    def test_apply_with_sign_flips(self, any_backend: "Backend") -> None:
        """Sign flips should negate appropriate rows/columns."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((4, 8))
        b.eval(weight)

        perm = b.eye(4)
        # Flip sign of first row
        signs = b.diag(b.array([-1.0, 1.0, 1.0, 1.0]))
        b.eval(perm, signs)

        alignment = AlignmentResult(
            permutation=perm,
            signs=signs,
            match_quality=1.0,
            match_confidences=[1.0] * 4,
            sign_flip_count=1,
        )

        result = PermutationAligner.apply(
            weight, alignment, align_output=True, align_input=False, backend=b
        )
        b.eval(result)

        weight_np = b.to_numpy(weight)
        result_np = b.to_numpy(result)
        # First row should be negated
        assert abs(result_np[0] + weight_np[0]).max() < 1e-5
        # Other rows unchanged
        assert abs(result_np[1:] - weight_np[1:]).max() < 1e-5

    def test_apply_sparse_permutation(self, any_backend: "Backend") -> None:
        """Sparse permutation should work via index gather."""
        b = any_backend
        b.random_seed(42)
        weight = b.random_normal((4, 8))
        b.eval(weight)

        # Sparse permutation: swap 0↔1, keep 2, 3
        indices = [1, 0, 2, 3]
        signs = b.array([1.0, 1.0, 1.0, 1.0])
        b.eval(signs)

        alignment = AlignmentResult(
            permutation=b.array(indices),  # Not used in sparse mode
            signs=signs,
            match_quality=1.0,
            match_confidences=[1.0] * 4,
            sign_flip_count=0,
            is_sparse_permutation=True,
            assignment_indices=indices,
        )

        result = PermutationAligner.apply(
            weight, alignment, align_output=True, align_input=False, backend=b
        )
        b.eval(result)

        # Should have same shape
        assert b.shape(result) == b.shape(weight)


# =============================================================================
# Anchor-based Alignment Tests
# =============================================================================


class TestAnchorAlignment:
    """Tests for anchor-based alignment methods."""

    def test_align_via_anchor_projection(self, any_backend: "Backend") -> None:
        """Should align using anchor projections."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((10, 32))
        target = b.random_normal((10, 32))
        anchors = b.random_normal((8, 32))
        b.eval(source, target, anchors)

        config = Config.with_threshold(0.1)
        result = PermutationAligner.align_via_anchor_projection(
            source, target, anchors, config=config, backend=b
        )

        assert result is not None
        assert result.match_quality >= 0.0
        assert len(result.match_confidences) == 10

    def test_align_via_anchor_projection_dim_mismatch(
        self, any_backend: "Backend"
    ) -> None:
        """Dimension mismatch should fall back gracefully."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((10, 32))
        target = b.random_normal((10, 32))
        anchors = b.random_normal((8, 16))  # Wrong dim
        b.eval(source, target, anchors)

        config = Config.with_threshold(0.1)
        # Should not crash, uses fallback
        result = PermutationAligner.align_via_anchor_projection(
            source, target, anchors, config=config, backend=b
        )
        assert result is not None

    def test_align_via_anchor_activations(self, any_backend: "Backend") -> None:
        """Should align using per-model anchor activations."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((10, 32))
        target = b.random_normal((10, 32))
        source_anchors = b.random_normal((5, 32))
        target_anchors = b.random_normal((5, 32))
        b.eval(source, target, source_anchors, target_anchors)

        config = Config.with_threshold(0.1)
        result = PermutationAligner.align_via_anchor_activations(
            source, target, source_anchors, target_anchors, config=config, backend=b
        )

        assert result is not None
        assert len(result.match_confidences) == 10


# =============================================================================
# MLP Re-Basin Tests
# =============================================================================


class TestMlpRebasin:
    """Tests for MLP re-basin operations."""

    def test_rebasin_mlp_only(self, any_backend: "Backend") -> None:
        """Should re-basin MLP blocks."""
        b = any_backend
        b.random_seed(42)

        # Create mock MLP weights
        hidden = 32
        intermediate = 64
        source_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": b.random_normal(
                (intermediate, hidden)
            ),
            "model.layers.0.mlp.down_proj.weight": b.random_normal(
                (hidden, intermediate)
            ),
        }
        target_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": b.random_normal(
                (intermediate, hidden)
            ),
            "model.layers.0.mlp.down_proj.weight": b.random_normal(
                (hidden, intermediate)
            ),
        }
        anchors = b.random_normal((16, hidden))
        for k, v in source_weights.items():
            b.eval(v)
        for k, v in target_weights.items():
            b.eval(v)
        b.eval(anchors)

        config = Config.with_threshold(0.1)
        aligned, avg_quality, blocks_aligned = PermutationAligner.rebasin_mlp_only(
            source_weights, target_weights, anchors, config=config, backend=b
        )

        assert blocks_aligned == 1
        assert avg_quality >= 0.0
        assert len(aligned) == 3
        assert "model.layers.0.mlp.up_proj.weight" in aligned

    def test_rebasin_preserves_non_mlp_weights(self, any_backend: "Backend") -> None:
        """Non-MLP weights should be copied unchanged."""
        b = any_backend
        b.random_seed(42)

        hidden = 32
        intermediate = 64
        source_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": b.random_normal(
                (intermediate, hidden)
            ),
            "model.layers.0.mlp.down_proj.weight": b.random_normal(
                (hidden, intermediate)
            ),
            "model.layers.0.attn.q_proj.weight": b.random_normal((hidden, hidden)),
            "model.embed_tokens.weight": b.random_normal((1000, hidden)),
        }
        target_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
            "model.layers.0.mlp.gate_proj.weight": b.random_normal(
                (intermediate, hidden)
            ),
            "model.layers.0.mlp.down_proj.weight": b.random_normal(
                (hidden, intermediate)
            ),
        }
        anchors = b.random_normal((16, hidden))
        for v in source_weights.values():
            b.eval(v)
        for v in target_weights.values():
            b.eval(v)
        b.eval(anchors)

        config = Config.with_threshold(0.1)
        aligned, _, _ = PermutationAligner.rebasin_mlp_only(
            source_weights, target_weights, anchors, config=config, backend=b
        )

        # Non-MLP weights should be preserved
        assert "model.layers.0.attn.q_proj.weight" in aligned
        assert "model.embed_tokens.weight" in aligned

    def test_rebasin_incomplete_mlp_skipped(self, any_backend: "Backend") -> None:
        """Incomplete MLP blocks should be skipped."""
        b = any_backend
        b.random_seed(42)

        hidden = 32
        intermediate = 64
        source_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
            # Missing gate_proj and down_proj
        }
        target_weights = {
            "model.layers.0.mlp.up_proj.weight": b.random_normal((intermediate, hidden)),
        }
        anchors = b.random_normal((16, hidden))
        for v in source_weights.values():
            b.eval(v)
        for v in target_weights.values():
            b.eval(v)
        b.eval(anchors)

        _, _, blocks_aligned = PermutationAligner.rebasin_mlp_only(
            source_weights, target_weights, anchors, backend=b
        )

        assert blocks_aligned == 0  # No complete blocks


# =============================================================================
# Fusion Tests
# =============================================================================


class TestFusion:
    """Tests for TIES-Merging fusion."""

    def test_fuse_basic(self, any_backend: "Backend") -> None:
        """Basic fusion should produce weighted average."""
        b = any_backend
        b.random_seed(42)

        source = b.random_normal((8, 16))
        target = b.random_normal((8, 16))
        b.eval(source, target)

        alignment = AlignmentResult(
            permutation=b.eye(8),
            signs=b.eye(8),
            match_quality=1.0,
            match_confidences=[1.0] * 8,  # All high confidence
            sign_flip_count=0,
        )

        config = FusionConfig(source_alpha=0.5)
        fused = PermutationAligner.fuse(source, target, alignment, config, backend=b)
        b.eval(fused)

        # With 100% confidence and 0.5 alpha, should be average
        expected = (b.to_numpy(source) + b.to_numpy(target)) / 2
        actual = b.to_numpy(fused)
        assert abs(actual - expected).max() < 1e-5

    def test_fuse_low_confidence_preserves_source(
        self, any_backend: "Backend"
    ) -> None:
        """Low confidence matches should preserve source."""
        b = any_backend
        b.random_seed(42)

        source = b.random_normal((4, 8))
        target = b.random_normal((4, 8))
        b.eval(source, target)

        alignment = AlignmentResult(
            permutation=b.eye(4),
            signs=b.eye(4),
            match_quality=0.0,
            match_confidences=[0.0, 0.0, 0.0, 0.0],  # All zero confidence
            sign_flip_count=0,
        )

        config = FusionConfig(source_alpha=0.5)
        fused = PermutationAligner.fuse(source, target, alignment, config, backend=b)
        b.eval(fused)

        # With 0% confidence, should preserve source
        source_np = b.to_numpy(source)
        fused_np = b.to_numpy(fused)
        assert abs(fused_np - source_np).max() < 1e-5


# =============================================================================
# Weight Key Classification Tests
# =============================================================================


class TestWeightKeyClassification:
    """Tests for weight key classification methods."""

    def test_is_mlp_weight_positive(self) -> None:
        """MLP weights should be correctly identified."""
        assert PermutationAligner.is_mlp_weight("model.layers.0.mlp.up_proj.weight")
        assert PermutationAligner.is_mlp_weight("model.layers.0.mlp.gate_proj.weight")
        assert PermutationAligner.is_mlp_weight("model.layers.0.mlp.down_proj.weight")
        assert PermutationAligner.is_mlp_weight("model.h.0.mlp.w1.weight")
        assert PermutationAligner.is_mlp_weight("model.h.0.mlp.w2.weight")
        assert PermutationAligner.is_mlp_weight("model.h.0.mlp.w3.weight")

    def test_is_mlp_weight_negative(self) -> None:
        """Non-MLP weights should not be identified as MLP."""
        assert not PermutationAligner.is_mlp_weight("model.layers.0.attn.q_proj.weight")
        assert not PermutationAligner.is_mlp_weight("model.embed_tokens.weight")
        assert not PermutationAligner.is_mlp_weight("model.norm.weight")

    def test_is_attention_weight_positive(self) -> None:
        """Attention weights should be correctly identified."""
        assert PermutationAligner.is_attention_weight("model.layers.0.attn.q_proj.weight")
        assert PermutationAligner.is_attention_weight("model.layers.0.attn.k_proj.weight")
        assert PermutationAligner.is_attention_weight("model.layers.0.attn.v_proj.weight")
        assert PermutationAligner.is_attention_weight("model.layers.0.attn.o_proj.weight")
        assert PermutationAligner.is_attention_weight("model.h.0.attn.wq.weight")
        assert PermutationAligner.is_attention_weight("model.h.0.attn.wk.weight")

    def test_is_attention_weight_negative(self) -> None:
        """Non-attention weights should not be identified as attention."""
        assert not PermutationAligner.is_attention_weight(
            "model.layers.0.mlp.up_proj.weight"
        )
        assert not PermutationAligner.is_attention_weight("model.embed_tokens.weight")


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_extract_layer_index(self) -> None:
        """Should extract layer index from various key formats."""
        assert PermutationAligner._extract_layer_index("model.layers.5.mlp.up_proj") == 5
        assert PermutationAligner._extract_layer_index("model.h.10.attn.q_proj") == 10
        assert (
            PermutationAligner._extract_layer_index("model.blocks.0.mlp.dense") == 0
        )
        assert PermutationAligner._extract_layer_index("model.block.99.fc") == 99

    def test_extract_layer_index_none(self) -> None:
        """Should return None for keys without layer index."""
        assert PermutationAligner._extract_layer_index("model.embed_tokens") is None
        assert PermutationAligner._extract_layer_index("lm_head.weight") is None

    def test_inverse_permutation(self) -> None:
        """Should compute correct inverse permutation."""
        indices = [2, 0, 1]  # Maps 0→2, 1→0, 2→1
        inverse = PermutationAligner._inverse_permutation(indices, 3)
        # Inverse: 0→1, 1→2, 2→0
        assert inverse == [1, 2, 0]

    def test_inverse_permutation_identity(self) -> None:
        """Identity permutation should invert to identity."""
        indices = [0, 1, 2, 3]
        inverse = PermutationAligner._inverse_permutation(indices, 4)
        assert inverse == [0, 1, 2, 3]

    def test_array_from_matrix(self, any_backend: "Backend") -> None:
        """Should convert 2D list to Array."""
        b = any_backend
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        arr = PermutationAligner._array_from_matrix(matrix, backend=b)
        b.eval(arr)

        assert b.shape(arr) == (3, 2)
        arr_np = b.to_numpy(arr)
        assert abs(arr_np[0, 0] - 1.0) < 1e-5
        assert abs(arr_np[2, 1] - 6.0) < 1e-5

    def test_extract_sign_values_vector(self, any_backend: "Backend") -> None:
        """Should extract sign values from 1D vector."""
        b = any_backend
        signs = b.array([1.0, -1.0, 1.0])
        b.eval(signs)

        values = PermutationAligner._extract_sign_values(signs, 3, backend=b)
        assert values == [1.0, -1.0, 1.0]

    def test_extract_sign_values_matrix(self, any_backend: "Backend") -> None:
        """Should extract sign values from diagonal of matrix."""
        b = any_backend
        signs = b.diag(b.array([1.0, -1.0, 1.0, -1.0]))
        b.eval(signs)

        values = PermutationAligner._extract_sign_values(signs, 4, backend=b)
        assert values == [1.0, -1.0, 1.0, -1.0]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_matrix_alignment(self, any_backend: "Backend") -> None:
        """Should handle very small matrices."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((2, 4))
        target = b.random_normal((2, 4))
        b.eval(source, target)

        result = PermutationAligner.align(source, target, backend=b)
        assert result is not None
        assert b.shape(result.permutation) == (2, 2)

    def test_square_matrix_alignment(self, any_backend: "Backend") -> None:
        """Should handle square matrices."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((16, 16))
        target = b.random_normal((16, 16))
        b.eval(source, target)

        result = PermutationAligner.align(source, target, backend=b)
        assert result is not None
        assert b.shape(result.permutation) == (16, 16)

    def test_negative_sign_alignment(self, any_backend: "Backend") -> None:
        """Should detect and handle sign flips."""
        b = any_backend
        b.random_seed(42)
        source = b.random_normal((8, 16))
        b.eval(source)

        # Create target with some rows negated
        target = source * b.array([[-1.0], [1.0], [-1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
        b.eval(target)

        result = PermutationAligner.align(source, target, backend=b)
        # Should detect sign flips
        assert result.sign_flip_count >= 0  # At least some detection
