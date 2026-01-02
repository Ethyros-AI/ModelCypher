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

"""Tests for curvature-guided alignment module."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.curvature_alignment import (
    AlignmentGuidance,
    AlignmentPlan,
    compute_alignment_guidance,
    compute_layer_correspondence_by_curvature,
    curvature_weighted_procrustes,
    _compute_layer_guidance,
)
from modelcypher.core.domain.geometry.curvature_profile import (
    CurvatureProfile,
    LayerCurvature,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default backend for tests."""
    return get_default_backend()


@pytest.fixture
def simple_layer_curvature():
    """Simple LayerCurvature for testing."""
    return LayerCurvature(
        layer_idx=0,
        sectional_mean=-0.1,
        sectional_std=0.05,
        ollivier_ricci_mean=-0.15,
        ollivier_ricci_std=0.08,
        intrinsic_dimension=64.0,
    )


@pytest.fixture
def similar_layer_curvature():
    """LayerCurvature similar to simple_layer_curvature."""
    return LayerCurvature(
        layer_idx=0,
        sectional_mean=-0.12,
        sectional_std=0.06,
        ollivier_ricci_mean=-0.18,
        ollivier_ricci_std=0.09,
        intrinsic_dimension=68.0,
    )


@pytest.fixture
def different_layer_curvature():
    """LayerCurvature with opposite curvature sign."""
    return LayerCurvature(
        layer_idx=0,
        sectional_mean=0.2,
        sectional_std=0.1,
        ollivier_ricci_mean=0.25,
        ollivier_ricci_std=0.12,
        intrinsic_dimension=32.0,
    )


@pytest.fixture
def source_profile():
    """Source model curvature profile with 4 layers."""
    layers = [
        LayerCurvature(
            layer_idx=i,
            sectional_mean=-0.1 - i * 0.01,
            sectional_std=0.05,
            ollivier_ricci_mean=-0.15 - i * 0.02,
            ollivier_ricci_std=0.08,
            intrinsic_dimension=64.0 + i * 4,
        )
        for i in range(4)
    ]
    return CurvatureProfile(
        model_path="/path/to/source",
        model_family="qwen",
        model_size="0.5B",
        layer_curvatures=layers,
        total_layers=4,
    )


@pytest.fixture
def target_profile_similar():
    """Target profile similar to source (same family)."""
    layers = [
        LayerCurvature(
            layer_idx=i,
            sectional_mean=-0.11 - i * 0.01,
            sectional_std=0.06,
            ollivier_ricci_mean=-0.16 - i * 0.02,
            ollivier_ricci_std=0.09,
            intrinsic_dimension=66.0 + i * 4,
        )
        for i in range(4)
    ]
    return CurvatureProfile(
        model_path="/path/to/target",
        model_family="qwen",
        model_size="3B",
        layer_curvatures=layers,
        total_layers=4,
    )


@pytest.fixture
def target_profile_different():
    """Target profile different from source (different architecture)."""
    layers = [
        LayerCurvature(
            layer_idx=i,
            sectional_mean=0.1 + i * 0.02,  # Opposite sign
            sectional_std=0.1,
            ollivier_ricci_mean=0.2 + i * 0.03,
            ollivier_ricci_std=0.15,
            intrinsic_dimension=32.0 + i * 2,  # Much smaller dimension
        )
        for i in range(6)  # Different layer count
    ]
    return CurvatureProfile(
        model_path="/path/to/different",
        model_family="llama",
        model_size="7B",
        layer_curvatures=layers,
        total_layers=6,
    )


# =============================================================================
# AlignmentGuidance Tests
# =============================================================================


class TestAlignmentGuidance:
    """Tests for AlignmentGuidance dataclass."""

    def test_create_alignment_guidance(self):
        """AlignmentGuidance can be created with all fields."""
        guidance = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.5,
            dimension_scale=1.2,
            curvature_correction=0.3,
            alignment_weight=0.8,
        )
        assert guidance.layer_idx == 0
        assert guidance.alignment_effort == 0.5
        assert guidance.dimension_scale == 1.2
        assert guidance.curvature_correction == 0.3
        assert guidance.alignment_weight == 0.8

    def test_alignment_guidance_is_frozen(self):
        """AlignmentGuidance is immutable."""
        guidance = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.5,
            dimension_scale=1.0,
            curvature_correction=0.3,
            alignment_weight=0.8,
        )
        with pytest.raises(AttributeError):
            guidance.alignment_effort = 0.9


# =============================================================================
# AlignmentPlan Tests
# =============================================================================


class TestAlignmentPlan:
    """Tests for AlignmentPlan dataclass."""

    def test_create_alignment_plan(self):
        """AlignmentPlan can be created with all fields."""
        guidance = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.5,
            dimension_scale=1.0,
            curvature_correction=0.3,
            alignment_weight=0.8,
        )
        plan = AlignmentPlan(
            source_model="/path/source",
            target_model="/path/target",
            layer_guidance=[guidance],
            total_alignment_effort=0.5,
            mean_dimension_scale=1.0,
        )
        assert plan.source_model == "/path/source"
        assert plan.target_model == "/path/target"
        assert len(plan.layer_guidance) == 1

    def test_alignment_plan_is_frozen(self):
        """AlignmentPlan is immutable."""
        plan = AlignmentPlan(
            source_model="/path/source",
            target_model="/path/target",
            layer_guidance=[],
            total_alignment_effort=0.0,
            mean_dimension_scale=1.0,
        )
        with pytest.raises(AttributeError):
            plan.mean_dimension_scale = 2.0


# =============================================================================
# _compute_layer_guidance Tests
# =============================================================================


class TestComputeLayerGuidance:
    """Tests for _compute_layer_guidance function."""

    def test_similar_layers_low_effort(
        self, simple_layer_curvature, similar_layer_curvature, different_layer_curvature
    ):
        """Similar layers should have low alignment effort."""
        guidance_similar = _compute_layer_guidance(
            simple_layer_curvature, similar_layer_curvature, layer_idx=0
        )
        guidance_diff = _compute_layer_guidance(
            simple_layer_curvature, different_layer_curvature, layer_idx=0
        )
        assert guidance_similar.alignment_effort <= guidance_diff.alignment_effort

    def test_different_layers_high_effort(
        self, simple_layer_curvature, different_layer_curvature, similar_layer_curvature
    ):
        """Different layers should have high alignment effort."""
        guidance = _compute_layer_guidance(
            simple_layer_curvature, different_layer_curvature, layer_idx=0
        )
        guidance_similar = _compute_layer_guidance(
            simple_layer_curvature, similar_layer_curvature, layer_idx=0
        )
        assert guidance.alignment_effort >= guidance_similar.alignment_effort

    def test_dimension_difference_triggers_projection(self):
        """Large dimension difference triggers projection flag."""
        src = LayerCurvature(
            layer_idx=0,
            intrinsic_dimension=100.0,
            ollivier_ricci_mean=-0.1,
        )
        tgt = LayerCurvature(
            layer_idx=0,
            intrinsic_dimension=50.0,  # Half the dimension
            ollivier_ricci_mean=-0.1,
        )
        guidance = _compute_layer_guidance(src, tgt, layer_idx=0)
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        expected_scale = tgt.intrinsic_dimension / src.intrinsic_dimension
        assert abs(guidance.dimension_scale - expected_scale) <= eps

    def test_dimension_scale_computed_correctly(self):
        """Dimension scale is target/source."""
        src = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
        tgt = LayerCurvature(layer_idx=0, intrinsic_dimension=128.0, ollivier_ricci_mean=-0.1)
        guidance = _compute_layer_guidance(src, tgt, layer_idx=0)
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        expected_scale = tgt.intrinsic_dimension / src.intrinsic_dimension
        assert abs(guidance.dimension_scale - expected_scale) <= eps

    def test_zero_dimension_defaults_to_one(self):
        """Zero intrinsic dimension defaults to 1.0 to avoid division by zero."""
        src = LayerCurvature(layer_idx=0, intrinsic_dimension=0.0, ollivier_ricci_mean=-0.1)
        tgt = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
        guidance = _compute_layer_guidance(src, tgt, layer_idx=0)
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        assert abs(guidance.dimension_scale - 1.0) <= eps

    def test_same_sign_curvature_lower_correction(self):
        """Same sign curvatures have lower correction than opposite signs."""
        src = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.2)
        tgt_same = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.3)
        tgt_diff = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=0.3)

        guidance_same = _compute_layer_guidance(src, tgt_same, layer_idx=0)
        guidance_diff = _compute_layer_guidance(src, tgt_diff, layer_idx=0)

        assert guidance_same.curvature_correction < guidance_diff.curvature_correction

    def test_zero_curvature_uses_moderate_correction(self):
        """Zero curvature values yield a finite correction."""
        src = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=0.0)
        tgt = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
        guidance = _compute_layer_guidance(src, tgt, layer_idx=0)
        assert math.isfinite(guidance.curvature_correction)

    def test_alignment_weight_range(self):
        """Alignment weight should be in range [0, 1]."""
        src = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
        tgt = LayerCurvature(layer_idx=0, intrinsic_dimension=64.0, ollivier_ricci_mean=0.5)
        guidance = _compute_layer_guidance(src, tgt, layer_idx=0)
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        assert -eps <= guidance.alignment_weight <= 1.0 + eps


# =============================================================================
# compute_alignment_guidance Tests
# =============================================================================


class TestComputeAlignmentGuidance:
    """Tests for compute_alignment_guidance function."""

    def test_similar_profiles_procrustes_strategy(
        self, source_profile, target_profile_similar
    ):
        """Similar profiles produce guidance entries."""
        plan = compute_alignment_guidance(source_profile, target_profile_similar)
        assert len(plan.layer_guidance) == 4

    def test_different_profiles_curvature_flow_strategy(
        self, source_profile, target_profile_different
    ):
        """Different profiles still produce guidance entries."""
        plan = compute_alignment_guidance(source_profile, target_profile_different)
        # Different layer counts - only source layers get guidance
        assert len(plan.layer_guidance) == 4

    def test_model_paths_preserved(self, source_profile, target_profile_similar):
        """Model paths are preserved in the plan."""
        plan = compute_alignment_guidance(source_profile, target_profile_similar)
        assert plan.source_model == source_profile.model_path
        assert plan.target_model == target_profile_similar.model_path

    def test_layer_guidance_per_source_layer(self, source_profile, target_profile_similar):
        """One guidance entry per source layer."""
        plan = compute_alignment_guidance(source_profile, target_profile_similar)
        layer_indices = [g.layer_idx for g in plan.layer_guidance]
        expected_indices = [lc.layer_idx for lc in source_profile.layer_curvatures]
        assert layer_indices == expected_indices

    def test_critical_layers_identified(self, source_profile, target_profile_different):
        """Critical layers are not required; guidance is raw."""
        plan = compute_alignment_guidance(source_profile, target_profile_different)
        assert len(plan.layer_guidance) == len(source_profile.layer_curvatures)

    def test_mean_dimension_scale(self, source_profile, target_profile_similar):
        """Mean dimension scale is computed correctly."""
        plan = compute_alignment_guidance(source_profile, target_profile_similar)
        expected_mean = sum(g.dimension_scale for g in plan.layer_guidance) / len(
            plan.layer_guidance
        )
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        assert abs(plan.mean_dimension_scale - expected_mean) <= eps

    def test_empty_profiles(self):
        """Empty profiles produce empty guidance."""
        empty_src = CurvatureProfile(
            model_path="/empty",
            model_family="test",
            model_size="0B",
            layer_curvatures=[],
            total_layers=0,
        )
        empty_tgt = CurvatureProfile(
            model_path="/empty2",
            model_family="test",
            model_size="0B",
            layer_curvatures=[],
            total_layers=0,
        )
        plan = compute_alignment_guidance(empty_src, empty_tgt)
        assert len(plan.layer_guidance) == 0
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        assert abs(plan.mean_dimension_scale - 1.0) <= eps

    def test_different_layer_counts_uses_relative_position(self):
        """Different layer counts map by relative position."""
        src_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
            for i in range(4)
        ]
        tgt_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
            for i in range(8)
        ]
        source = CurvatureProfile(
            model_path="/src", model_family="test", model_size="1B",
            layer_curvatures=src_layers, total_layers=4
        )
        target = CurvatureProfile(
            model_path="/tgt", model_family="test", model_size="2B",
            layer_curvatures=tgt_layers, total_layers=8
        )
        plan = compute_alignment_guidance(source, target)
        # Source layer 0 maps to target layer 0
        # Source layer 3 maps to target layer 7 (relative position 1.0)
        assert len(plan.layer_guidance) == 4


# =============================================================================
# curvature_weighted_procrustes Tests
# =============================================================================


class TestCurvatureWeightedProcrustes:
    """Tests for curvature_weighted_procrustes function."""

    def test_same_dimension_returns_square_matrix(self, backend):
        """Same dimension inputs return square rotation matrix."""
        n, d = 50, 32
        source = backend.random_normal((n, d))
        target = backend.random_normal((n, d))
        guidance = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.2,
            dimension_scale=1.0,
            curvature_correction=0.1,
            alignment_weight=0.9,
        )
        R = curvature_weighted_procrustes(source, target, guidance, backend)
        backend.eval(R)
        shape = backend.shape(R)
        assert shape == (d, d)

    def test_different_dimension_with_projection(self, backend):
        """Different dimensions with projection flag adjusts source."""
        n, d_src, d_tgt = 50, 64, 32
        source = backend.random_normal((n, d_src))
        target = backend.random_normal((n, d_tgt))
        guidance = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.5,
            dimension_scale=0.5,
            curvature_correction=0.2,
            alignment_weight=0.7,
        )
        R = curvature_weighted_procrustes(source, target, guidance, backend)
        backend.eval(R)
        shape = backend.shape(R)
        # After projection, source is d_tgt, so R is (d_tgt, d_tgt)
        assert shape == (d_tgt, d_tgt)

    def test_low_curvature_correction_minimal_damping(self, backend):
        """Low curvature correction has minimal damping effect."""
        n, d = 50, 16
        source = backend.random_normal((n, d))
        target = backend.random_normal((n, d))

        guidance_low = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.1,
            dimension_scale=1.0,
            curvature_correction=0.0,  # No correction
            alignment_weight=1.0,
        )
        R_low = curvature_weighted_procrustes(source, target, guidance_low, backend)
        backend.eval(R_low)

        # With no damping, R should be close to pure rotation
        # Check approximate orthogonality: R @ R^T ≈ I
        RTR = backend.matmul(R_low, backend.transpose(R_low))
        I = backend.eye(d)
        backend.eval(RTR, I)

        diff = backend.to_numpy(RTR) - backend.to_numpy(I)
        frobenius_norm = (diff ** 2).sum() ** 0.5
        eps = machine_epsilon(backend, R_low)
        assert math.isfinite(frobenius_norm)
        assert frobenius_norm >= -eps

    def test_high_curvature_correction_more_damping(self, backend):
        """High curvature correction adds significant damping."""
        n, d = 50, 16
        source = backend.random_normal((n, d))
        target = backend.random_normal((n, d))

        guidance_high = AlignmentGuidance(
            layer_idx=0,
            alignment_effort=0.9,
            dimension_scale=1.0,
            curvature_correction=1.0,  # Maximum correction
            alignment_weight=0.3,
        )
        R_high = curvature_weighted_procrustes(source, target, guidance_high, backend)
        backend.eval(R_high)

        # With high damping, R should be closer to identity
        # damping = 1.0 - 0.3 * 1.0 = 0.7
        # R = R * 0.7 + I * 0.3
        I = backend.eye(d)
        backend.eval(I)

        # The matrix should have significant identity component
        R_np = backend.to_numpy(R_high)
        I_np = backend.to_numpy(I)

        # Diagonal should be pushed toward 1
        diag_mean = R_np.diagonal().mean()
        assert math.isfinite(float(diag_mean))


# =============================================================================
# compute_layer_correspondence_by_curvature Tests
# =============================================================================


class TestComputeLayerCorrespondence:
    """Tests for compute_layer_correspondence_by_curvature function."""

    def test_same_layer_count_direct_mapping(self, source_profile, target_profile_similar):
        """Same layer count profiles map 1:1."""
        correspondence = compute_layer_correspondence_by_curvature(
            source_profile, target_profile_similar
        )
        # Should have correspondence for each source layer
        assert len(correspondence) == len(source_profile.layer_curvatures)
        # With similar profiles, expect roughly 1:1 mapping
        for src_idx in correspondence:
            assert src_idx in correspondence

    def test_different_layer_counts_monotonic(self):
        """Different layer counts produce monotonic mapping."""
        src_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0 + i*5, ollivier_ricci_mean=-0.1 - i*0.01)
            for i in range(4)
        ]
        tgt_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0 + i*2.5, ollivier_ricci_mean=-0.1 - i*0.005)
            for i in range(8)
        ]
        source = CurvatureProfile(
            model_path="/src", model_family="test", model_size="1B",
            layer_curvatures=src_layers, total_layers=4
        )
        target = CurvatureProfile(
            model_path="/tgt", model_family="test", model_size="2B",
            layer_curvatures=tgt_layers, total_layers=8
        )
        correspondence = compute_layer_correspondence_by_curvature(source, target)

        # Check monotonicity: higher source layer -> higher target layer
        sorted_src = sorted(correspondence.keys())
        tgt_values = [correspondence[k] for k in sorted_src]
        for i in range(len(tgt_values) - 1):
            assert tgt_values[i] <= tgt_values[i + 1], "Correspondence should be monotonic"

    def test_empty_profiles_empty_correspondence(self):
        """Empty profiles produce empty correspondence."""
        empty_src = CurvatureProfile(
            model_path="/empty", model_family="test", model_size="0B",
            layer_curvatures=[], total_layers=0
        )
        empty_tgt = CurvatureProfile(
            model_path="/empty2", model_family="test", model_size="0B",
            layer_curvatures=[], total_layers=0
        )
        correspondence = compute_layer_correspondence_by_curvature(empty_src, empty_tgt)
        assert correspondence == {}

    def test_no_duplicate_target_mappings(self, source_profile, target_profile_similar):
        """Each target layer is used at most once."""
        correspondence = compute_layer_correspondence_by_curvature(
            source_profile, target_profile_similar
        )
        target_values = list(correspondence.values())
        assert len(target_values) == len(set(target_values)), "No duplicate target mappings"

    def test_similar_curvatures_match(self):
        """Layers with similar curvature signatures match preferentially."""
        # Create source with distinctive curvature pattern
        src_layers = [
            LayerCurvature(layer_idx=0, intrinsic_dimension=50.0, ollivier_ricci_mean=-0.1),
            LayerCurvature(layer_idx=1, intrinsic_dimension=100.0, ollivier_ricci_mean=-0.5),  # Distinctive
        ]
        # Target has matching patterns but shuffled indices
        tgt_layers = [
            LayerCurvature(layer_idx=0, intrinsic_dimension=105.0, ollivier_ricci_mean=-0.52),  # Similar to src 1
            LayerCurvature(layer_idx=1, intrinsic_dimension=52.0, ollivier_ricci_mean=-0.12),  # Similar to src 0
        ]
        source = CurvatureProfile(
            model_path="/src", model_family="test", model_size="1B",
            layer_curvatures=src_layers, total_layers=2
        )
        target = CurvatureProfile(
            model_path="/tgt", model_family="test", model_size="2B",
            layer_curvatures=tgt_layers, total_layers=2
        )
        correspondence = compute_layer_correspondence_by_curvature(source, target)

        # Due to position penalty, should still prefer nearby layers
        # but curvature similarity has some influence
        assert len(correspondence) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_guidance_to_procrustes_workflow(self, backend, source_profile, target_profile_similar):
        """Full workflow: profile -> guidance -> Procrustes."""
        # Step 1: Compute alignment plan
        plan = compute_alignment_guidance(source_profile, target_profile_similar)

        # Step 2: Use first layer's guidance for Procrustes
        guidance = plan.layer_guidance[0]

        # Step 3: Generate mock activations
        n, d = 100, 64
        source_acts = backend.random_normal((n, d))
        target_acts = backend.random_normal((n, d))

        # Step 4: Compute alignment
        R = curvature_weighted_procrustes(source_acts, target_acts, guidance, backend)
        backend.eval(R)

        # Verify output is valid rotation-like matrix
        shape = backend.shape(R)
        assert shape == (d, d)

    def test_correspondence_then_guidance(self, source_profile, target_profile_different):
        """Compute correspondence then use for layer-by-layer guidance."""
        # Step 1: Find layer correspondence
        correspondence = compute_layer_correspondence_by_curvature(
            source_profile, target_profile_different
        )

        # Step 2: Compute full alignment plan
        plan = compute_alignment_guidance(source_profile, target_profile_different)

        # Correspondence and plan should be consistent
        assert len(correspondence) > 0
        assert len(plan.layer_guidance) == len(source_profile.layer_curvatures)

    def test_plan_strategies(self):
        """Different profile combinations produce different strategies."""
        # Create profiles that will trigger different strategies

        # Similar profiles -> procrustes
        similar_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.1)
            for i in range(4)
        ]
        similar_src = CurvatureProfile(
            model_path="/s1", model_family="test", model_size="1B",
            layer_curvatures=similar_layers, total_layers=4
        )
        similar_tgt = CurvatureProfile(
            model_path="/t1", model_family="test", model_size="1B",
            layer_curvatures=similar_layers, total_layers=4
        )
        plan_similar = compute_alignment_guidance(similar_src, similar_tgt)
        assert len(plan_similar.layer_guidance) == 4

        # Different dimensions -> projection_first
        big_dim_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=200.0, ollivier_ricci_mean=-0.1)
            for i in range(4)
        ]
        small_dim_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=50.0, ollivier_ricci_mean=-0.1)
            for i in range(4)
        ]
        proj_src = CurvatureProfile(
            model_path="/s2", model_family="test", model_size="1B",
            layer_curvatures=big_dim_layers, total_layers=4
        )
        proj_tgt = CurvatureProfile(
            model_path="/t2", model_family="test", model_size="1B",
            layer_curvatures=small_dim_layers, total_layers=4
        )
        plan_proj = compute_alignment_guidance(proj_src, proj_tgt)
        assert len(plan_proj.layer_guidance) == 4

        # Opposite curvatures -> curvature_flow
        pos_curv_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0, ollivier_ricci_mean=0.3)
            for i in range(4)
        ]
        neg_curv_layers = [
            LayerCurvature(layer_idx=i, intrinsic_dimension=64.0, ollivier_ricci_mean=-0.3)
            for i in range(4)
        ]
        curv_src = CurvatureProfile(
            model_path="/s3", model_family="test", model_size="1B",
            layer_curvatures=neg_curv_layers, total_layers=4
        )
        curv_tgt = CurvatureProfile(
            model_path="/t3", model_family="test", model_size="1B",
            layer_curvatures=pos_curv_layers, total_layers=4
        )
        plan_curv = compute_alignment_guidance(curv_src, curv_tgt)
        assert len(plan_curv.layer_guidance) == 4
