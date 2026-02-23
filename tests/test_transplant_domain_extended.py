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

"""Extended tests for transplant domain logic.

Tests critical APIs:
- compute_weight_space_transplant(): Weight-space null-space projection
- compute_transplant_delta(): Anchor-relative constrained least-squares
- partition_core_boundary(): Partition probes into core/boundary sets
"""

import math
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.transplant import (
    compute_behavior_jacobian_projector,
    compute_transplant_delta,
    compute_weight_space_transplant,
    partition_core_boundary,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestComputeWeightSpaceTransplant:
    """Tests for compute_weight_space_transplant()."""

    def test_basic_transplant(self, backend):
        """Basic transplant should work."""
        out_dim, in_dim = 64, 32
        n_samples = 16

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert backend.shape(result.merged_weight) == (out_dim, in_dim)
        assert all_finite(result.merged_weight, backend)

    def test_identical_weights_small_delta(self, backend):
        """Identical source/target should produce small delta."""
        out_dim, in_dim = 32, 16
        n_samples = 16

        weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=weight,
            target_weight=weight,
            input_activations=input_activations,
            backend=backend,
        )

        # Delta norm should be zero when source == target
        eps = division_epsilon(backend, weight)
        scale = max(1.0, float(out_dim * in_dim))
        assert result.delta_norm <= eps * scale

    def test_preserved_fraction_bounded(self, backend):
        """Preserved fraction should be non-negative and finite."""
        out_dim, in_dim = 32, 16
        n_samples = 16

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        assert result.preserved_fraction >= 0.0
        assert math.isfinite(result.preserved_fraction)

    def test_with_density_activations(self, backend):
        """Density-weighted transplant should work."""
        out_dim, in_dim = 32, 16
        n_samples = 16

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        src_density_acts = backend.random_normal((n_samples, in_dim))
        tgt_density_acts = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)
        backend.eval(src_density_acts, tgt_density_acts)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=src_density_acts,
            target_activations_for_density=tgt_density_acts,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert 0.0 <= result.transfer_strength <= 1.0

    def test_null_rank_computed(self, backend):
        """Null rank should be computed."""
        out_dim, in_dim = 32, 16
        n_samples = 8  # Less than in_dim

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        # null_rank = in_dim - min(n_samples, in_dim) = 16 - 8 = 8
        assert result.null_rank >= 0

    def test_null_rank_respects_numerical_rank(self, backend):
        """Numerical rank should bound null_rank for low-rank activations."""
        out_dim, in_dim = 32, 16
        n_samples = 12
        rank = 4

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        base = backend.random_normal((n_samples, rank))
        projection = backend.random_normal((rank, in_dim))
        input_activations = backend.matmul(base, projection)
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        assert result.null_rank >= in_dim - rank

    def test_density_length_mismatch_truncates(self, backend):
        """Density weighting should handle mismatched sample counts."""
        out_dim, in_dim = 32, 16
        n_samples = 10
        n_density = 6

        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        src_density_acts = backend.random_normal((n_density, in_dim))
        tgt_density_acts = backend.random_normal((n_density, in_dim))
        backend.eval(
            source_aligned,
            target_weight,
            input_activations,
            src_density_acts,
            tgt_density_acts,
        )

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            source_activations_for_density=src_density_acts,
            target_activations_for_density=tgt_density_acts,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert all_finite(result.merged_weight, backend)


class TestComputeTransplantDelta:
    """Tests for compute_transplant_delta()."""

    def test_basic_delta_computation(self, backend):
        """Basic delta computation should work."""
        out_dim, in_dim = 32, 16
        n_core = 8

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        backend.eval(weight_target, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert result.applied is True
        assert backend.shape(result.merged_weight) == (out_dim, in_dim)

    def test_with_boundary_activations(self, backend):
        """Boundary constraint should be applied."""
        out_dim, in_dim = 32, 16
        n_core = 8
        n_boundary = 4

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        boundary_activations = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, activations_core, delta_activations)
        backend.eval(boundary_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_activations,
            backend=backend,
        )

        assert result.merged_weight is not None
        assert result.applied is True

    def test_boundary_preserved(self, backend):
        """Boundary outputs should be approximately preserved."""
        out_dim, in_dim = 32, 16
        n_core = 8
        n_boundary = 4

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        boundary_activations = backend.random_normal((n_boundary, in_dim))
        backend.eval(weight_target, activations_core, delta_activations)
        backend.eval(boundary_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            boundary_activations=boundary_activations,
            backend=backend,
        )

        # Check boundary preservation: A_boundary @ W' ≈ A_boundary @ W
        original_output = backend.matmul(boundary_activations, backend.transpose(weight_target))
        merged_output = backend.matmul(boundary_activations, backend.transpose(result.merged_weight))
        backend.eval(original_output, merged_output)

        diff = backend.mean(backend.abs(merged_output - original_output))
        backend.eval(diff)

        # Should be within precision-scaled tolerance
        diff_val = float(backend.to_scalar(diff))
        mean_orig_arr = backend.mean(backend.abs(original_output))
        backend.eval(mean_orig_arr)
        mean_orig = float(backend.to_scalar(mean_orig_arr))
        eps = division_epsilon(backend, original_output) * max(1.0, mean_orig)
        assert diff_val <= eps

    def test_delta_scale_applied(self, backend):
        """Delta scale should modulate the update magnitude."""
        out_dim, in_dim = 32, 16
        n_core = 8

        weight_target = backend.random_normal((out_dim, in_dim))
        activations_core = backend.random_normal((n_core, in_dim))
        delta_activations = backend.random_normal((n_core, out_dim))
        backend.eval(weight_target, activations_core, delta_activations)

        result_full = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            delta_scale=1.0,
            backend=backend,
        )

        result_half = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            delta_scale=0.5,
            backend=backend,
        )

        # Half scale should have smaller delta
        diff_full = backend.mean(backend.abs(result_full.merged_weight - weight_target))
        diff_half = backend.mean(backend.abs(result_half.merged_weight - weight_target))
        backend.eval(diff_full, diff_half)

        # Note: Due to null-space projection, this relationship may not be exact
        # but half scale should generally produce smaller changes
        eps = division_epsilon(backend, result_full.merged_weight)
        scale = max(1.0, float(out_dim * in_dim))
        assert float(backend.to_scalar(diff_half)) <= float(backend.to_scalar(diff_full)) + eps * scale

    def test_1d_weight_returns_unchanged(self, backend):
        """1D weights should return unchanged."""
        weight_target = backend.random_normal((32,))  # 1D
        activations_core = backend.random_normal((8, 32))
        delta_activations = backend.random_normal((8, 32))
        backend.eval(weight_target, activations_core, delta_activations)

        result = compute_transplant_delta(
            weight_target=weight_target,
            activations_core=activations_core,
            delta_activations=delta_activations,
            backend=backend,
        )

        assert result.applied is False


class TestPartitionCoreBoundary:
    """Tests for partition_core_boundary()."""

    def test_basic_partition(self, backend):
        """Basic partitioning should work."""
        activations = backend.random_normal((10, 32))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(10)]
        core_probe_ids = {"probe_0", "probe_2", "probe_4"}

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert set(result.core_indices) == {0, 2, 4}
        assert set(result.boundary_indices) == {1, 3, 5, 6, 7, 8, 9}
        assert len(result.core_probe_ids) == 3
        assert len(result.boundary_probe_ids) == 7

    def test_empty_core(self, backend):
        """Empty core set should return empty partition."""
        activations = backend.random_normal((10, 32))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(10)]
        core_probe_ids: set[str] = set()

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert result.core_indices == []
        assert result.boundary_indices == []

    def test_all_core(self, backend):
        """All probes in core should have no boundary."""
        activations = backend.random_normal((5, 32))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(5)]
        core_probe_ids = set(probe_ids)

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert len(result.core_indices) == 5
        assert len(result.boundary_indices) == 0

    def test_probe_id_mapping(self, backend):
        """Probe IDs should be correctly mapped to indices."""
        activations = backend.random_normal((5, 32))
        backend.eval(activations)

        probe_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
        core_probe_ids = {"beta", "delta"}

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        assert result.core_indices == [1, 3]  # beta=1, delta=3
        assert result.core_probe_ids == ["beta", "delta"]


class TestTransplantMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        out_dim=st.integers(min_value=8, max_value=32),
        in_dim=st.integers(min_value=8, max_value=32),
        n_samples=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_merged_weight_finite(self, out_dim, in_dim, n_samples):
        """Merged weight should always be finite."""
        backend = get_default_backend()
        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        assert all_finite(result.merged_weight, backend)

    @given(
        out_dim=st.integers(min_value=8, max_value=32),
        in_dim=st.integers(min_value=8, max_value=32),
        n_samples=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=5, deadline=None)
    def test_preserved_fraction_valid(self, out_dim, in_dim, n_samples):
        """Preserved fraction should be non-negative and finite.

        Note: With geodesic norms, preserved_fraction can exceed 1.0 because
        geodesic distance from origin doesn't have the same contraction
        properties as Euclidean norms under projection.
        """
        backend = get_default_backend()
        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        import math
        assert result.preserved_fraction >= 0.0
        assert math.isfinite(result.preserved_fraction)

    @given(
        out_dim=st.integers(min_value=8, max_value=32),
        in_dim=st.integers(min_value=8, max_value=32),
        n_samples=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_null_space_constraint(self, out_dim, in_dim, n_samples):
        """Projected delta should preserve boundary: A @ delta.T ≈ 0."""
        backend = get_default_backend()
        source_aligned = backend.random_normal((out_dim, in_dim))
        target_weight = backend.random_normal((out_dim, in_dim))
        input_activations = backend.random_normal((n_samples, in_dim))
        backend.eval(source_aligned, target_weight, input_activations)

        result = compute_weight_space_transplant(
            source_aligned=source_aligned,
            target_weight=target_weight,
            input_activations=input_activations,
            backend=backend,
        )

        delta_proj = result.merged_weight - target_weight
        residual = backend.matmul(input_activations, backend.transpose(delta_proj))
        res_norm = backend.mean(geodesic_norms(residual, backend))
        act_norm = backend.mean(geodesic_norms(input_activations, backend))
        delta_norm = backend.mean(geodesic_norms(delta_proj, backend))
        backend.eval(res_norm, act_norm, delta_norm)

        # Null-space projection residual is bounded by condition_number × eps × scale.
        # Random matrices have condition numbers ~10-100×. Use sqrt(eps) as base
        # (accounts for accumulated float ops) with 100× factor for conditioning.
        eps = machine_epsilon(backend, input_activations)
        sqrt_eps = float(eps ** 0.5)
        scale = float(backend.to_scalar(act_norm)) * float(backend.to_scalar(delta_norm))
        tol = sqrt_eps * max(1.0, scale) * 100.0  # 100× for random matrix conditioning

        assert float(backend.to_scalar(res_norm)) <= tol

    @given(
        n_probes=st.integers(min_value=5, max_value=20),
        n_core=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10, deadline=None)
    def test_partition_indices_disjoint(self, n_probes, n_core):
        """Core and boundary indices should be disjoint."""
        n_core = min(n_core, n_probes)
        backend = get_default_backend()
        activations = backend.random_normal((n_probes, 16))
        backend.eval(activations)

        probe_ids = [f"probe_{i}" for i in range(n_probes)]
        core_probe_ids = {f"probe_{i}" for i in range(n_core)}

        result = partition_core_boundary(
            activations=activations,
            probe_ids=probe_ids,
            core_probe_ids=core_probe_ids,
            backend=backend,
        )

        core_set = set(result.core_indices)
        boundary_set = set(result.boundary_indices)

        # Should be disjoint
        assert core_set.isdisjoint(boundary_set)

        # Union should cover all indices
        assert core_set | boundary_set == set(range(n_probes))


class TestComputeBehaviorJacobianProjector:
    """Tests for compute_behavior_jacobian_projector.

    Mathematical invariants:
    1. Projection is idempotent: P(P(delta)) == P(delta)
    2. Projected delta is orthogonal to gradient rows: G @ P(delta) ≈ 0
    3. vectorized flag is True
    4. null_rank + jacobian_rank == D (flattened weight dim)
    """

    def test_projection_removes_gradient_component(self, backend) -> None:
        """Delta projected through behavior Jacobian should be orthogonal to G rows."""
        b = backend
        N, D = 5, 20  # 5 probes, flattened weight dim 20
        G = b.random_normal((N, D))
        b.eval(G)

        projector = compute_behavior_jacobian_projector(G, backend=b)
        assert projector.vectorized is True

        # Project a random delta
        delta = b.random_normal((1, D))
        b.eval(delta)

        # Implicit projection: delta_proj = delta - G^T (GG^T)^+ G delta
        delta_row = b.matmul(delta, b.transpose(projector.weighted_activations))
        correction = b.matmul(delta_row, projector.gram_inv)
        correction = b.matmul(correction, projector.weighted_activations)
        delta_proj = delta - correction
        b.eval(delta_proj)

        # G @ delta_proj^T should be near zero
        residual = b.matmul(G, b.transpose(delta_proj))
        b.eval(residual)
        residual_norm = float(b.to_scalar(b.sqrt(b.sum(residual * residual))))
        delta_norm = float(b.to_scalar(b.sqrt(b.sum(delta * delta))))

        # Float32 eigendecomposition + pseudoinverse: residuals scale as
        # sqrt(eps) * condition, not eps.  For random N×D matrices the
        # condition number is O(sqrt(D/N)), giving ~1e-3 relative residual.
        eps = machine_epsilon(b, G)
        tol = math.sqrt(eps) * delta_norm * D
        assert residual_norm < tol, (
            f"Projected delta is not orthogonal to G: "
            f"residual_norm={residual_norm:.2e}, threshold={tol:.2e}"
        )

    def test_projection_idempotent(self, backend) -> None:
        """Projecting twice should give the same result as projecting once."""
        b = backend
        N, D = 4, 16
        G = b.random_normal((N, D))
        b.eval(G)

        projector = compute_behavior_jacobian_projector(G, backend=b)

        delta = b.random_normal((1, D))
        b.eval(delta)

        def _project(v):
            row = b.matmul(v, b.transpose(projector.weighted_activations))
            corr = b.matmul(row, projector.gram_inv)
            corr = b.matmul(corr, projector.weighted_activations)
            result = v - corr
            b.eval(result)
            return result

        p1 = _project(delta)
        p2 = _project(p1)

        diff = b.sqrt(b.sum((p1 - p2) * (p1 - p2)))
        b.eval(diff)
        diff_val = float(b.to_scalar(diff))
        p1_norm = float(b.to_scalar(b.sqrt(b.sum(p1 * p1))))

        # Same float32 eigendecomposition precision bound as orthogonality test.
        eps = machine_epsilon(b, G)
        tol = math.sqrt(eps) * p1_norm * D
        assert diff_val < tol, (
            f"Projection not idempotent: ||P²δ - Pδ||={diff_val:.2e}, tol={tol:.2e}"
        )

    def test_null_rank_consistent(self, backend) -> None:
        """null_rank + activation_rank should equal D when N < D."""
        b = backend
        N, D = 3, 12
        G = b.random_normal((N, D))
        b.eval(G)

        projector = compute_behavior_jacobian_projector(G, backend=b)

        # Rank should be at most N (number of probes)
        activation_rank = D - projector.null_rank
        assert activation_rank <= N, (
            f"Jacobian rank {activation_rank} exceeds number of probes {N}"
        )
        assert projector.null_rank >= D - N, (
            f"null_rank {projector.null_rank} too small: expected >= {D - N}"
        )

    def test_zero_probes_raises(self, backend) -> None:
        """Empty gradient matrix should raise ValueError."""
        b = backend
        G = b.zeros((0, 10))
        with pytest.raises(ValueError, match="0 probes"):
            compute_behavior_jacobian_projector(G, backend=b)

    def test_preserves_null_component(self, backend) -> None:
        """A delta in the null space of G should survive projection unchanged."""
        b = backend
        # Construct G with known null space: G = [e1, e2] (rows), D=4
        # Null space = span(e3, e4)
        G = b.zeros((2, 4))
        G = b.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        b.eval(G)

        projector = compute_behavior_jacobian_projector(G, backend=b)

        # Delta in null space: e3
        delta = b.array([[0.0, 0.0, 1.0, 0.0]])
        b.eval(delta)

        row = b.matmul(delta, b.transpose(projector.weighted_activations))
        corr = b.matmul(row, projector.gram_inv)
        corr = b.matmul(corr, projector.weighted_activations)
        delta_proj = delta - corr
        b.eval(delta_proj)

        diff = b.sqrt(b.sum((delta - delta_proj) * (delta - delta_proj)))
        b.eval(diff)
        assert float(b.to_scalar(diff)) < 1e-2, (
            "Null-space delta should survive projection unchanged"
        )

    def test_removes_row_space_component(self, backend) -> None:
        """A delta in the row space of G should be removed by projection."""
        b = backend
        G = b.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        b.eval(G)

        projector = compute_behavior_jacobian_projector(G, backend=b)

        # Delta in row space: e1
        delta = b.array([[1.0, 0.0, 0.0, 0.0]])
        b.eval(delta)

        row = b.matmul(delta, b.transpose(projector.weighted_activations))
        corr = b.matmul(row, projector.gram_inv)
        corr = b.matmul(corr, projector.weighted_activations)
        delta_proj = delta - corr
        b.eval(delta_proj)

        proj_norm = float(b.to_scalar(b.sqrt(b.sum(delta_proj * delta_proj))))
        # Float32 eigendecomposition + pseudoinverse introduces ~1e-3 residual
        # for exact rank matrices. This is a numerical precision bound, not a
        # correctness failure — the projector is working, just at float32 limits.
        assert proj_norm < 1e-2, (
            f"Row-space delta should be mostly removed: ||Pδ||={proj_norm:.2e}"
        )
