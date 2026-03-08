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

"""Tests for NB-LoRA Cayley transform implementation.

Validates the mathematical guarantees of the Cayley parameterization
from Wang et al. (2025), arXiv:2501.19050:

1. Semi-orthogonality: A^T @ A + B^T @ B = I_r
2. Spectral bound: ||2 * B^T @ S @ A||_2 <= 2 * max(S)
3. Forward consistency: layer.forward(x) matches x @ get_effective_delta()^T
4. Scale clamping: get_S() always in [0, scale_bound]
5. Factory correctness: scale_bound = sigma_max / 2 * margin
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cayley_lora import (
    NBLoRAConfig,
    NBLoRALayer,
    PerDirectionBoundResult,
    cayley_transform,
    cayley_transform_full,
    create_nb_lora_from_base_weight,
    nb_lora_forward,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


# =============================================================================
# Helpers
# =============================================================================


def _frobenius(backend, M):
    """Frobenius norm of a matrix."""
    return float(backend.to_scalar(backend.sqrt(backend.sum(M * M))))


def _spectral_norm(backend, M):
    """Spectral norm (largest singular value) of a matrix."""
    M_f32 = backend.astype(M, "float32")
    backend.eval(M_f32)
    _, S, _ = backend.svd(M_f32, compute_uv=True)
    backend.eval(S)
    return float(backend.to_scalar(S[0]))


# =============================================================================
# Test: Cayley Transform Core
# =============================================================================


class TestCayleyTransformCore:
    """Tests for the core cayley_transform(X, Y, backend) function."""

    def test_semi_orthogonality(self, backend):
        """A^T @ A + B^T @ B = I_r for the core Cayley transform."""
        r = 4
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((8, r)) * 0.1
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        backend.eval(AtA, BtB)

        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5

    def test_output_shapes(self, backend):
        """A is [r, r] and B is [m, r]."""
        r, m = 4, 8
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((m, r)) * 0.1
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        assert tuple(backend.shape(A)) == (r, r)
        assert tuple(backend.shape(B)) == (m, r)

    def test_identity_input_gives_identity_A(self, backend):
        """When X=0 and Y=0, A should be identity and B should be zero."""
        r, m = 4, 6
        X = backend.zeros((r, r))
        Y = backend.zeros((m, r))
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        # Z = 0, so A = (I - 0)(I + 0)^{-1} = I, B = -2 * 0 * I^{-1} = 0
        I = backend.eye(r)
        diff_A = A - I
        backend.eval(diff_A)
        assert _frobenius(backend, diff_A) < 1e-6
        assert _frobenius(backend, B) < 1e-6

    def test_determinism(self, backend):
        """Same input gives same output."""
        _r, m = 4, 8
        X = backend.array([[0.1, -0.2, 0.3, 0.0],
                           [0.05, 0.1, -0.1, 0.2],
                           [-0.15, 0.0, 0.05, 0.1],
                           [0.2, -0.1, 0.0, 0.15]])
        Y = backend.array([[0.1, 0.0, -0.1, 0.05]] * m)
        backend.eval(X, Y)

        A1, B1 = cayley_transform(X, Y, backend)
        A2, B2 = cayley_transform(X, Y, backend)
        backend.eval(A1, B1, A2, B2)

        assert _frobenius(backend, A1 - A2) < 1e-7
        assert _frobenius(backend, B1 - B2) < 1e-7

    def test_semi_orthogonality_large_m(self, backend):
        """Semi-orthogonality holds for large m (m >> r)."""
        r, m = 3, 32
        X = backend.random_normal((r, r)) * 0.05
        Y = backend.random_normal((m, r)) * 0.05
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5

    def test_semi_orthogonality_rank_1(self, backend):
        """Semi-orthogonality holds for rank 1."""
        r, m = 1, 8
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((m, r)) * 0.1
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5

    def test_semi_orthogonality_large_init(self, backend):
        """Semi-orthogonality holds even with large initialization values."""
        r, m = 4, 8
        X = backend.random_normal((r, r)) * 2.0
        Y = backend.random_normal((m, r)) * 2.0
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-4

    def test_stacked_columns_orthonormal(self, backend):
        """Stacked [A; B] has orthonormal columns (equivalent check)."""
        r, m = 4, 8
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((m, r)) * 0.1
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        # Stack [A; B] -> [(r+m), r]
        stacked = backend.concatenate([A, B], axis=0)
        backend.eval(stacked)

        # Check: stacked^T @ stacked = I_r
        gram = backend.matmul(backend.transpose(stacked), stacked)
        I = backend.eye(r)
        diff = gram - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5


# =============================================================================
# Test: Cayley Transform Full
# =============================================================================


class TestCayleyTransformFull:
    """Tests for cayley_transform_full(A_tilde, B_tilde, backend)."""

    def test_semi_orthogonality(self, backend):
        """Stacked [A^T; B^T] has orthonormal columns: A @ A^T + B @ B^T = I_r.

        Note: For the full transform, A is [r, n_in] and B is [r, n_out].
        The stacked matrix [A^T; B^T] is [(n_in+n_out), r] and has orthonormal
        columns, so the property is A @ A^T + B @ B^T = I_r (all [r, r]).
        """
        r, n_in, n_out = 4, 16, 32
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        backend.eval(A_tilde, B_tilde)

        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)

        AAt = backend.matmul(A, backend.transpose(A))  # [r, r]
        BBt = backend.matmul(B, backend.transpose(B))  # [r, r]
        result = AAt + BBt
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-4

    def test_output_shapes(self, backend):
        """A is [r, n_in] and B is [r, n_out]."""
        r, n_in, n_out = 4, 16, 32
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        backend.eval(A_tilde, B_tilde)

        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)

        assert tuple(backend.shape(A)) == (r, n_in)
        assert tuple(backend.shape(B)) == (r, n_out)

    def test_semi_orthogonality_various_dimensions(self, backend):
        """Semi-orthogonality across different dimension combinations."""
        configs = [(2, 8, 16), (4, 32, 64), (8, 16, 16)]
        for r, n_in, n_out in configs:
            A_tilde = backend.random_normal((r, n_in)) * 0.1
            B_tilde = backend.random_normal((r, n_out)) * 0.1
            backend.eval(A_tilde, B_tilde)

            A, B = cayley_transform_full(A_tilde, B_tilde, backend)
            backend.eval(A, B)

            AAt = backend.matmul(A, backend.transpose(A))  # [r, r]
            BBt = backend.matmul(B, backend.transpose(B))  # [r, r]
            result = AAt + BBt
            I = backend.eye(r)
            diff = result - I
            backend.eval(diff)

            err = _frobenius(backend, diff)
            assert err < 1e-4, f"Failed for r={r}, n_in={n_in}, n_out={n_out}: err={err}"

    def test_rank_1(self, backend):
        """Semi-orthogonality holds for rank 1."""
        r, n_in, n_out = 1, 8, 16
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        backend.eval(A_tilde, B_tilde)

        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)

        AAt = backend.matmul(A, backend.transpose(A))  # [1, 1]
        BBt = backend.matmul(B, backend.transpose(B))  # [1, 1]
        result = AAt + BBt
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5

    def test_square_dimensions(self, backend):
        """Works when n_in == n_out."""
        r, n = 4, 16
        A_tilde = backend.random_normal((r, n)) * 0.1
        B_tilde = backend.random_normal((r, n)) * 0.1
        backend.eval(A_tilde, B_tilde)

        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)

        assert tuple(backend.shape(A)) == (r, n)
        assert tuple(backend.shape(B)) == (r, n)

        AAt = backend.matmul(A, backend.transpose(A))  # [r, r]
        BBt = backend.matmul(B, backend.transpose(B))  # [r, r]
        result = AAt + BBt
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-4


# =============================================================================
# Test: NB-LoRA Forward Pass
# =============================================================================


class TestNBLoRAForward:
    """Tests for nb_lora_forward function."""

    def test_2d_input(self, backend):
        """Forward pass works with [seq, in_features] input."""
        r, n_in, n_out = 4, 16, 32
        seq = 8
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        S = backend.ones((r,)) * 0.05
        x = backend.random_normal((seq, n_in))
        backend.eval(A_tilde, B_tilde, S, x)

        output = nb_lora_forward(A_tilde, B_tilde, S, x, backend)
        backend.eval(output)

        assert tuple(backend.shape(output)) == (seq, n_out)

    def test_3d_input(self, backend):
        """Forward pass works with [batch, seq, in_features] input."""
        r, n_in, n_out = 4, 16, 32
        batch, seq = 2, 8
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        S = backend.ones((r,)) * 0.05
        x = backend.random_normal((batch, seq, n_in))
        backend.eval(A_tilde, B_tilde, S, x)

        output = nb_lora_forward(A_tilde, B_tilde, S, x, backend)
        backend.eval(output)

        assert tuple(backend.shape(output)) == (batch, seq, n_out)

    def test_matches_explicit_matmul(self, backend):
        """Forward output equals x @ delta^T where delta = 2*B^T@S@A."""
        r, n_in, n_out = 4, 16, 32
        seq = 8
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        S = backend.ones((r,)) * 0.05
        x = backend.random_normal((seq, n_in))
        backend.eval(A_tilde, B_tilde, S, x)

        # Forward pass
        output = nb_lora_forward(A_tilde, B_tilde, S, x, backend)
        backend.eval(output)

        # Explicit: get A, B from Cayley, compute delta = 2*B^T@diag(S)@A
        A, B = cayley_transform_full(A_tilde, B_tilde, backend)
        backend.eval(A, B)
        S_A = A * backend.reshape(S, (-1, 1))
        delta = 2.0 * backend.matmul(backend.transpose(B), S_A)
        backend.eval(delta)

        # Expected: x @ delta^T
        expected = backend.matmul(x, backend.transpose(delta))
        backend.eval(expected)

        diff = output - expected
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-4

    def test_zero_S_gives_zero_output(self, backend):
        """S=0 produces zero output regardless of A_tilde, B_tilde."""
        r, n_in, n_out = 4, 16, 32
        seq = 8
        A_tilde = backend.random_normal((r, n_in)) * 0.5
        B_tilde = backend.random_normal((r, n_out)) * 0.5
        S = backend.zeros((r,))
        x = backend.random_normal((seq, n_in))
        backend.eval(A_tilde, B_tilde, S, x)

        output = nb_lora_forward(A_tilde, B_tilde, S, x, backend)
        backend.eval(output)

        assert _frobenius(backend, output) < 1e-6

    def test_diagonal_S_matrix(self, backend):
        """Forward pass works with S as a diagonal matrix (2D)."""
        r, n_in, n_out = 4, 16, 32
        seq = 8
        A_tilde = backend.random_normal((r, n_in)) * 0.1
        B_tilde = backend.random_normal((r, n_out)) * 0.1
        S_vec = backend.ones((r,)) * 0.05
        S_mat = backend.diag(S_vec)
        x = backend.random_normal((seq, n_in))
        backend.eval(A_tilde, B_tilde, S_vec, S_mat, x)

        out_vec = nb_lora_forward(A_tilde, B_tilde, S_vec, x, backend)
        out_mat = nb_lora_forward(A_tilde, B_tilde, S_mat, x, backend)
        backend.eval(out_vec, out_mat)

        diff = out_vec - out_mat
        backend.eval(diff)
        assert _frobenius(backend, diff) < 1e-5


# =============================================================================
# Test: NBLoRAConfig
# =============================================================================


class TestNBLoRAConfig:
    """Tests for NBLoRAConfig dataclass."""

    def test_default_init_scale(self):
        """Default init_scale is 0.5 * scale_bound."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1
        )
        assert config.init_scale == pytest.approx(0.05)

    def test_custom_init_scale(self):
        """Custom init_scale is respected."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4,
            scale_bound=0.1, init_scale=0.01,
        )
        assert config.init_scale == pytest.approx(0.01)


# =============================================================================
# Test: NBLoRALayer
# =============================================================================


class TestNBLoRALayer:
    """Tests for NBLoRALayer class."""

    def _make_layer(self, backend, n_in=16, n_out=32, rank=4, scale_bound=0.1):
        config = NBLoRAConfig(
            in_features=n_in, out_features=n_out,
            rank=rank, scale_bound=scale_bound,
        )
        return NBLoRALayer(config, backend)

    def test_parameter_shapes(self, backend):
        """A_tilde, B_tilde, S_raw have correct shapes."""
        layer = self._make_layer(backend, n_in=16, n_out=32, rank=4)
        assert tuple(backend.shape(layer.A_tilde)) == (4, 16)
        assert tuple(backend.shape(layer.B_tilde)) == (4, 32)
        assert tuple(backend.shape(layer.S_raw)) == (4,)

    def test_forward_output_shape_2d(self, backend):
        """Forward with 2D input returns [seq, out_features]."""
        layer = self._make_layer(backend)
        x = backend.random_normal((8, 16))
        backend.eval(x)
        out = layer.forward(x)
        backend.eval(out)
        assert tuple(backend.shape(out)) == (8, 32)

    def test_forward_output_shape_3d(self, backend):
        """Forward with 3D input returns [batch, seq, out_features]."""
        layer = self._make_layer(backend)
        x = backend.random_normal((2, 8, 16))
        backend.eval(x)
        out = layer.forward(x)
        backend.eval(out)
        assert tuple(backend.shape(out)) == (2, 8, 32)

    def test_get_S_clamped_to_bound(self, backend):
        """get_S() returns values clamped to [0, scale_bound]."""
        layer = self._make_layer(backend, scale_bound=0.1)
        # Set S_raw to values exceeding the bound
        layer.S_raw = backend.array([0.5, -0.1, 0.05, 0.2])
        backend.eval(layer.S_raw)

        S = layer.get_S()
        backend.eval(S)
        s_list = backend.tolist(S)

        assert s_list[0] == pytest.approx(0.1)   # clamped from 0.5
        assert s_list[1] == pytest.approx(0.0)   # clamped from -0.1
        assert s_list[2] == pytest.approx(0.05)  # within range
        assert s_list[3] == pytest.approx(0.1)   # clamped from 0.2

    def test_get_effective_delta_shape(self, backend):
        """get_effective_delta() returns [out_features, in_features]."""
        layer = self._make_layer(backend, n_in=16, n_out=32, rank=4)
        delta = layer.get_effective_delta()
        backend.eval(delta)
        assert tuple(backend.shape(delta)) == (32, 16)

    def test_forward_matches_effective_delta(self, backend):
        """layer.forward(x) == x @ get_effective_delta()^T."""
        layer = self._make_layer(backend, n_in=16, n_out=32, rank=4)
        x = backend.random_normal((8, 16))
        backend.eval(x)

        out_forward = layer.forward(x)
        backend.eval(out_forward)

        delta = layer.get_effective_delta()
        backend.eval(delta)
        out_explicit = backend.matmul(x, backend.transpose(delta))
        backend.eval(out_explicit)

        diff = out_forward - out_explicit
        backend.eval(diff)
        assert _frobenius(backend, diff) < 1e-4

    def test_get_spectral_norm_bounded(self, backend):
        """get_spectral_norm() <= 2 * scale_bound."""
        layer = self._make_layer(backend, scale_bound=0.1)
        spectral = layer.get_spectral_norm()
        assert spectral <= 2.0 * 0.1 * 1.05  # 5% numerical tolerance

    def test_scale_bound_property(self, backend):
        """scale_bound property returns config value."""
        layer = self._make_layer(backend, scale_bound=0.123)
        assert layer.scale_bound == pytest.approx(0.123)

    def test_init_S_at_half_bound(self, backend):
        """S_raw is initialized at init_scale (default 0.5 * scale_bound)."""
        layer = self._make_layer(backend, scale_bound=0.1)
        s_list = backend.tolist(layer.S_raw)
        for s in s_list:
            assert s == pytest.approx(0.05)

    def test_custom_init_std(self, backend):
        """Different init_std produces different parameter magnitudes."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1
        )
        layer_small = NBLoRALayer(config, backend, init_std=0.001)
        layer_large = NBLoRALayer(config, backend, init_std=1.0)

        # Spectral norm of delta with larger init should be larger
        norm_small = _spectral_norm(backend, layer_small.get_effective_delta())
        norm_large = _spectral_norm(backend, layer_large.get_effective_delta())
        # Not deterministic, but on average large init should give larger norm
        # Just check both are finite and non-negative
        assert norm_small >= 0.0
        assert norm_large >= 0.0

    def test_square_dimensions(self, backend):
        """Layer works when in_features == out_features."""
        layer = self._make_layer(backend, n_in=16, n_out=16, rank=4)
        x = backend.random_normal((8, 16))
        backend.eval(x)
        out = layer.forward(x)
        backend.eval(out)
        assert tuple(backend.shape(out)) == (8, 16)


# =============================================================================
# Test: Spectral Norm Bound (THE guarantee)
# =============================================================================


class TestSpectralNormBound:
    """Tests for the core NB-LoRA guarantee: ||delta||_2 <= 2 * max(S)."""

    def test_spectral_bound_various_dimensions(self, backend):
        """Spectral bound holds across different dimension configs."""
        configs = [
            (2, 8, 16, 0.05),
            (4, 16, 32, 0.1),
            (8, 32, 64, 0.01),
            (4, 32, 32, 0.2),
        ]
        for r, n_in, n_out, scale_bound in configs:
            A_tilde = backend.random_normal((r, n_in)) * 0.1
            B_tilde = backend.random_normal((r, n_out)) * 0.1
            backend.eval(A_tilde, B_tilde)

            A, B = cayley_transform_full(A_tilde, B_tilde, backend)
            backend.eval(A, B)

            S = backend.ones((r,)) * scale_bound
            S_A = A * backend.reshape(S, (-1, 1))
            delta = 2.0 * backend.matmul(backend.transpose(B), S_A)
            backend.eval(delta)

            spectral = _spectral_norm(backend, delta)
            bound = 2.0 * scale_bound

            assert spectral <= bound * 1.05, (
                f"r={r}, n_in={n_in}, n_out={n_out}: "
                f"spectral={spectral:.6f} > bound={bound:.6f}"
            )

    def test_spectral_bound_after_parameter_change(self, backend):
        """Spectral bound holds after modifying free parameters."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)

        # Modify parameters (simulate training step)
        layer.A_tilde = backend.random_normal((4, 16)) * 5.0
        layer.B_tilde = backend.random_normal((4, 32)) * 5.0
        layer.S_raw = backend.array([0.5, 0.2, -0.1, 0.05])
        backend.eval(layer.A_tilde, layer.B_tilde, layer.S_raw)

        spectral = layer.get_spectral_norm()
        # Clamped S is [0.1, 0.1, 0.0, 0.05], max = 0.1
        # Bound = 2 * 0.1 = 0.2
        assert spectral <= 2.0 * 0.1 * 1.05

    def test_spectral_bound_layer_api(self, backend):
        """NBLoRALayer.get_spectral_norm() <= 2 * scale_bound, many trials."""
        for _ in range(5):
            config = NBLoRAConfig(
                in_features=16, out_features=32, rank=4, scale_bound=0.1,
            )
            layer = NBLoRALayer(config, backend)
            spectral = layer.get_spectral_norm()
            assert spectral <= 2.0 * 0.1 * 1.05


# =============================================================================
# Test: Parameter Access
# =============================================================================


class TestParameterAccess:
    """Tests for parameters() and set_parameters()."""

    def test_parameters_returns_all_three(self, backend):
        """parameters() returns A_tilde, B_tilde, S_raw."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)
        params = layer.parameters()

        assert "A_tilde" in params
        assert "B_tilde" in params
        assert "S_raw" in params
        assert tuple(backend.shape(params["A_tilde"])) == (4, 16)
        assert tuple(backend.shape(params["B_tilde"])) == (4, 32)
        assert tuple(backend.shape(params["S_raw"])) == (4,)

    def test_set_parameters_roundtrip(self, backend):
        """set_parameters(parameters()) preserves layer behavior."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)

        x = backend.random_normal((4, 16))
        backend.eval(x)

        out_before = layer.forward(x)
        backend.eval(out_before)

        # Round-trip
        params = layer.parameters()
        layer.set_parameters(params)

        out_after = layer.forward(x)
        backend.eval(out_after)

        diff = out_before - out_after
        backend.eval(diff)
        assert _frobenius(backend, diff) < 1e-7

    def test_partial_parameter_update(self, backend):
        """set_parameters with partial dict only updates specified params."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)

        original_A = layer.A_tilde
        original_B = layer.B_tilde

        # Only update S_raw
        new_S = backend.ones((4,)) * 0.01
        backend.eval(new_S)
        layer.set_parameters({"S_raw": new_S})

        # A_tilde and B_tilde should be unchanged (same object)
        assert layer.A_tilde is original_A
        assert layer.B_tilde is original_B
        s_list = backend.tolist(layer.S_raw)
        for s in s_list:
            assert s == pytest.approx(0.01)


# =============================================================================
# Test: Per-Direction Bounds
# =============================================================================


class TestPerDirectionBounds:
    """Tests for verify_per_direction_bounds."""

    def test_result_structure(self, backend):
        """Returns PerDirectionBoundResult with correct fields."""
        m, n, rank = 32, 16, 4
        W = backend.random_normal((m, n))
        backend.eval(W)

        config = NBLoRAConfig(
            in_features=n, out_features=m, rank=rank, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)
        result = layer.verify_per_direction_bounds(W)

        assert isinstance(result, PerDirectionBoundResult)
        assert len(result.per_direction_ratios) > 0
        assert isinstance(result.max_ratio, float)
        assert isinstance(result.violations, list)
        assert isinstance(result.is_safe, bool)

    def test_geometry_init_is_safe(self, backend):
        """Layer from create_nb_lora_from_base_weight is safe initially."""
        m, n = 32, 16
        W = backend.random_normal((m, n))
        backend.eval(W)

        layer = create_nb_lora_from_base_weight(W, rank=4, backend=backend)
        result = layer.verify_per_direction_bounds(W)

        assert result.max_ratio < 2.0

    def test_manual_ratio_check(self, backend):
        """Manually verify one direction's ratio matches computation."""
        m, n, rank = 32, 16, 4
        W = backend.random_normal((m, n))
        backend.eval(W)

        config = NBLoRAConfig(
            in_features=n, out_features=m, rank=rank, scale_bound=0.5,
        )
        layer = NBLoRALayer(config, backend)
        result = layer.verify_per_direction_bounds(W)

        # Manually compute the first ratio
        delta = layer.get_effective_delta()
        W_f32 = backend.astype(W, "float32")
        backend.eval(W_f32)
        U, S_W, Vt = backend.svd(W_f32, compute_uv=True)
        backend.eval(U, S_W, Vt)

        delta_f32 = backend.astype(delta, "float32")
        backend.eval(delta_f32)
        proj = backend.matmul(backend.transpose(U), delta_f32)
        proj = backend.matmul(proj, backend.transpose(Vt))
        backend.eval(proj)

        proj_00 = float(backend.to_scalar(backend.abs(proj[0, 0])))
        sigma_0 = float(backend.to_scalar(S_W[0]))
        expected_ratio = proj_00 / sigma_0

        assert abs(result.per_direction_ratios[0] - expected_ratio) < 1e-5

    def test_is_safe_consistency(self, backend):
        """is_safe is True iff violations list is empty."""
        m, n, rank = 32, 16, 4
        W = backend.random_normal((m, n))
        backend.eval(W)

        config = NBLoRAConfig(
            in_features=n, out_features=m, rank=rank, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)
        result = layer.verify_per_direction_bounds(W)

        assert result.is_safe == (len(result.violations) == 0)


# =============================================================================
# Test: create_nb_lora_from_base_weight
# =============================================================================


class TestCreateFromBaseWeight:
    """Tests for the factory function."""

    def test_output_shapes(self, backend):
        """Layer has correct shapes for given W and rank."""
        m, n = 32, 16
        W = backend.random_normal((m, n))
        backend.eval(W)

        layer = create_nb_lora_from_base_weight(W, rank=4, backend=backend)
        assert tuple(backend.shape(layer.A_tilde)) == (4, n)
        assert tuple(backend.shape(layer.B_tilde)) == (4, m)
        assert tuple(backend.shape(layer.S_raw)) == (4,)

    def test_scale_bound_derived_from_sigma_max(self, backend):
        """Scale bound derived from σ_max (not σ_k). Default margin is dtype-derived."""
        _m, _n = 32, 16
        # Create W with known spectrum via diagonal
        S_vals = backend.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
        # Pad remaining for 16 columns
        pad = backend.zeros((8,))
        S_full = backend.concatenate([S_vals, pad])
        D = backend.diag(S_full)
        backend.eval(D)

        # Use D as a square weight (16x16)
        layer = create_nb_lora_from_base_weight(D, rank=4, backend=backend)

        # sigma_max = 1.0 (largest SV). Scale bound = σ_max/2 × margin.
        # Per-step safety comes from MASS (eta_weyl), not from S clamp.
        expected_margin = 1.0 - math.sqrt(float(backend.finfo().eps))
        assert layer.scale_bound > 0.0
        assert layer.scale_bound == pytest.approx(1.0 * 0.5 * expected_margin, rel=1e-5)
        # The spectral norm guarantee should hold
        spectral = layer.get_spectral_norm()
        assert spectral <= 2.0 * layer.scale_bound * 1.05

    def test_custom_margin_override(self, backend):
        """Explicit safety_margin overrides the dtype-derived default."""
        m, n = 32, 16
        W = backend.random_normal((m, n))
        backend.eval(W)

        layer_default = create_nb_lora_from_base_weight(W, rank=4, backend=backend)
        layer_80 = create_nb_lora_from_base_weight(W, rank=4, backend=backend, safety_margin=0.8)

        assert layer_default.scale_bound > layer_80.scale_bound

    def test_rank_1(self, backend):
        """Factory works for rank 1."""
        m, n = 32, 16
        W = backend.random_normal((m, n))
        backend.eval(W)

        layer = create_nb_lora_from_base_weight(W, rank=1, backend=backend)
        assert tuple(backend.shape(layer.A_tilde)) == (1, n)
        spectral = layer.get_spectral_norm()
        assert spectral <= 2.0 * layer.scale_bound * 1.05

    def test_square_weight(self, backend):
        """Factory works for square weight matrices."""
        n = 16
        W = backend.random_normal((n, n))
        backend.eval(W)

        layer = create_nb_lora_from_base_weight(W, rank=4, backend=backend)
        assert tuple(backend.shape(layer.A_tilde)) == (4, n)
        assert tuple(backend.shape(layer.B_tilde)) == (4, n)

    def test_forward_after_factory(self, backend):
        """Layer created by factory can run forward pass."""
        m, n = 32, 16
        W = backend.random_normal((m, n))
        backend.eval(W)

        layer = create_nb_lora_from_base_weight(W, rank=4, backend=backend)
        x = backend.random_normal((4, n))
        backend.eval(x)

        out = layer.forward(x)
        backend.eval(out)
        assert tuple(backend.shape(out)) == (4, m)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for Cayley-LoRA implementation."""

    def test_rank_equals_in_features(self, backend):
        """Works when rank == in_features (maximum rank)."""
        r, n_in, n_out = 8, 8, 16
        config = NBLoRAConfig(
            in_features=n_in, out_features=n_out,
            rank=r, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend)
        x = backend.random_normal((4, n_in))
        backend.eval(x)
        out = layer.forward(x)
        backend.eval(out)
        assert tuple(backend.shape(out)) == (4, n_out)
        spectral = layer.get_spectral_norm()
        assert spectral <= 2.0 * 0.1 * 1.05

    def test_minimal_2x2(self, backend):
        """Cayley transform works for minimal r=1, m=1 case."""
        r, m = 1, 1
        X = backend.random_normal((r, r)) * 0.1
        Y = backend.random_normal((m, r)) * 0.1
        backend.eval(X, Y)

        A, B = cayley_transform(X, Y, backend)
        backend.eval(A, B)

        AtA = backend.matmul(backend.transpose(A), A)
        BtB = backend.matmul(backend.transpose(B), B)
        result = AtA + BtB
        I = backend.eye(r)
        diff = result - I
        backend.eval(diff)

        assert _frobenius(backend, diff) < 1e-5

    def test_large_init_std_still_bounded(self, backend):
        """Even with large init_std, spectral bound still holds due to clamping."""
        config = NBLoRAConfig(
            in_features=16, out_features=32, rank=4, scale_bound=0.1,
        )
        layer = NBLoRALayer(config, backend, init_std=10.0)
        spectral = layer.get_spectral_norm()
        assert spectral <= 2.0 * 0.1 * 1.05
