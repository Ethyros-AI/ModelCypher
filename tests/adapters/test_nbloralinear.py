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

"""Tests for NBLoRALinear — the Cayley-parameterized LoRA module.

NBLoRALinear is the core training adapter. It wraps a base nn.Linear and
adds a norm-bounded LoRA contribution via the Cayley transform.

Every test verifies a mathematical invariant:
- Semi-orthogonality: A@A^T + B@B^T = I_r (Cayley construction)
- Spectral bound: ||delta||_2 <= 2 * scale_bound (by construction)
- Zero initialization: LoRA adds nothing before training
- Scale clamping: S_raw stays in [0, scale_bound]

No disk I/O. All tests use synthetic nn.Linear modules.
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import mlx.nn as nn

from modelcypher.backends.mlx_training_adapter_core import NBLoRALinear

pytestmark = pytest.mark.mlx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_nb(in_f=16, out_f=32, rank=4, scale_bound=0.1):
    """Create NBLoRALinear wrapping a synthetic nn.Linear."""
    base = nn.Linear(in_f, out_f, bias=False)
    nb = NBLoRALinear.from_base(base, rank=rank, scale_bound=scale_bound)
    mx.eval(*[v for _, v in nn.utils.tree_flatten(nb.parameters())])
    return nb


def _perturb(nb, scale=0.5):
    """Set A_tilde, B_tilde to random non-zero values for non-trivial tests."""
    r, n_in = nb.A_tilde.shape
    _, n_out = nb.B_tilde.shape
    nb.A_tilde = mx.random.normal((r, n_in)) * scale
    nb.B_tilde = mx.random.normal((r, n_out)) * scale
    mx.eval(nb.A_tilde, nb.B_tilde)


def _frobenius(M):
    """Frobenius norm of an MLX array."""
    mx.eval(M)
    return float(mx.sqrt(mx.sum(M * M)).item())


# ===================================================================
# Class 1: Construction
# ===================================================================
class TestConstruction:
    """Verify from_base creates correct shapes and initial values."""

    def test_a_tilde_shape(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        assert nb.A_tilde.shape == (4, 16)

    def test_b_tilde_shape(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        assert nb.B_tilde.shape == (4, 32)

    def test_s_raw_shape_and_init_value(self):
        nb = _make_nb(rank=4, scale_bound=0.1)
        assert nb.S_raw.shape == (4,)
        expected = 0.1 / 2.0
        for i in range(4):
            assert float(nb.S_raw[i].item()) == pytest.approx(expected)

    def test_a_tilde_b_tilde_init_zero(self):
        nb = _make_nb()
        assert _frobenius(nb.A_tilde) == 0.0
        assert _frobenius(nb.B_tilde) == 0.0

    def test_scale_bound_property(self):
        nb = _make_nb(scale_bound=0.42)
        assert nb.scale_bound == 0.42

    def test_from_base_quantized_linear(self):
        """QuantizedLinear unpacks dims: input_dims = weight.shape[1] * 32 // bits."""
        base = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)
        nb = NBLoRALinear.from_base(base, rank=4, scale_bound=0.1)
        mx.eval(*[v for _, v in nn.utils.tree_flatten(nb.parameters())])
        # weight shape is (64, 16) for quantized, but input should unpack to 128
        assert nb.A_tilde.shape == (4, 128)
        assert nb.B_tilde.shape == (4, 64)


# ===================================================================
# Class 2: Zero Initialization — "do no harm" guarantee
# ===================================================================
class TestZeroInit:
    """At init, LoRA contributes exactly zero."""

    def test_cayley_forward_zero_at_init(self):
        nb = _make_nb()
        x = mx.random.normal((4, 16))
        mx.eval(x)
        lora_out = nb._cayley_forward(x)
        mx.eval(lora_out)
        assert _frobenius(lora_out) == 0.0

    def test_full_forward_equals_base_at_init(self):
        nb = _make_nb(in_f=16, out_f=32)
        x = mx.random.normal((4, 16))
        mx.eval(x)
        full_out = nb(x)
        base_out = nb.linear(x)
        mx.eval(full_out, base_out)
        diff = _frobenius(full_out - base_out)
        assert diff == 0.0

    def test_effective_delta_zero_at_init(self):
        nb = _make_nb()
        delta = nb.get_effective_delta()
        assert _frobenius(delta) == 0.0


# ===================================================================
# Class 3: Cayley Semi-Orthogonality
# ===================================================================
class TestCayleySemiOrthogonality:
    """Cayley transform produces A, B where A@A^T + B@B^T = I_r."""

    def _check_semi_ortho(self, nb, tol=1e-5):
        _perturb(nb)
        A, B = nb._cayley_transform()
        mx.eval(A, B)
        r = A.shape[0]
        gram = A @ A.T + B @ B.T
        mx.eval(gram)
        err = _frobenius(gram - mx.eye(r))
        assert err < tol, f"Semi-orthogonality error: {err:.2e}"

    def test_standard_dims(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        self._check_semi_ortho(nb)

    def test_stacked_columns_orthonormal(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        A, B = nb._cayley_transform()
        mx.eval(A, B)
        # [A^T; B^T] has shape [(in+out), r]
        stacked = mx.concatenate([A.T, B.T], axis=0)
        gram = stacked.T @ stacked
        mx.eval(gram)
        r = A.shape[0]
        err = _frobenius(gram - mx.eye(r))
        assert err < 1e-5, f"Orthonormal columns error: {err:.2e}"

    def test_rank_1_edge_case(self):
        nb = _make_nb(in_f=8, out_f=16, rank=1)
        self._check_semi_ortho(nb)

    def test_square_dims(self):
        nb = _make_nb(in_f=32, out_f=32, rank=4)
        self._check_semi_ortho(nb)

    def test_large_perturbation(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb, scale=5.0)
        A, B = nb._cayley_transform()
        mx.eval(A, B)
        r = A.shape[0]
        gram = A @ A.T + B @ B.T
        mx.eval(gram)
        err = _frobenius(gram - mx.eye(r))
        assert err < 1e-4, f"Semi-orthogonality error (large params): {err:.2e}"


# ===================================================================
# Class 4: Spectral Bound — THE guarantee
# ===================================================================
class TestSpectralBound:
    """||delta||_2 <= 2 * scale_bound by construction."""

    def test_bounded_at_default_init(self):
        nb = _make_nb(scale_bound=0.1)
        # At zero init, spectral norm should be 0
        assert nb.get_spectral_norm() <= 2 * 0.1

    def test_bounded_after_perturbation(self):
        nb = _make_nb(scale_bound=0.1)
        _perturb(nb)
        assert nb.get_spectral_norm() <= 2 * 0.1 * 1.05  # 5% numerical tolerance

    def test_bounded_extreme_params(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4, scale_bound=0.1)
        _perturb(nb, scale=10.0)
        assert nb.get_spectral_norm() <= 2 * 0.1 * 1.05

    @pytest.mark.parametrize("rank,in_f,out_f,bound", [
        (2, 8, 16, 0.05),
        (4, 32, 64, 0.1),
        (8, 32, 32, 0.01),
        (1, 16, 64, 0.2),
    ])
    def test_bounded_various_configs(self, rank, in_f, out_f, bound):
        nb = _make_nb(in_f=in_f, out_f=out_f, rank=rank, scale_bound=bound)
        _perturb(nb)
        assert nb.get_spectral_norm() <= 2 * bound * 1.05

    def test_spectral_norm_zero_when_s_raw_zero(self):
        nb = _make_nb(scale_bound=0.1)
        _perturb(nb)
        nb.S_raw = mx.zeros_like(nb.S_raw)
        mx.eval(nb.S_raw)
        assert nb.get_spectral_norm() == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# Class 5: Scale Clamping
# ===================================================================
class TestScaleClamping:
    """clamp_scale() enforces [0, scale_bound] — called every optimizer step."""

    def test_clamp_enforces_upper_bound(self):
        nb = _make_nb(rank=2, scale_bound=0.1)
        nb.S_raw = mx.array([0.5, 0.3])
        mx.eval(nb.S_raw)
        nb.clamp_scale()
        mx.eval(nb.S_raw)
        for i in range(2):
            assert float(nb.S_raw[i].item()) == pytest.approx(0.1)

    def test_clamp_enforces_lower_bound(self):
        nb = _make_nb(rank=2, scale_bound=0.1)
        nb.S_raw = mx.array([-0.1, -0.5])
        mx.eval(nb.S_raw)
        nb.clamp_scale()
        mx.eval(nb.S_raw)
        for i in range(2):
            assert float(nb.S_raw[i].item()) == pytest.approx(0.0)

    def test_clamp_preserves_valid_values(self):
        nb = _make_nb(rank=3, scale_bound=0.1)
        valid = mx.array([0.0, 0.05, 0.1])
        nb.S_raw = valid
        mx.eval(nb.S_raw)
        nb.clamp_scale()
        mx.eval(nb.S_raw)
        for i, expected in enumerate([0.0, 0.05, 0.1]):
            assert float(nb.S_raw[i].item()) == pytest.approx(expected)

    def test_forward_uses_clamped_s_internally(self):
        """Even without explicit clamp_scale(), _cayley_forward clips S_raw."""
        nb = _make_nb(rank=4, scale_bound=0.1)
        _perturb(nb)
        # Set S_raw way above bound
        nb.S_raw = mx.ones((4,)) * 10.0
        mx.eval(nb.S_raw)

        x = mx.random.normal((4, 16))
        mx.eval(x)
        out_unclamped = nb._cayley_forward(x)
        mx.eval(out_unclamped)

        # Now clamp and compute again — should be identical
        nb.clamp_scale()
        mx.eval(nb.S_raw)
        out_clamped = nb._cayley_forward(x)
        mx.eval(out_clamped)

        diff = _frobenius(out_unclamped - out_clamped)
        assert diff < 1e-6


# ===================================================================
# Class 6: Forward Pass
# ===================================================================
class TestForwardPass:
    """Forward correctness and shape preservation."""

    def test_output_shape_2d(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((8, 16))
        mx.eval(x)
        out = nb(x)
        mx.eval(out)
        assert out.shape == (8, 32)

    def test_output_shape_3d(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((2, 8, 16))
        mx.eval(x)
        out = nb(x)
        mx.eval(out)
        assert out.shape == (2, 8, 32)

    def test_forward_matches_explicit_delta(self):
        """nb(x) - base(x) == x @ delta.T."""
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((4, 16))
        mx.eval(x)

        full_out = nb(x)
        base_out = nb.linear(x)
        delta = nb.get_effective_delta()
        mx.eval(full_out, base_out, delta)

        lora_via_forward = full_out - base_out
        lora_via_delta = x @ delta.T
        mx.eval(lora_via_forward, lora_via_delta)

        err = _frobenius(lora_via_forward - lora_via_delta)
        assert err < 1e-5, f"Forward-delta mismatch: {err:.2e}"

    def test_output_dtype_matches_input(self):
        nb = _make_nb(in_f=16, out_f=32)
        _perturb(nb)
        x = mx.random.normal((4, 16))
        mx.eval(x)
        out = nb(x)
        mx.eval(out)
        assert out.dtype == x.dtype

    def test_forward_deterministic(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((4, 16))
        mx.eval(x)

        out1 = nb(x)
        out2 = nb(x)
        mx.eval(out1, out2)

        assert _frobenius(out1 - out2) == 0.0


# ===================================================================
# Class 7: Conversion (to_standard_lora, get_effective_delta)
# ===================================================================
class TestConversion:
    """to_standard_lora() and get_effective_delta() for saving adapters."""

    def test_to_standard_lora_shapes(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        lora_a, lora_b = nb.to_standard_lora()
        mx.eval(lora_a, lora_b)
        assert lora_a.shape == (16, 4)   # [in, r]
        assert lora_b.shape == (4, 32)   # [r, out]

    def test_to_standard_lora_forward_equivalence(self):
        """x @ lora_a @ lora_b == _cayley_forward(x)."""
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((4, 16))
        mx.eval(x)

        lora_a, lora_b = nb.to_standard_lora()
        lora_out = x @ lora_a @ lora_b
        cayley_out = nb._cayley_forward(x)
        mx.eval(lora_out, cayley_out)

        err = _frobenius(lora_out - cayley_out)
        assert err < 1e-5, f"Standard LoRA vs Cayley mismatch: {err:.2e}"

    def test_to_standard_lora_delta_equivalence(self):
        """lora_b.T @ lora_a.T == get_effective_delta()."""
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)

        lora_a, lora_b = nb.to_standard_lora()
        delta_from_lora = lora_b.T @ lora_a.T  # [out, in]
        delta_direct = nb.get_effective_delta()
        mx.eval(delta_from_lora, delta_direct)

        err = _frobenius(delta_from_lora - delta_direct)
        assert err < 1e-5, f"Delta equivalence error: {err:.2e}"

    def test_effective_delta_shape(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        delta = nb.get_effective_delta()
        assert delta.shape == (32, 16)  # [out, in]

    def test_effective_delta_matches_forward(self):
        """_cayley_forward(x) == x @ delta.T."""
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb)
        x = mx.random.normal((4, 16))
        mx.eval(x)

        cayley_out = nb._cayley_forward(x)
        delta = nb.get_effective_delta()
        delta_out = x @ delta.T
        mx.eval(cayley_out, delta_out)

        err = _frobenius(cayley_out - delta_out)
        assert err < 1e-5, f"Delta-forward mismatch: {err:.2e}"


# ===================================================================
# Class 8: Autograd
# ===================================================================
class TestAutograd:
    """Gradient flow through the Cayley transform via _inv_with_grad VJP."""

    def test_gradients_flow_to_all_params(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb, scale=0.1)
        nb.linear.freeze()

        x = mx.random.normal((4, 16))
        mx.eval(x)

        def loss_fn(model):
            return mx.mean(model(x))

        loss, grads = nn.value_and_grad(nb, loss_fn)(nb)
        mx.eval(loss)

        # Flatten grads to find our three parameters
        flat_grads = dict(nn.utils.tree_flatten(grads))

        for name in ["A_tilde", "B_tilde", "S_raw"]:
            assert name in flat_grads, f"Missing gradient for {name}"
            g = flat_grads[name]
            mx.eval(g)
            norm = _frobenius(g)
            assert norm > 0, f"Zero gradient for {name}"

    def test_gradient_shapes_match_params(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        _perturb(nb, scale=0.1)
        nb.linear.freeze()

        x = mx.random.normal((4, 16))
        mx.eval(x)

        def loss_fn(model):
            return mx.mean(model(x))

        _, grads = nn.value_and_grad(nb, loss_fn)(nb)

        flat_grads = dict(nn.utils.tree_flatten(grads))
        assert flat_grads["A_tilde"].shape == (4, 16)
        assert flat_grads["B_tilde"].shape == (4, 32)
        assert flat_grads["S_raw"].shape == (4,)

    def test_base_linear_frozen(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        nb.linear.freeze()
        trainable = dict(nn.utils.tree_flatten(nb.trainable_parameters()))
        param_names = set(trainable.keys())
        assert param_names == {"A_tilde", "B_tilde", "S_raw"}


# ===================================================================
# Class 9: Initialization Vectors
# ===================================================================
class TestInitializationVectors:
    """set_initialization_vectors stores frozen diagnostic vectors."""

    def test_set_and_get_vectors(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        u_k = mx.random.normal((32, 1))
        v_k = mx.random.normal((16, 1))
        mx.eval(u_k, v_k)

        nb.set_initialization_vectors(u_k, v_k)

        assert nb.base_u_k is not None
        assert nb.base_v_k is not None
        assert nb.base_u_k.shape == (32, 1)
        assert nb.base_v_k.shape == (16, 1)

    def test_vectors_are_frozen(self):
        nb = _make_nb(in_f=16, out_f=32, rank=4)
        nb.linear.freeze()
        u_k = mx.random.normal((32, 1))
        v_k = mx.random.normal((16, 1))
        mx.eval(u_k, v_k)

        nb.set_initialization_vectors(u_k, v_k)

        trainable = dict(nn.utils.tree_flatten(nb.trainable_parameters()))
        assert "_base_u_k" not in trainable
        assert "_base_v_k" not in trainable
        # Only the 3 LoRA params should be trainable
        assert set(trainable.keys()) == {"A_tilde", "B_tilde", "S_raw"}
