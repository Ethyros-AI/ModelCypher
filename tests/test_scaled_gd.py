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

"""Tests for ScaledGD preconditioning.

Backend-agnostic. Uses synthetic data, no model loading.

1. Identity-like B -> preconditioner ~ (1/eps)I (uniform scaling at init)
2. Known A,B -> verify grad_A @ (BBt+eI)^{-1} matches manual computation
3. Decay adds decay * param to preconditioned gradient
4. Missing keys -> skipped gracefully
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.geometric_optimizer import (
    LayerOptimizerConfig,
    OptimizerGeometryConfig,
)
from modelcypher.core.domain.training.scaled_gd import precondition_lora_gradients
from modelcypher.ports.training import LoRALayerConfig


@pytest.fixture
def backend():
    """Get the default compute backend."""
    return get_default_backend()


def _make_opt_config(layer_key: str, epsilon: float, decay: float) -> OptimizerGeometryConfig:
    """Create a minimal OptimizerGeometryConfig for testing."""
    return OptimizerGeometryConfig(
        base_lr=0.01,
        max_sigma=1.0,
        layer_configs={
            layer_key: LayerOptimizerConfig(
                layer_key=layer_key,
                sigma_max=1.0,
                sigma_k=0.1,
                lr_scale=1.0,
                epsilon=epsilon,
                decay_scale=decay,
                spectral_gap=0.0,
            )
        },
    )


class TestPreconditionLoraGradients:
    """Tests for precondition_lora_gradients."""

    def test_identity_b_uniform_scaling(self, backend):
        """When B ~ identity-like (small), preconditioner ~ (1/eps)I."""
        b = backend
        rank = 4
        in_dim = 8
        out_dim = 8

        # Small B (near zero) -> BBt ~ 0 -> (BBt + eI)^-1 ~ (1/e)I
        eps = 1.0
        lora_a = b.random_normal((in_dim, rank))
        lora_b = b.zeros((rank, out_dim))  # B = 0
        b.eval(lora_a, lora_b)

        grad_a = b.random_normal((in_dim, rank))
        grad_b = b.random_normal((rank, out_dim))
        b.eval(grad_a, grad_b)

        layer_key = "model.layers.0.self_attn.q_proj.weight"
        prefix = layer_key.replace(".weight", "")

        grad_flat = {
            prefix + ".lora_a": grad_a,
            prefix + ".lora_b": grad_b,
        }
        param_flat = {
            prefix + ".lora_a": lora_a,
            prefix + ".lora_b": lora_b,
        }

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=rank, sigma_k=0.1,
            in_features=in_dim, out_features=out_dim,
        )
        opt_config = _make_opt_config(layer_key, epsilon=eps, decay=0.0)

        result = precondition_lora_gradients(
            grad_flat, param_flat, [cfg], opt_config, b,
        )

        # With B=0: (0 + eps*I)^-1 = (1/eps)*I
        # So result[a_key] = grad_a @ (1/eps)*I = grad_a / eps
        result_a = result[prefix + ".lora_a"]
        expected_a = grad_a  # eps=1.0 so (1/1)*I = I
        b.eval(result_a)

        diff = b.abs(result_a - expected_a)
        max_diff = float(b.to_scalar(b.max(diff)))
        assert max_diff < 1e-4, f"Expected uniform scaling, max diff = {max_diff}"

    def test_known_preconditioner_computation(self, backend):
        """Verify exact preconditioning math with known A, B."""
        b = backend
        rank = 2
        in_dim = 4
        out_dim = 4

        # Known matrices
        import numpy as np

        np.random.seed(123)
        A_np = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
        B_np = np.array([[2.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        grad_a_np = np.ones((in_dim, rank), dtype=np.float32)
        grad_b_np = np.ones((rank, out_dim), dtype=np.float32)

        eps = 0.01

        # Manual computation
        BBt_np = B_np @ B_np.T + eps * np.eye(rank, dtype=np.float32)
        BBt_inv_np = np.linalg.inv(BBt_np)
        expected_grad_a = grad_a_np @ BBt_inv_np

        AtA_np = A_np.T @ A_np + eps * np.eye(rank, dtype=np.float32)
        AtA_inv_np = np.linalg.inv(AtA_np)
        expected_grad_b = AtA_inv_np @ grad_b_np

        # Backend computation
        layer_key = "model.layers.0.self_attn.q_proj.weight"
        prefix = layer_key.replace(".weight", "")

        grad_flat = {
            prefix + ".lora_a": b.array(grad_a_np),
            prefix + ".lora_b": b.array(grad_b_np),
        }
        param_flat = {
            prefix + ".lora_a": b.array(A_np),
            prefix + ".lora_b": b.array(B_np),
        }

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=rank, sigma_k=0.1,
            in_features=in_dim, out_features=out_dim,
        )
        opt_config = _make_opt_config(layer_key, epsilon=eps, decay=0.0)

        result = precondition_lora_gradients(
            grad_flat, param_flat, [cfg], opt_config, b,
        )

        result_a = result[prefix + ".lora_a"]
        result_b = result[prefix + ".lora_b"]
        b.eval(result_a, result_b)

        # Compare with numpy result using tolist (to_numpy is disabled)
        result_a_list = b.tolist(result_a)
        result_b_list = b.tolist(result_b)

        for i in range(in_dim):
            for j in range(rank):
                assert abs(result_a_list[i][j] - expected_grad_a[i][j]) < 1e-4, (
                    f"grad_A[{i},{j}] mismatch: {result_a_list[i][j]} vs {expected_grad_a[i][j]}"
                )

        for i in range(rank):
            for j in range(out_dim):
                assert abs(result_b_list[i][j] - expected_grad_b[i][j]) < 1e-4, (
                    f"grad_B[{i},{j}] mismatch: {result_b_list[i][j]} vs {expected_grad_b[i][j]}"
                )

    def test_decay_adds_to_gradient(self, backend):
        """Decay should add decay * param to preconditioned gradient."""
        b = backend
        rank = 2
        in_dim = 4
        out_dim = 4
        decay = 0.1

        layer_key = "model.layers.0.self_attn.q_proj.weight"
        prefix = layer_key.replace(".weight", "")

        lora_a = b.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        lora_b = b.zeros((rank, out_dim))
        grad_a = b.zeros((in_dim, rank))
        grad_b = b.zeros((rank, out_dim))
        b.eval(lora_a, lora_b, grad_a, grad_b)

        grad_flat = {
            prefix + ".lora_a": grad_a,
            prefix + ".lora_b": grad_b,
        }
        param_flat = {
            prefix + ".lora_a": lora_a,
            prefix + ".lora_b": lora_b,
        }

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=rank, sigma_k=0.1,
            in_features=in_dim, out_features=out_dim,
        )
        opt_config = _make_opt_config(layer_key, epsilon=1.0, decay=decay)

        result = precondition_lora_gradients(
            grad_flat, param_flat, [cfg], opt_config, b,
        )

        # With zero gradients, result should be decay * param
        result_a = result[prefix + ".lora_a"]
        expected_a = decay * lora_a
        b.eval(result_a, expected_a)

        diff = b.abs(result_a - expected_a)
        max_diff = float(b.to_scalar(b.max(diff)))
        assert max_diff < 1e-5, f"Decay mismatch, max diff = {max_diff}"

    def test_missing_keys_skipped(self, backend):
        """Missing lora keys in param_flat should be skipped gracefully."""
        b = backend
        layer_key = "model.layers.0.self_attn.q_proj.weight"

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=4, sigma_k=0.1,
            in_features=8, out_features=8,
        )
        opt_config = _make_opt_config(layer_key, epsilon=1e-7, decay=0.0)

        # Empty dicts -> nothing to precondition
        result = precondition_lora_gradients({}, {}, [cfg], opt_config, b)
        assert result == {}

    def test_zero_rank_skipped(self, backend):
        """Configs with rank=0 should be skipped."""
        b = backend
        layer_key = "model.layers.0.self_attn.q_proj.weight"

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=0, sigma_k=0.1,
            in_features=8, out_features=8,
        )
        opt_config = _make_opt_config(layer_key, epsilon=1e-7, decay=0.0)

        grad_flat = {"some_key": b.array([1.0, 2.0])}
        result = precondition_lora_gradients(grad_flat, {}, [cfg], opt_config, b)
        # Original should pass through unchanged
        assert "some_key" in result

    def test_does_not_mutate_input(self, backend):
        """Original grad_flat dict should not be mutated."""
        b = backend
        rank = 2
        in_dim = 4
        out_dim = 4

        layer_key = "model.layers.0.self_attn.q_proj.weight"
        prefix = layer_key.replace(".weight", "")

        original_grad_a = b.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        b.eval(original_grad_a)

        grad_flat = {
            prefix + ".lora_a": original_grad_a,
            prefix + ".lora_b": b.zeros((rank, out_dim)),
        }
        param_flat = {
            prefix + ".lora_a": b.eye(in_dim, rank),
            prefix + ".lora_b": b.zeros((rank, out_dim)),
        }

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=rank, sigma_k=0.1,
            in_features=in_dim, out_features=out_dim,
        )
        opt_config = _make_opt_config(layer_key, epsilon=1.0, decay=0.0)

        # Save reference to original array
        original_ref = grad_flat[prefix + ".lora_a"]

        precondition_lora_gradients(grad_flat, param_flat, [cfg], opt_config, b)

        # The dict entry should still point to the original array
        assert grad_flat[prefix + ".lora_a"] is original_ref

    def test_fallback_epsilon_is_sqrt_eps(self, backend):
        """When layer_opt is None, fallback epsilon should be sqrt(eps_mach).

        The fallback derives epsilon from the tensor's dtype machine epsilon,
        not an arbitrary constant like 1e-7.
        """
        b = backend
        rank = 2
        in_dim = 4
        out_dim = 4

        layer_key = "model.layers.99.self_attn.q_proj.weight"  # Not in opt_config
        prefix = layer_key.replace(".weight", "")

        lora_a = b.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        lora_b = b.zeros((rank, out_dim))
        grad_a = b.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        grad_b = b.zeros((rank, out_dim))
        b.eval(lora_a, lora_b, grad_a, grad_b)

        grad_flat = {
            prefix + ".lora_a": grad_a,
            prefix + ".lora_b": grad_b,
        }
        param_flat = {
            prefix + ".lora_a": lora_a,
            prefix + ".lora_b": lora_b,
        }

        cfg = LoRALayerConfig(
            layer_key=layer_key, rank=rank, sigma_k=0.1,
            in_features=in_dim, out_features=out_dim,
        )
        # Use an opt_config that does NOT contain the layer key
        opt_config = OptimizerGeometryConfig(
            base_lr=0.01,
            max_sigma=1.0,
            layer_configs={},  # Empty — forces fallback
        )

        result = precondition_lora_gradients(
            grad_flat, param_flat, [cfg], opt_config, b,
        )

        # With B=0: (0 + eps*I)^-1 = (1/eps)*I
        # result[a_key] = grad_a @ (1/eps)*I = grad_a / eps
        # The fallback eps = sqrt(machine_eps) ~ 3.45e-4 for float32
        # So result should be grad_a / eps ~ grad_a * 2899
        result_a = result[prefix + ".lora_a"]
        b.eval(result_a)

        # Just verify it's not using 1e-7 (which would give grad_a * 1e7)
        # sqrt(eps_f32) ~ 3.45e-4, so 1/eps ~ 2899
        result_max = float(b.to_scalar(b.max(b.abs(result_a))))
        assert result_max < 1e5, (
            f"Fallback epsilon seems too small (result_max={result_max}). "
            f"Expected sqrt(eps_mach)-scale, not 1e-7-scale."
        )
        assert result_max > 100, (
            f"Fallback epsilon seems too large (result_max={result_max})."
        )
