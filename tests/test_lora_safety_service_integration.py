# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for LoRA Safety Service per-direction integration."""

import math
from types import SimpleNamespace

import pytest

from modelcypher.core.domain._backend import get_default_backend


@pytest.fixture
def backend():
    return get_default_backend()

class TestLoRASafetyPerDirection:
    """Tests verify_per_direction_bounds in LoRASafetyService."""

    def test_safe_adapter_passes(self, backend):
        """A safe adapter (small magnitude) passes verification."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        m, n, rank = 64, 32, 4
        W = backend.random_normal((m, n))
        A = backend.random_normal((rank, n)) * 0.01
        B = backend.random_normal((m, rank)) * 0.01
        backend.eval(W, A, B)

        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)

        assert result.is_safe
        assert result.max_ratio < 1.0

    def test_unsafe_adapter_fails(self, backend):
        """An unsafe adapter (large magnitude) fails verification."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        m, n, rank = 64, 32, 4

        # Create W with small singular values
        W = backend.random_normal((m, n)) * 0.1
        backend.eval(W)

        # Create huge LoRA delta
        A = backend.random_normal((rank, n)) * 10.0
        B = backend.random_normal((m, rank)) * 10.0
        backend.eval(A, B)

        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)

        assert not result.is_safe
        assert len(result.violations) > 0
        assert result.max_ratio > 1.0

    def test_orthogonal_adapter_passes(self, backend):
        """Adapter orthogonal to W passes (since it doesn't affect W's directions).

        Note: This tests the current diagonal-only check. A full-matrix check
        would flag off-diagonal interactions, but our current scope is protecting
        existing singular values.
        """
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        m, n, rank = 64, 32, 4

        # Create W with defined spectrum to separate signal from noise
        # m=64, n=32. Rank=4.
        # Signal: singular values [10, 10, 10, 10]
        # Noise: singular values [1e-8, ...]
        U_full, _, Vt_full = backend.svd(backend.random_normal((m, n)), compute_uv=True)

        # Construct S with gap
        S_vals = [10.0] * rank + [1e-8] * (n - rank)
        S = backend.array(S_vals)
        backend.zeros((m, n))
        # Place diagonal
        # This is a bit manual in MLX if we can't just assign diagonal.
        # Easier: W = U[:,:n] @ diag(S) @ Vt
        # But U is [m, m], Vt is [n, n], S is [n] (conceptually)

        # W = U[:, :n] @ (S * Vt)
        # S * Vt scales rows of Vt
        SVt = backend.reshape(S, (-1, 1)) * Vt_full
        W = backend.matmul(U_full[:, :n], SVt)
        backend.eval(W)

        U, S, Vt = backend.svd(W, compute_uv=True)
        U_k = U[:, :rank]
        backend.eval(U_k)

        # Create Delta orthogonal to BOTH U_k and V_k
        # Delta = (I - U_k U_k^T) @ Random @ (I - V_k V_k^T)

        # Get V_k from Vt (rows of Vt are v_j^T, so columns of V are v_j)
        V = backend.transpose(Vt)
        V_k = V[:, :rank]
        backend.eval(V_k)

        Rand = backend.random_normal((m, n))

        # Project out U components
        Proj_U = backend.matmul(U_k, backend.matmul(backend.transpose(U_k), Rand))
        Res_U = Rand - Proj_U

        # Project out V components (from the result)
        # Res_U @ V_k @ V_k^T
        Proj_V = backend.matmul(Res_U, backend.matmul(V_k, backend.transpose(V_k)))
        Delta = Res_U - Proj_V

        # Scale it up - it's large but orthogonal
        Delta = Delta * 100.0
        backend.eval(Delta)

        # Factorize Delta into B, A for the API
        # (Just use SVD to get factors)
        U_d, S_d, Vt_d = backend.svd(Delta, compute_uv=True)
        # B = U_d @ sqrt(S_d), A = sqrt(S_d) @ Vt_d
        S_sqrt = backend.sqrt(S_d[:rank])
        B = U_d[:, :rank] * S_sqrt
        A = Vt_d[:rank, :] * backend.reshape(S_sqrt, (-1, 1))
        backend.eval(B, A)

        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)

        # Should be safe because projection U^T @ Delta @ V is near zero on diagonal
        # (Diagonal entries measure alignment)

        assert result.is_safe, f"Orthogonal delta should be safe, max ratio: {result.max_ratio}"


class TestLoRASafetyQuantizedBase:
    """Tests quantized-base handling for spectral scale analysis."""

    @staticmethod
    def _build_model_with_down_proj(linear_module):
        layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=linear_module))
        return SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    def test_get_weight_for_svd_dequantizes_quantized_linear(self, backend, monkeypatch):
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()

        W_fp = backend.astype(backend.random_normal((64, 64)), "float32")
        backend.eval(W_fp)
        q_weight, q_scales, q_biases = backend.quantize(
            W_fp, group_size=64, bits=8, mode="affine",
        )
        backend.eval(q_weight, q_scales)
        if q_biases is not None:
            backend.eval(q_biases)

        quantized_linear = SimpleNamespace(
            weight=q_weight,
            scales=q_scales,
            biases=q_biases,
            group_size=64,
            bits=8,
        )
        model = self._build_model_with_down_proj(quantized_linear)

        calls = {"n": 0}
        original_dequantize = backend.dequantize

        def _wrapped_dequantize(weight, scales, biases, group_size, bits, mode):
            calls["n"] += 1
            return original_dequantize(
                weight,
                scales,
                biases=biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )

        monkeypatch.setattr(backend, "dequantize", _wrapped_dequantize)

        W_svd = service._get_weight_for_svd(
            model.model,
            "model.layers.0.mlp.down_proj",
            backend,
        )
        assert W_svd is not None
        assert calls["n"] >= 1
        assert tuple(int(x) for x in W_svd.shape) == (64, 64)

    def test_quantized_scale_bound_matches_full_precision(self, backend):
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()

        U_rand = backend.random_normal((64, 64))
        V_rand = backend.random_normal((64, 64))
        U, _, _ = backend.svd(U_rand, compute_uv=True)
        V, _, _ = backend.svd(V_rand, compute_uv=True)
        singular_vals = backend.linspace(10.0, 1.0, 64, dtype="float32")
        backend.eval(U, V, singular_vals)
        W_fp = backend.matmul(U, singular_vals[:, None] * backend.transpose(V))
        backend.eval(W_fp)

        q_weight, q_scales, q_biases = backend.quantize(
            W_fp, group_size=64, bits=8, mode="affine",
        )
        backend.eval(q_weight, q_scales)
        if q_biases is not None:
            backend.eval(q_biases)

        q_linear = SimpleNamespace(
            weight=q_weight,
            scales=q_scales,
            biases=q_biases,
            group_size=64,
            bits=8,
        )
        q_model = self._build_model_with_down_proj(q_linear).model

        rank = 8
        lora_a = backend.astype(backend.random_normal((rank, 64)), "float32")
        lora_b = backend.astype(backend.random_normal((64, rank)), "float32")
        backend.eval(lora_a, lora_b)

        def _sigma_k(weight):
            W_f32 = backend.astype(weight, "float32")
            backend.eval(W_f32)
            _, S, _ = backend.svd(W_f32, compute_uv=True)
            backend.eval(S)
            sigma_max = float(backend.to_scalar(S[0]))
            max_dim = max(int(W_f32.shape[0]), int(W_f32.shape[1]))
            eps_svd = float(backend.finfo(S.dtype).eps)
            threshold = float(max_dim) * eps_svd * sigma_max
            significant_mask = S > threshold
            eff_rank_arr = backend.sum(backend.astype(significant_mask, "int32"))
            backend.eval(eff_rank_arr)
            effective_rank = int(backend.to_scalar(eff_rank_arr))
            return (
                float(backend.to_scalar(S[effective_rank - 1]))
                if effective_rank > 0
                else float(backend.to_scalar(S[-1]))
            )

        D = backend.matmul(backend.transpose(lora_b), backend.transpose(lora_a))
        D_f32 = backend.astype(D, "float32")
        backend.eval(D_f32)
        _, S_D, _ = backend.svd(D_f32, compute_uv=True)
        backend.eval(S_D)
        delta_spectral = float(backend.to_scalar(S_D[0]))
        assert delta_spectral > 0.0

        W_q_for_svd = service._get_weight_for_svd(
            q_model,
            "model.layers.0.mlp.down_proj",
            backend,
        )
        assert W_q_for_svd is not None

        sigma_k_fp = _sigma_k(W_fp)
        scale_fp = sigma_k_fp / delta_spectral
        scale_q = _sigma_k(W_q_for_svd) / delta_spectral
        rel_diff = abs(scale_q - scale_fp) / max(abs(scale_fp), 1e-12)

        # Weyl: |sigma_k(W_q) - sigma_k(W)| <= ||W_q - W||_2.
        quantization_error = W_q_for_svd - W_fp
        _, error_singular_values, _ = backend.svd(
            quantization_error, compute_uv=True
        )
        backend.eval(error_singular_values)
        error_spectral = float(backend.to_scalar(error_singular_values[0]))
        weyl_relative_bound = error_spectral / sigma_k_fp
        roundoff = max(int(W_fp.shape[0]), int(W_fp.shape[1])) * float(
            backend.finfo(W_fp.dtype).eps
        )
        assert math.isfinite(rel_diff)
        assert rel_diff <= weyl_relative_bound + roundoff

    def test_get_weight_for_svd_returns_none_if_dequantization_fails(self, backend, monkeypatch):
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()
        W_fp = backend.astype(backend.random_normal((64, 64)), "float32")
        backend.eval(W_fp)
        q_weight, q_scales, q_biases = backend.quantize(
            W_fp, group_size=64, bits=8, mode="affine",
        )
        backend.eval(q_weight, q_scales)
        if q_biases is not None:
            backend.eval(q_biases)

        quantized_linear = SimpleNamespace(
            weight=q_weight,
            scales=q_scales,
            biases=q_biases,
            group_size=64,
            bits=8,
            mode="affine",
        )
        model = self._build_model_with_down_proj(quantized_linear)

        def _raise_dequantize(*_args, **_kwargs):
            raise RuntimeError("dequantize failed")

        monkeypatch.setattr(backend, "dequantize", _raise_dequantize)

        W_svd = service._get_weight_for_svd(
            model.model,
            "model.layers.0.mlp.down_proj",
            backend,
        )
        assert W_svd is None

    def test_set_base_weight_requantizes_quantized_module(self, backend):
        from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

        service = LoRASafetyService()

        W_fp = backend.astype(backend.random_normal((64, 64)), "float32")
        backend.eval(W_fp)
        q_weight, q_scales, q_biases = backend.quantize(
            W_fp, group_size=64, bits=8, mode="affine",
        )
        backend.eval(q_weight, q_scales)
        if q_biases is not None:
            backend.eval(q_biases)

        quantized_linear = SimpleNamespace(
            weight=q_weight,
            scales=q_scales,
            biases=q_biases,
            group_size=64,
            bits=8,
            mode="affine",
        )
        model = self._build_model_with_down_proj(quantized_linear)
        new_weight = W_fp * 0.9
        backend.eval(new_weight)

        service._set_base_weight(
            model.model,
            "model.layers.0.mlp.down_proj",
            new_weight,
            backend=backend,
        )

        restored = service._get_weight_for_svd(
            model.model,
            "model.layers.0.mlp.down_proj",
            backend,
        )
        assert restored is not None
        diff = backend.norm(backend.astype(restored, "float32") - backend.astype(new_weight, "float32"))
        base_norm = backend.norm(backend.astype(new_weight, "float32"))
        backend.eval(diff, base_norm)
        rel_err = float(backend.to_scalar(diff)) / max(float(backend.to_scalar(base_norm)), 1e-12)

        scales_f32 = backend.astype(quantized_linear.scales, "float32")
        backend.eval(scales_f32)
        max_scale = float(backend.to_scalar(backend.max(scales_f32)))
        m, n = int(new_weight.shape[0]), int(new_weight.shape[1])
        # Affine quantization bound: |e_ij| <= scale_group/2, so
        # ||E||_F <= sqrt(m*n) * max_scale / 2.
        abs_err_bound = 0.5 * max_scale * math.sqrt(float(m * n))
        denom = max(float(backend.to_scalar(base_norm)), 1e-12)
        tol = (abs_err_bound / denom) + float(sqrt_scalar(backend.finfo().eps, backend))
        assert math.isfinite(rel_err)
        assert rel_err <= tol
