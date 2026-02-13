# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.training.geometric_optimizer import (
    LayerOptimizerConfig,
    OptimizerGeometryConfig,
)
from modelcypher.core.domain.training.scaled_gd import precondition_lora_gradients
from modelcypher.ports.training import LoRALayerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lora_config(
    key: str = "model.layers.0.self_attn.q_proj.weight",
    rank: int = 4,
    in_features: int = 8,
    out_features: int = 8,
) -> LoRALayerConfig:
    return LoRALayerConfig(
        layer_key=key,
        rank=rank,
        sigma_k=0.5,
        in_features=in_features,
        out_features=out_features,
    )


def _make_opt_config(
    key: str = "model.layers.0.self_attn.q_proj.weight",
    epsilon: float = 1e-4,
    decay: float = 0.0,
) -> OptimizerGeometryConfig:
    layer_cfg = LayerOptimizerConfig(
        layer_key=key,
        sigma_max=1.0,
        sigma_k=0.5,
        lr_scale=1.0,
        epsilon=epsilon,
        decay_scale=decay,
        spectral_gap=0.1,
    )
    return OptimizerGeometryConfig(
        base_lr=1e-4,
        max_sigma=1.0,
        layer_configs={key: layer_cfg},
    )


class TestPreconditionLoraGradients:
    """Tests for precondition_lora_gradients() — ScaledGD preconditioning."""

    def test_rank_zero_skipped(self, any_backend):
        """Rank-0 layer is skipped, output = input gradients."""
        b = any_backend
        cfg = _make_lora_config(rank=0)
        opt = _make_opt_config()

        grad = {"some.key": b.eye(4)}
        param = {"some.key": b.eye(4)}
        b.eval(grad["some.key"], param["some.key"])

        result = precondition_lora_gradients(grad, param, [cfg], opt, b)
        # Input copied through unchanged
        assert "some.key" in result

    def test_missing_param_keys_skipped(self, any_backend):
        """Missing lora_a/lora_b keys → no error, gradients copied."""
        b = any_backend
        cfg = _make_lora_config()
        opt = _make_opt_config()

        grad = {"unrelated.key": b.eye(4)}
        param = {"unrelated.key": b.eye(4)}
        b.eval(grad["unrelated.key"], param["unrelated.key"])

        result = precondition_lora_gradients(grad, param, [cfg], opt, b)
        assert "unrelated.key" in result

    def test_identity_a_b_preserves_grad(self, any_backend):
        """A=I, B=I → preconditioner ~ (I + eI)^{-1}, grad preserved up to epsilon."""
        b = any_backend
        rank = 4
        key = "model.layers.0.self_attn.q_proj.weight"
        cfg = _make_lora_config(key=key, rank=rank, in_features=rank, out_features=rank)
        opt = _make_opt_config(key=key, epsilon=1e-6)

        prefix = key.replace(".weight", "")
        a_key = prefix + ".lora_a"
        b_key = prefix + ".lora_b"

        A = b.eye(rank)
        B = b.eye(rank)
        grad_a = b.eye(rank)
        grad_b = b.eye(rank)
        b.eval(A, B, grad_a, grad_b)

        param = {a_key: A, b_key: B}
        grad = {a_key: grad_a, b_key: grad_b}

        result = precondition_lora_gradients(grad, param, [cfg], opt, b)

        # (I * I^T + eI)^{-1} = (1+e)^{-1} * I → grad ~ (1+e)^{-1} * I
        b.eval(result[a_key], result[b_key])
        # Should be close to identity scaled by ~1/(1+eps)
        for i in range(rank):
            val_a = float(b.to_scalar(result[a_key][i][i]))
            val_b = float(b.to_scalar(result[b_key][i][i]))
            assert val_a == pytest.approx(1.0 / (1.0 + 1e-6), abs=1e-4)
            assert val_b == pytest.approx(1.0 / (1.0 + 1e-6), abs=1e-4)

    def test_output_dimensions_match_input(self, any_backend):
        """Preconditioned gradients have same shapes as input."""
        b = any_backend
        rank = 4
        in_feat, out_feat = 8, 8
        key = "model.layers.0.self_attn.q_proj.weight"
        cfg = _make_lora_config(key=key, rank=rank, in_features=in_feat, out_features=out_feat)
        opt = _make_opt_config(key=key)

        prefix = key.replace(".weight", "")
        a_key = prefix + ".lora_a"
        b_key = prefix + ".lora_b"

        A = b.array([[0.1] * rank] * in_feat)
        B = b.array([[0.1] * out_feat] * rank)
        grad_a = b.array([[0.01] * rank] * in_feat)
        grad_b = b.array([[0.01] * out_feat] * rank)
        b.eval(A, B, grad_a, grad_b)

        param = {a_key: A, b_key: B}
        grad = {a_key: grad_a, b_key: grad_b}

        result = precondition_lora_gradients(grad, param, [cfg], opt, b)

        assert result[a_key].shape == grad_a.shape
        assert result[b_key].shape == grad_b.shape

    def test_weight_decay_adds_contribution(self, any_backend):
        """decay > 0 → gradient includes decay term."""
        b = any_backend
        rank = 2
        key = "model.layers.0.self_attn.q_proj.weight"
        cfg = _make_lora_config(key=key, rank=rank, in_features=rank, out_features=rank)
        opt_no_decay = _make_opt_config(key=key, decay=0.0, epsilon=1e-6)
        opt_with_decay = _make_opt_config(key=key, decay=0.1, epsilon=1e-6)

        prefix = key.replace(".weight", "")
        a_key = prefix + ".lora_a"
        b_key = prefix + ".lora_b"

        A = b.eye(rank)
        B = b.eye(rank)
        grad_a = b.eye(rank)
        grad_b = b.eye(rank)
        b.eval(A, B, grad_a, grad_b)

        param = {a_key: A, b_key: B}
        grad = {a_key: grad_a, b_key: grad_b}

        r_no = precondition_lora_gradients(dict(grad), dict(param), [cfg], opt_no_decay, b)
        r_yes = precondition_lora_gradients(dict(grad), dict(param), [cfg], opt_with_decay, b)
        b.eval(r_no[a_key], r_yes[a_key])

        # With decay, the gradient should be larger (adds decay * param)
        no_norm = float(b.to_scalar(b.norm(r_no[a_key])))
        yes_norm = float(b.to_scalar(b.norm(r_yes[a_key])))
        assert yes_norm > no_norm

    def test_input_dict_not_mutated(self, any_backend):
        """Precondition should not mutate the input gradient dict."""
        b = any_backend
        rank = 2
        key = "model.layers.0.self_attn.q_proj.weight"
        cfg = _make_lora_config(key=key, rank=rank, in_features=rank, out_features=rank)
        opt = _make_opt_config(key=key)

        prefix = key.replace(".weight", "")
        a_key = prefix + ".lora_a"
        b_key = prefix + ".lora_b"

        A = b.eye(rank)
        B = b.eye(rank)
        grad_a = b.eye(rank)
        grad_b = b.eye(rank)
        b.eval(A, B, grad_a, grad_b)

        param = {a_key: A, b_key: B}
        grad = {a_key: grad_a, b_key: grad_b}
        original_keys = set(grad.keys())

        precondition_lora_gradients(grad, param, [cfg], opt, b)

        # Keys should be unchanged (dict was copied internally)
        assert set(grad.keys()) == original_keys
