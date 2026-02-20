# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter, NBLoRALinear
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


class _Node:
    """Minimal attribute container for synthetic model trees."""


def _make_single_nb_lora_model() -> tuple[object, NBLoRALinear]:
    """Build minimal model tree expected by _iter_nb_lora_modules()."""
    model = _Node()
    model.model = _Node()

    layer = _Node()
    attn = _Node()
    attn.q_proj = NBLoRALinear(
        in_features=8,
        out_features=8,
        rank=4,
        scale_bound=1.0,
    )
    layer.self_attn = attn
    layer.mlp = None

    model.model.layers = [layer]
    return model, attn.q_proj


@pytest.mark.mlx
def test_apply_cayley_preconditioner_emits_ipz_diagnostics(mlx_backend) -> None:
    """I+Z conditioning diagnostics are emitted and internally consistent."""
    adapter = MLXTrainingAdapter(mlx_backend)
    model, nb_lora = _make_single_nb_lora_model()

    a_key = "model.layers.0.self_attn.q_proj.weight.A_tilde"
    b_key = "model.layers.0.self_attn.q_proj.weight.B_tilde"
    grad = {
        a_key: mx.ones_like(nb_lora.A_tilde),
        b_key: mx.ones_like(nb_lora.B_tilde),
    }
    mx.eval(grad[a_key], grad[b_key])

    precond_grad, metrics = adapter._apply_cayley_preconditioner(model, grad)
    precond_flat = dict(tree_flatten(precond_grad))

    assert a_key in precond_flat
    assert b_key in precond_flat
    assert tuple(precond_flat[a_key].shape) == tuple(nb_lora.A_tilde.shape)
    assert tuple(precond_flat[b_key].shape) == tuple(nb_lora.B_tilde.shape)

    required_keys = (
        "precond_ipz_kappa_upper_max",
        "precond_ipz_rel_error_upper_max",
        "precond_ipz_warn_fraction",
        "precond_ipz_warn_any",
    )
    for key in required_keys:
        assert key in metrics

    kappa_upper = metrics["precond_ipz_kappa_upper_max"]
    rel_error_upper = metrics["precond_ipz_rel_error_upper_max"]
    warn_fraction = metrics["precond_ipz_warn_fraction"]
    warn_any = metrics["precond_ipz_warn_any"]

    assert kappa_upper > 0.0
    assert rel_error_upper >= 0.0
    assert 0.0 <= warn_fraction <= 1.0
    assert warn_any in (0.0, 1.0)
    assert (warn_any == 1.0) == (warn_fraction > 0.0)

    eps = float(machine_epsilon(mlx_backend, mx.array([1.0])))
    tol = math.sqrt(eps)

    # One layer in this synthetic model, so max diagnostics should match
    # the analytic κ*eps relation used in the implementation.
    assert rel_error_upper == pytest.approx(kappa_upper * eps, rel=1e-6)

    expected_warn = 1.0 if rel_error_upper >= tol else 0.0
    assert warn_any == expected_warn
