# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import mlx.nn as nn
import pytest

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    select_target_modules,
)

_MODEL_CACHE: dict[str, tuple[object, object, object]] = {}


def _get_backend_or_fail(backend_name: str):
    from modelcypher.backends import get_backend

    try:
        return get_backend(backend_name)
    except Exception as exc:
        pytest.fail(f"{backend_name} backend unavailable: {exc}")


class _ToyTokenizer:
    def encode(self, text: str) -> list[int]:
        words = [w for w in text.split() if w]
        return [i + 1 for i in range(len(words))]


class _ToyAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)


class _ToyMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = _ToyAttention(hidden_dim)
        self.mlp = _ToyMLP(hidden_dim)


class _ToyBaseModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.layers = [_ToyLayer(hidden_dim) for _ in range(n_layers)]


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.model = _ToyBaseModel(n_layers=n_layers, hidden_dim=hidden_dim)


def _load_model_and_tokenizer(backend_name: str) -> tuple[object, object, object]:
    cached = _MODEL_CACHE.get(backend_name)
    if cached is not None:
        return cached

    backend = _get_backend_or_fail(backend_name)
    model = _ToyModel()
    tokenizer = _ToyTokenizer()
    payload = (backend, model, tokenizer)
    _MODEL_CACHE[backend_name] = payload
    return payload


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_extract_weight_matrices(backend_name) -> None:
    backend, model, _ = _load_model_and_tokenizer(backend_name)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)

    assert len(weights) > 0
    for key in weights:
        assert key.startswith("model.layers.")
        assert key.endswith(".weight")


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_prepare_dataset(backend_name) -> None:
    backend, _, tokenizer = _load_model_and_tokenizer(backend_name)
    adapter = MLXTrainingAdapter(backend)

    samples = [{"text": "The quick brown fox"}, {"text": "jumps over"}]
    dataset = adapter.prepare_dataset(samples, tokenizer)

    assert len(dataset) == 2
    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 2
    assert dataset[0][1] == 0


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_inject_nb_lora_rank_override_clamps(backend_name) -> None:
    """rank_overrides > tail_dims is clamped; rank_overrides <= 0 is skipped."""
    backend, model, _ = _load_model_and_tokenizer(backend_name)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)
    geometries = analyze_weight_geometries(weights, backend)
    targets = select_target_modules(geometries)

    assert targets, "Expected targetable layers in toy model"

    # Pick one target and create overrides: one oversized, one zero
    first = targets[0]
    second = targets[1] if len(targets) > 1 else None

    overrides = {first: 999_999}  # way over tail_dims
    if second:
        overrides[second] = 0  # non-positive → should be skipped

    n_injected = adapter.inject_nb_lora(
        model, geometries, targets,
        rank_overrides=overrides,
    )

    # At least some layers should inject (the ones without overrides, plus the clamped one)
    assert n_injected >= 1

    # Verify the clamped layer got rank == tail_dims (not 999_999)
    from modelcypher.backends.mlx_training_adapter import NBLoRALinear

    parent, attr = adapter._resolve_parent_and_attr(model, first)
    nb_module = getattr(parent, attr)
    assert isinstance(nb_module, NBLoRALinear)
    actual_rank = int(nb_module.A_tilde.shape[0])
    expected_rank = geometries[first].tail_dims
    assert actual_rank == expected_rank


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_save_adapter_per_layer_ranks(backend_name, tmp_path) -> None:
    """adapter_config.json includes per_layer_ranks with full weight keys."""
    import json

    backend, model, _ = _load_model_and_tokenizer(backend_name)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)
    geometries = analyze_weight_geometries(weights, backend)
    targets = select_target_modules(geometries)

    assert targets, "Expected targetable layers in toy model"

    adapter.inject_nb_lora(model, geometries, targets)

    output_dir = tmp_path / "adapter_out"
    adapter.save_adapter(model, output_dir)

    config_path = output_dir / "adapter_config.json"
    assert config_path.exists()

    with config_path.open() as f:
        config = json.load(f)

    assert "per_layer_ranks" in config
    plr = config["per_layer_ranks"]
    assert isinstance(plr, dict)
    assert len(plr) > 0

    # Keys should be full weight keys (matching geometry namespace)
    for key, rank in plr.items():
        assert key.startswith("model.layers."), f"Bad key namespace: {key}"
        assert key.endswith(".weight"), f"Key missing .weight suffix: {key}"
        assert isinstance(rank, int)
        assert rank > 0
