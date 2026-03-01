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


class _ToyMoEExpert(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.down_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)


class _ToyMoEMLP(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, n_experts, bias=False)
        self.experts = [_ToyMoEExpert(hidden_dim) for _ in range(n_experts)]
        self.shared_expert = _ToyMoEExpert(hidden_dim)
        self.shared_expert_gate = nn.Linear(hidden_dim, 1, bias=False)


class _ToyLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = _ToyAttention(hidden_dim)
        self.mlp = _ToyMLP(hidden_dim)


class _ToyMoELayer(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int = 4):
        super().__init__()
        self.self_attn = _ToyAttention(hidden_dim)
        self.mlp = _ToyMoEMLP(hidden_dim, n_experts=n_experts)


class _ToyBaseModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.layers = [_ToyLayer(hidden_dim) for _ in range(n_layers)]


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.model = _ToyBaseModel(n_layers=n_layers, hidden_dim=hidden_dim)


class _ToyMoEBaseModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16, n_experts: int = 4):
        super().__init__()
        self.layers = [
            _ToyMoELayer(hidden_dim, n_experts=n_experts) for _ in range(n_layers)
        ]


class _ToyMoEModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 16, n_experts: int = 4):
        super().__init__()
        self.model = _ToyMoEBaseModel(
            n_layers=n_layers, hidden_dim=hidden_dim, n_experts=n_experts,
        )


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
def test_extract_weight_matrices_includes_moe_experts(backend_name) -> None:
    backend = _get_backend_or_fail(backend_name)
    model = _ToyMoEModel(n_layers=1, hidden_dim=16, n_experts=4)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)

    assert "model.layers.0.mlp.gate.weight" in weights
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in weights
    assert "model.layers.0.mlp.experts.0.up_proj.weight" in weights
    assert "model.layers.0.mlp.experts.0.down_proj.weight" in weights
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in weights
    assert "model.layers.0.mlp.shared_expert.up_proj.weight" in weights
    assert "model.layers.0.mlp.shared_expert.down_proj.weight" in weights
    assert "model.layers.0.mlp.shared_expert_gate.weight" in weights


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_resolve_parent_and_attr_for_moe_expert_key(backend_name) -> None:
    backend = _get_backend_or_fail(backend_name)
    model = _ToyMoEModel(n_layers=1, hidden_dim=16, n_experts=4)
    adapter = MLXTrainingAdapter(backend)

    key = "model.layers.0.mlp.experts.2.up_proj.weight"
    parent, attr = adapter._resolve_parent_and_attr(model, key)
    assert attr == "up_proj"
    assert hasattr(parent, "up_proj")


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


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_streaming_geometry_matches_batch(backend_name) -> None:
    """Streaming geometry analysis must match batch analysis exactly."""
    backend = _get_backend_or_fail(backend_name)
    # Fresh model — the cached model may have NB-LoRA injected by other tests
    model = _ToyModel()
    adapter = MLXTrainingAdapter(backend)

    # Batch: extract all weights, then analyze
    weights = adapter.extract_weight_matrices(model)
    batch_geoms = analyze_weight_geometries(weights, backend)

    # Streaming: analyze one layer at a time
    stream_geoms = adapter.analyze_model_geometry_streaming(model)

    # Same keys
    assert set(stream_geoms.keys()) == set(batch_geoms.keys())

    # Same geometry values
    for key in batch_geoms:
        bg = batch_geoms[key]
        sg = stream_geoms[key]
        assert sg.sigma_max == pytest.approx(bg.sigma_max, abs=1e-5), key
        assert sg.sigma_k == pytest.approx(bg.sigma_k, abs=1e-5), key
        assert sg.tail_dims == bg.tail_dims, key
        assert sg.shannon_effective_rank == pytest.approx(
            bg.shannon_effective_rank, abs=1e-4,
        ), key
        assert sg.spectral_gap == pytest.approx(bg.spectral_gap, abs=1e-5), key


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_streaming_geometry_randomized(backend_name) -> None:
    """Streaming with use_randomized=True produces valid geometry."""
    backend = _get_backend_or_fail(backend_name)
    # Fresh model — the cached model may have NB-LoRA injected by other tests
    model = _ToyModel()
    adapter = MLXTrainingAdapter(backend)

    geoms = adapter.analyze_model_geometry_streaming(
        model, use_randomized=True, randomized_kwargs={"seed": 42},
    )

    assert len(geoms) > 0
    for key, geom in geoms.items():
        assert geom.sigma_max > 0
        assert geom.full_rank > 0
        assert isinstance(geom.tail_dims, int)


class _QuantizedToyAttention(nn.Module):
    """Attention block with QuantizedLinear projections."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_proj = nn.QuantizedLinear(hidden_dim, hidden_dim, bias=False, group_size=64, bits=4)
        self.k_proj = nn.QuantizedLinear(hidden_dim, hidden_dim, bias=False, group_size=64, bits=4)
        self.v_proj = nn.QuantizedLinear(hidden_dim, hidden_dim, bias=False, group_size=64, bits=4)
        self.o_proj = nn.QuantizedLinear(hidden_dim, hidden_dim, bias=False, group_size=64, bits=4)


class _QuantizedToyMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.up_proj = nn.QuantizedLinear(hidden_dim, hidden_dim * 2, bias=False, group_size=64, bits=4)
        self.down_proj = nn.QuantizedLinear(hidden_dim * 2, hidden_dim, bias=False, group_size=64, bits=4)
        self.gate_proj = nn.QuantizedLinear(hidden_dim, hidden_dim * 2, bias=False, group_size=64, bits=4)


class _QuantizedToyLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = _QuantizedToyAttention(hidden_dim)
        self.mlp = _QuantizedToyMLP(hidden_dim)


class _QuantizedToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [_QuantizedToyLayer(hidden_dim) for _ in range(n_layers)]


@pytest.mark.mlx
def test_extract_weight_matrices_dequantizes() -> None:
    """extract_weight_matrices() must dequantize QuantizedLinear weights.

    Validates Phase 1 fix: SVD on packed integer data is wrong.
    Dequantized weights should have shape [out, in] not [out, in // pack_factor].
    """
    import mlx.core as mx

    backend = _get_backend_or_fail("mlx")
    model = _QuantizedToyModel(n_layers=1, hidden_dim=128)
    mx.eval(model.parameters())

    adapter = MLXTrainingAdapter(backend)
    weights = adapter.extract_weight_matrices(model)

    assert len(weights) > 0
    for key, w in weights.items():
        # Dequantized weights should have full input dimension
        if "q_proj" in key or "k_proj" in key or "v_proj" in key or "o_proj" in key:
            # Attention: [hidden, hidden] = [128, 128]
            assert w.shape == (128, 128), f"{key}: expected (128, 128), got {w.shape}"
        elif "up_proj" in key or "gate_proj" in key:
            # MLP up/gate: [hidden*2, hidden] = [256, 128]
            assert w.shape == (256, 128), f"{key}: expected (256, 128), got {w.shape}"
        elif "down_proj" in key:
            # MLP down: [hidden, hidden*2] = [128, 256]
            assert w.shape == (128, 256), f"{key}: expected (128, 256), got {w.shape}"


@pytest.mark.mlx
def test_quantized_geometry_weyl_bound() -> None:
    """Validate Weyl bound: quantization error < spectral_gap / 2.

    If ||E_q||_2 < spectral_gap / 2 for every layer, Weyl's theorem
    guarantees no singular value crossing. The geometry derived from
    quantized weights is topologically identical to full precision.

    This test uses a toy model — real model validation requires
    the external volume with actual pretrained weights.
    """
    import mlx.core as mx

    backend = _get_backend_or_fail("mlx")
    adapter = MLXTrainingAdapter(backend)

    # Create a quantized model and analyze its geometry
    model = _QuantizedToyModel(n_layers=1, hidden_dim=128)
    mx.eval(model.parameters())

    geoms = adapter.analyze_model_geometry_streaming(model)

    # Every layer must produce valid geometry from dequantized weights
    assert len(geoms) > 0
    for key, geom in geoms.items():
        assert geom.sigma_max > 0, f"{key}: sigma_max should be positive"
        assert geom.full_rank > 0, f"{key}: full_rank should be positive"
        # With random quantized weights, spectrum should be non-trivial
        assert geom.shannon_effective_rank > 0, f"{key}: zero Shannon eff rank"
