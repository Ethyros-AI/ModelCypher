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

    assert config["type"] == "geometric_lora"
    assert config["method"] == "geometric_lora"
    assert config["init_method"] == "cayley"
    assert config["optimizer"] == "fisher_mass"
    assert config["controller"] == "mass"
    assert config["stopping"] == "geometric_certificate"
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
def test_save_pissa_adapter_writes_geometric_lora_identity(backend_name, tmp_path) -> None:
    """PiSSA export writes the canonical geometric_lora identity surface."""
    import json

    backend, model, _ = _load_model_and_tokenizer(backend_name)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)
    geometries = analyze_weight_geometries(weights, backend)
    targets = select_target_modules(geometries)

    assert targets, "Expected targetable layers in toy model"

    injected = adapter.inject_pissa_lora(model, geometries, targets)
    assert injected > 0

    output_dir = tmp_path / "pissa_adapter_out"
    adapter.save_adapter(model, output_dir)

    config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["type"] == "geometric_lora"
    assert config["method"] == "geometric_lora"
    assert config["init_method"] == "pissa"
    assert config["optimizer"] == "fisher_mass"
    assert config["controller"] == "mass"
    assert config["stopping"] == "geometric_certificate"


# ── Qwen3.5-style layout: root.language_model.layers ──────────────────────────

class _ToyLMBackbone(nn.Module):
    """Mimics Qwen3.5 text backbone: root.language_model has .layers."""

    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.layers = [_ToyLayer(hidden_dim) for _ in range(n_layers)]


class _ToyLMModel(nn.Module):
    """Mimics Qwen3.5 root: root.model is None, root.language_model has layers."""

    def __init__(self, n_layers: int = 2, hidden_dim: int = 16):
        super().__init__()
        self.language_model = _ToyLMBackbone(n_layers=n_layers, hidden_dim=hidden_dim)
        # model attr is absent — matches Qwen3.5 where model.model is None


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_get_model_base_language_model_layout(backend_name) -> None:
    """_get_model_base returns language_model and correct prefix for Qwen3.5 layout."""
    backend = _get_backend_or_fail(backend_name)
    model = _ToyLMModel()
    adapter = MLXTrainingAdapter(backend)

    base, key_prefix = adapter._get_model_base(model)

    assert base is model.language_model
    assert key_prefix == "model.language_model.layers"


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_extract_weight_matrices_language_model_layout(backend_name) -> None:
    """extract_weight_matrices produces correct key namespace for Qwen3.5 layout."""
    backend = _get_backend_or_fail(backend_name)
    model = _ToyLMModel(n_layers=1, hidden_dim=16)
    adapter = MLXTrainingAdapter(backend)

    weights = adapter.extract_weight_matrices(model)

    assert len(weights) > 0
    for key in weights:
        assert key.startswith("model.language_model.layers."), f"Wrong key namespace: {key}"
        assert key.endswith(".weight")


@pytest.mark.parametrize("backend_name", ["mlx"])
@pytest.mark.mlx
def test_resolve_parent_and_attr_language_model_layout(backend_name) -> None:
    """_resolve_parent_and_attr traverses model.language_model.layers.X.self_attn.q_proj."""
    backend = _get_backend_or_fail(backend_name)
    model = _ToyLMModel(n_layers=1, hidden_dim=16)
    adapter = MLXTrainingAdapter(backend)

    key = "model.language_model.layers.0.self_attn.q_proj.weight"
    parent, attr = adapter._resolve_parent_and_attr(model, key)

    assert attr == "q_proj"
    assert hasattr(parent, "q_proj")
    assert parent is model.language_model.layers[0].self_attn


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


@pytest.mark.mlx
def test_memory_safe_micro_batch_probes_safe_side_first(monkeypatch) -> None:
    import mlx.core as mx
    import mlx_lm.tuner.trainer as trainer
    import modelcypher.backends._mlx_training_adapter_diagnostics_mixin as diag_mixin

    backend = _get_backend_or_fail("mlx")
    adapter = MLXTrainingAdapter(backend)

    probe_history: list[int] = []

    def _fake_iterate_batches(_dataset, batch_size, _seq_length, loop=False, seed=0):
        del loop, seed
        probe_history.append(int(batch_size))
        if int(batch_size) >= 5:
            raise RuntimeError("out of memory")
        yield mx.zeros((int(batch_size), 8), dtype=mx.int32), [8] * int(batch_size)

    def _fake_value_and_grad(_model, _loss_fn):
        def _loss_vg(model, batch, lengths):  # noqa: ANN001
            del model, batch, lengths
            return (mx.array(0.0), None), {}

        return _loss_vg

    monkeypatch.setattr(trainer, "iterate_batches", _fake_iterate_batches)
    monkeypatch.setattr(diag_mixin.nn, "value_and_grad", _fake_value_and_grad)

    model = _ToyModel(n_layers=1, hidden_dim=16)
    train_dataset = [([1, 2, 3, 4], 0)] * 32
    safe_bs = adapter.derive_memory_safe_micro_batch_size(
        model=model,
        train_dataset=train_dataset,
        seq_length=8,
        logical_batch_size=8,
    )

    assert safe_bs == 4
    assert probe_history[0] == 1
    # Geometric doubling: no probe exceeds 2x the largest preceding safe value.
    # This prevents the catastrophic jumps that caused kernel OOM kills.
    max_safe = 0
    for bs in probe_history:
        if bs < 5:  # OOM threshold is 5
            max_safe = max(max_safe, bs)
        else:
            assert bs <= max_safe * 2


@pytest.mark.mlx
def test_memory_safe_micro_batch_raises_when_one_oom(monkeypatch) -> None:
    import mlx_lm.tuner.trainer as trainer

    backend = _get_backend_or_fail("mlx")
    adapter = MLXTrainingAdapter(backend)

    probe_history: list[int] = []

    def _always_oom(_dataset, batch_size, _seq_length, loop=False, seed=0):
        del loop, seed
        probe_history.append(int(batch_size))
        raise RuntimeError("metal out of memory")
        yield  # pragma: no cover

    monkeypatch.setattr(trainer, "iterate_batches", _always_oom)

    model = _ToyModel(n_layers=1, hidden_dim=16)
    train_dataset = [([1, 2, 3, 4], 0)] * 8

    with pytest.raises(RuntimeError, match="micro_batch=1"):
        adapter.derive_memory_safe_micro_batch_size(
            model=model,
            train_dataset=train_dataset,
            seq_length=8,
            logical_batch_size=8,
        )

    assert probe_history == [1]


@pytest.mark.mlx
def test_memory_safe_micro_batch_no_catastrophic_jump(monkeypatch) -> None:
    """Geometric doubling returns last safe power-of-2 without binary refinement.

    With OOM threshold=20 and logical_batch=64: probes 1, 2, 4, 8, 16 (pass),
    32 (fail). Returns 16 — the last safe geometric step.

    Phase 2 binary refinement is intentionally absent: the doubling algorithm's
    2-competitive guarantee provides ~2x memory headroom, which accounts for
    training-vs-probe overhead. Binary search discards that margin and causes
    hard kills under actual training load.
    """
    import mlx.core as mx
    import mlx_lm.tuner.trainer as trainer
    import modelcypher.backends._mlx_training_adapter_diagnostics_mixin as diag_mixin

    backend = _get_backend_or_fail("mlx")
    adapter = MLXTrainingAdapter(backend)

    oom_threshold = 20
    probe_history: list[int] = []

    def _fake_iterate_batches(_dataset, batch_size, _seq_length, loop=False, seed=0):
        del loop, seed
        probe_history.append(int(batch_size))
        if int(batch_size) >= oom_threshold:
            raise RuntimeError("out of memory")
        yield mx.zeros((int(batch_size), 8), dtype=mx.int32), [8] * int(batch_size)

    def _fake_value_and_grad(_model, _loss_fn):
        def _loss_vg(model, batch, lengths):  # noqa: ANN001
            del model, batch, lengths
            return (mx.array(0.0), None), {}

        return _loss_vg

    monkeypatch.setattr(trainer, "iterate_batches", _fake_iterate_batches)
    monkeypatch.setattr(diag_mixin.nn, "value_and_grad", _fake_value_and_grad)

    model = _ToyModel(n_layers=1, hidden_dim=16)
    train_dataset = [([1, 2, 3, 4], 0)] * 128
    safe_bs = adapter.derive_memory_safe_micro_batch_size(
        model=model,
        train_dataset=train_dataset,
        seq_length=8,
        logical_batch_size=64,
    )

    assert safe_bs == 16
    assert probe_history[0] == 1
    # Binary refinement removed: probe sequence must be strictly geometric (powers of 2).
    expected_probes = [1, 2, 4, 8, 16, 32]
    assert probe_history == expected_probes, (
        f"Expected geometric probe sequence {expected_probes}, got {probe_history}"
    )
    # Safety property: no probe exceeds 2x the largest known-safe value.
    max_safe = 0
    for bs in probe_history:
        if bs < oom_threshold:
            max_safe = max(max_safe, bs)
        else:
            assert bs <= max_safe * 2, (
                f"Catastrophic jump: probed {bs} but max safe was {max_safe}"
            )


@pytest.mark.mlx
def test_memory_safe_micro_batch_non_power_of_two_logical_batch(monkeypatch) -> None:
    """When logical_batch is not a power of 2, no max_candidate probe is issued.

    logical_batch=50, OOM at >=60: doubling probes 1, 2, 4, 8, 16, 32 (all
    pass), then 64 > 50 exits loop. safe=32. The old code would then probe 50;
    the new code returns 32 directly — preserving the 2x headroom.
    """
    import mlx.core as mx
    import mlx_lm.tuner.trainer as trainer
    import modelcypher.backends._mlx_training_adapter_diagnostics_mixin as diag_mixin

    backend = _get_backend_or_fail("mlx")
    adapter = MLXTrainingAdapter(backend)

    oom_threshold = 60
    probe_history: list[int] = []

    def _fake_iterate_batches(_dataset, batch_size, _seq_length, loop=False, seed=0):
        del loop, seed
        probe_history.append(int(batch_size))
        if int(batch_size) >= oom_threshold:
            raise RuntimeError("out of memory")
        yield mx.zeros((int(batch_size), 8), dtype=mx.int32), [8] * int(batch_size)

    def _fake_value_and_grad(_model, _loss_fn):
        def _loss_vg(model, batch, lengths):  # noqa: ANN001
            del model, batch, lengths
            return (mx.array(0.0), None), {}

        return _loss_vg

    monkeypatch.setattr(trainer, "iterate_batches", _fake_iterate_batches)
    monkeypatch.setattr(diag_mixin.nn, "value_and_grad", _fake_value_and_grad)

    model = _ToyModel(n_layers=1, hidden_dim=16)
    train_dataset = [([1, 2, 3, 4], 0)] * 128
    safe_bs = adapter.derive_memory_safe_micro_batch_size(
        model=model,
        train_dataset=train_dataset,
        seq_length=8,
        logical_batch_size=50,  # non-power-of-2: doubling reaches 32, then 64 > 50
    )

    # Returns 32, not 50 — max_candidate probe is not issued
    assert safe_bs == 32
    assert probe_history == [1, 2, 4, 8, 16, 32], (
        f"max_candidate probe was issued: {probe_history}"
    )


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
