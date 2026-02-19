# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

import pytest

from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter

MODEL_PATH = Path("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not available")

_MODEL_CACHE: dict[str, tuple[object, object, object]] = {}


def _get_backend_or_skip(backend_name: str):
    from modelcypher.backends import get_backend

    try:
        return get_backend(backend_name)
    except Exception as exc:
        pytest.skip(f"{backend_name} backend unavailable: {exc}")


def _load_model_and_tokenizer(backend_name: str) -> tuple[object, object, object]:
    cached = _MODEL_CACHE.get(backend_name)
    if cached is not None:
        return cached

    backend = _get_backend_or_skip(backend_name)
    model, tokenizer = backend.load_model(str(MODEL_PATH))
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
