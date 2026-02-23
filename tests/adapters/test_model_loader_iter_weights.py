# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import json

from modelcypher.adapters.model_loader import ModelLoader


class _FakeBackend:
    def __init__(self, shards: dict[str, dict[str, object]]) -> None:
        self._shards = shards

    def load_safetensors(self, path: str):
        return dict(self._shards[path])

    def eval(self, *args):
        return None

    def load_model(self, model_path: str, adapter_path: str | None = None):
        return object(), object()

    def generate(self, model, tokenizer, prompt: str, max_tokens: int = 512, **kwargs):
        return prompt


def test_iter_weights_is_deterministic_for_shards(tmp_path) -> None:
    shard_a = (tmp_path / "a.safetensors").resolve()
    shard_b = (tmp_path / "b.safetensors").resolve()
    shard_a.touch()
    shard_b.touch()

    backend = _FakeBackend(
        {
            str(shard_a): {
                "layer.1.weight": "w1",
                "layer.0.weight": "w0",
            },
            str(shard_b): {
                "layer.3.weight": "w3",
                "layer.2.weight": "w2",
            },
        },
    )
    loader = ModelLoader(backend=backend)

    items = list(loader.iter_weights(str(tmp_path)))
    names = [name for name, _ in items]
    values = [value for _, value in items]

    assert names == [
        "layer.0.weight",
        "layer.1.weight",
        "layer.2.weight",
        "layer.3.weight",
    ]
    assert values == ["w0", "w1", "w2", "w3"]


def test_iter_weights_uses_index_manifest_when_shards_not_globbed(tmp_path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_1 = shard_dir / "model-00001-of-00002.safetensors"
    shard_2 = shard_dir / "model-00002-of-00002.safetensors"
    shard_1.touch()
    shard_2.touch()

    index_payload = {
        "weight_map": {
            "layer.2.weight": f"{shard_dir.name}/{shard_2.name}",
            "layer.0.weight": f"{shard_dir.name}/{shard_1.name}",
            "layer.1.weight": f"{shard_dir.name}/{shard_1.name}",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index_payload),
        encoding="utf-8",
    )

    backend = _FakeBackend(
        {
            str(shard_1.resolve()): {
                "layer.1.weight": "w1",
                "layer.0.weight": "w0",
            },
            str(shard_2.resolve()): {
                "layer.2.weight": "w2",
            },
        },
    )
    loader = ModelLoader(backend=backend)

    items = list(loader.iter_weights(str(tmp_path)))
    names = [name for name, _ in items]

    assert names == ["layer.0.weight", "layer.1.weight", "layer.2.weight"]

    loaded = loader.load_weights(str(tmp_path))
    assert sorted(loaded.keys()) == names
