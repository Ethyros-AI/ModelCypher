# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import pytest

from lm_eval.api.model import LM
from mlx_lm.evaluate import MLXLM

from modelcypher.adapters.lm_eval_mlx_wrapper import MLXModelWrapper


def test_compat_patch_exposes_missing_transformers_attr():
    import transformers

    resolved = getattr(transformers, "AutoModelForVision2Seq")
    assert resolved.__name__ == "_DummyAutoModel"


def test_wrapper_is_lm_subclass():
    assert issubclass(MLXModelWrapper, MLXLM)
    assert issubclass(MLXModelWrapper, LM)


def test_wrapper_loads_adapter_when_provided(monkeypatch):
    init_calls: list[tuple[str, int]] = []
    adapter_calls: list[tuple[object, str]] = []

    def _fake_init(self, path_or_hf_repo: str, batch_size: int = 1):
        self._model = object()
        self.tokenizer = object()
        init_calls.append((path_or_hf_repo, batch_size))

    monkeypatch.setattr(MLXLM, "__init__", _fake_init)

    import mlx_lm.lora as lora

    monkeypatch.setattr(
        lora,
        "load_adapters",
        lambda model, adapter_path: adapter_calls.append((model, adapter_path)),
    )

    wrapper = MLXModelWrapper(
        model_path="/tmp/model",
        adapter_path="/tmp/adapter",
        batch_size=3,
    )

    assert init_calls == [("/tmp/model", 3)]
    assert adapter_calls == [(wrapper._model, "/tmp/adapter")]
    assert wrapper.model_path == "/tmp/model"
    assert wrapper.adapter_path == "/tmp/adapter"


def test_cleanup_clears_cache_without_error(monkeypatch):
    wrapper = MLXModelWrapper.__new__(MLXModelWrapper)
    wrapper._model = object()
    wrapper.model = object()
    wrapper.tokenizer = object()

    collect_calls: list[str] = []
    clear_calls: list[str] = []

    monkeypatch.setattr(
        "modelcypher.adapters.lm_eval_mlx_wrapper.gc.collect",
        lambda: collect_calls.append("gc"),
    )

    import mlx.core as mx

    monkeypatch.setattr(mx, "clear_cache", lambda: clear_calls.append("clear"))

    wrapper.cleanup()

    assert collect_calls == ["gc"]
    assert clear_calls == ["clear"]
    assert not hasattr(wrapper, "_model")
    assert not hasattr(wrapper, "model")
    assert not hasattr(wrapper, "tokenizer")
