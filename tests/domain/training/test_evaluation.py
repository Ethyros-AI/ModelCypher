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

import numpy as np
import pytest

from modelcypher.core.domain.training.evaluation import (
    EvaluationBatch,
    EvaluationConfig,
    EvaluationEngine,
    EvaluationError,
    EvaluationMetric,
    EvaluationProgress,
)


class _MiniBackend:
    def array(self, data, dtype: str | None = None):
        return np.array(data, dtype=dtype)

    def eval(self, *_args) -> None:
        return None

    def reshape(self, arr, shape):
        return np.reshape(arr, shape)

    def log_softmax(self, arr, axis: int = -1):
        arr64 = np.asarray(arr, dtype=np.float64)
        shifted = arr64 - np.max(arr64, axis=axis, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    def exp(self, arr):
        return np.exp(arr)

    def log(self, arr):
        return np.log(arr)

    def to_scalar(self, value):
        return np.asarray(value).item()

    def sum(self, arr):
        return np.sum(arr)

    def arange(self, n: int):
        return np.arange(n)

    def argmax(self, arr, axis: int = -1):
        return np.argmax(arr, axis=axis)

    def encode_tokens(self, tokenizer, sample: str):
        return tokenizer.encode(sample)


class _Tokenizer:
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self._mapping = mapping

    def encode(self, text: str) -> list[int]:
        return self._mapping[text]


def test_evaluation_progress_percentage_and_zero_guard() -> None:
    progress = EvaluationProgress(samples_processed=3, total_samples=4, current_metric=0.5)
    assert progress.percentage == pytest.approx(0.75)

    zero_total = EvaluationProgress(samples_processed=3, total_samples=0)
    assert zero_total.percentage == 0.0


def test_evaluate_raises_when_dataset_is_empty(monkeypatch, tmp_path: Path) -> None:
    backend = _MiniBackend()
    engine = EvaluationEngine(backend=backend, config=EvaluationConfig.default())
    monkeypatch.setattr(engine, "_load_dataset", lambda _path: [])

    with pytest.raises(EvaluationError, match="No samples found"):
        engine.evaluate(model=object(), tokenizer=object(), dataset_path=tmp_path / "missing.jsonl")


def test_evaluate_computes_all_metrics_and_progress(monkeypatch, tmp_path: Path) -> None:
    backend = _MiniBackend()
    config = EvaluationConfig(
        metrics=[
            EvaluationMetric.LOSS,
            EvaluationMetric.PERPLEXITY,
            EvaluationMetric.ACCURACY,
            EvaluationMetric.BITS_PER_CHARACTER,
        ],
        batch_size=2,
    )
    engine = EvaluationEngine(backend=backend, config=config)

    monkeypatch.setattr(engine, "_load_dataset", lambda _path: ["s1", "s2"])

    batches = [
        EvaluationBatch(
            inputs=np.array([[1, 2]], dtype=np.int32),
            targets=np.array([[1, 2]], dtype=np.int32),
            mask=np.array([[1.0, 1.0]], dtype=np.float32),
            valid_token_counts=[2],
        ),
        EvaluationBatch(
            inputs=np.array([[3, 4]], dtype=np.int32),
            targets=np.array([[3, 4]], dtype=np.int32),
            mask=np.array([[1.0, 1.0]], dtype=np.float32),
            valid_token_counts=[2],
        ),
    ]
    monkeypatch.setattr(engine, "_create_batches", lambda _samples, _tok: iter(batches))
    monkeypatch.setattr(engine, "_forward", lambda _model, _inputs: np.zeros((1, 2, 3)))

    loss_values = iter([(2.0, 3), (1.0, 1)])
    monkeypatch.setattr(engine, "_compute_loss", lambda *_args: next(loss_values))

    accuracy_values = iter([2, 1])
    monkeypatch.setattr(engine, "_compute_accuracy", lambda *_args: next(accuracy_values))

    progress_updates: list[EvaluationProgress] = []
    result = engine.evaluate(
        model=object(),
        tokenizer=object(),
        dataset_path=tmp_path / "eval.jsonl",
        progress_callback=progress_updates.append,
    )

    expected_loss = (2.0 * 3 + 1.0 * 1) / 4.0

    assert result.samples_evaluated == 2
    assert result.tokens_evaluated == 4
    assert result.loss == pytest.approx(expected_loss)
    assert result.perplexity == pytest.approx(np.exp(expected_loss))
    assert result.accuracy == pytest.approx(3.0 / 4.0)
    assert result.metrics[EvaluationMetric.BITS_PER_CHARACTER] == pytest.approx(
        expected_loss / np.log(2.0)
    )
    assert len(progress_updates) == 2
    assert progress_updates[-1].percentage == 1.0


def test_compute_loss_accuracy_and_gather_paths() -> None:
    backend = _MiniBackend()
    engine = EvaluationEngine(backend=backend)

    logits = np.array(
        [
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    targets = np.array([[0, 1, 2]], dtype=np.int32)
    full_mask = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    loss, token_count = engine._compute_loss(logits, targets, full_mask)
    accuracy = engine._compute_accuracy(logits, targets, full_mask)

    assert token_count == 2
    assert loss < 0.05
    assert accuracy == 2

    zero_mask = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    zero_loss, zero_tokens = engine._compute_loss(logits, targets, zero_mask)
    assert zero_tokens == 0
    assert zero_loss == 0.0

    arr = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    indices = np.array([2, 1], dtype=np.int32)
    gathered = engine._gather_along_axis(arr, indices, axis=1)
    assert np.array_equal(gathered, np.array([30.0, 50.0], dtype=np.float32))


def test_load_dataset_and_batch_creation_paths(tmp_path: Path) -> None:
    backend = _MiniBackend()
    engine = EvaluationEngine(
        backend=backend,
        config=EvaluationConfig(batch_size=2, sequence_length=4),
    )

    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"text":"keep text"}',
                "not json",
                '{"content":"fallback content"}',
                '{"text":""}',
            ]
        )
    )

    loaded = engine._load_dataset(dataset_path)
    assert loaded == ["keep text", "fallback content"]

    tokenizer = _Tokenizer(
        {
            "skip": [1],
            "long": [1, 2, 3, 4, 5],
            "mid": [9, 8],
        }
    )

    batches = list(engine._create_batches(["skip", "long", "mid"], tokenizer))
    assert len(batches) == 1

    batch = batches[0]
    assert tuple(batch.inputs.shape) == (2, 4)
    assert tuple(batch.targets.shape) == (2, 4)
    assert tuple(batch.mask.shape) == (2, 4)
    assert batch.valid_token_counts == [4, 2]
