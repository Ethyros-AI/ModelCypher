# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from types import SimpleNamespace

import pytest

import modelcypher.core.domain.geometry.probe_generator as probe_generator


class _Tokenizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if self.fail:
            raise ValueError("decode failed")
        return f"tok-{token_ids[0]}"


def test_validate_full_rank_coverage_reports_deficits(any_backend) -> None:
    b = any_backend
    source = {
        0: b.array([[1.0, 0.0], [0.0, 1.0]]),
        1: [b.array([1.0, 0.0]), b.array([1.0, 0.0])],
    }
    target = {
        0: b.array([[1.0, 0.0], [0.0, 1.0]]),
        1: b.array([[1.0, 0.0], [0.0, 1.0]]),
    }

    coverage = probe_generator.validate_full_rank_coverage(source, target, b)

    assert coverage[0]["full_rank_achieved"] is True
    assert coverage[1]["full_rank_achieved"] is False
    assert coverage[1]["source_deficit"] == 1
    assert coverage[1]["deficit"] >= 1


def test_score_tokens_for_null_space(any_backend) -> None:
    b = any_backend
    activations = b.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
            [1.0, 1.0],
        ]
    )
    U_null = b.array([[1.0], [0.0]])

    scores = probe_generator.score_tokens_for_null_space(activations, U_null, b)
    values = b.tolist(scores)

    assert values[0] == pytest.approx(1.0, rel=1e-4, abs=1e-4)
    assert values[1] == pytest.approx(0.0, abs=1e-5)
    assert values[2] == pytest.approx(2**-0.5, rel=1e-3, abs=1e-3)


def test_generate_orthogonal_probe_handles_success_decode_failure_and_empty(monkeypatch, any_backend) -> None:
    b = any_backend
    generator = probe_generator.OrthogonalProbeGenerator(backend=b)
    U_null = b.array([[1.0], [0.0]])

    monkeypatch.setattr(
        probe_generator,
        "find_null_space_tokens_closed_form",
        lambda **_kwargs: [(7, 0.9)],
    )
    result = generator.generate_orthogonal_probe(object(), _Tokenizer(), U_null, layer_idx=0)
    assert result is not None
    assert result.token_ids == [7]
    assert result.text == "tok-7"
    assert result.orthogonal_component_norm == 0.9

    result_decode_fail = generator.generate_orthogonal_probe(
        object(),
        _Tokenizer(fail=True),
        U_null,
        layer_idx=0,
    )
    assert result_decode_fail is not None
    assert result_decode_fail.text == "<decode-failed>"

    monkeypatch.setattr(
        probe_generator,
        "find_null_space_tokens_closed_form",
        lambda **_kwargs: [],
    )
    assert generator.generate_orthogonal_probe(object(), _Tokenizer(), U_null, layer_idx=0) is None


def test_find_null_space_tokens_closed_form_handles_null_rank_and_arch_failures(
    monkeypatch,
    any_backend,
) -> None:
    b = any_backend
    weight = b.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    arch_ok = SimpleNamespace(
        embed_module=SimpleNamespace(weight=weight),
        layers=[lambda h: h],
    )
    monkeypatch.setattr(probe_generator, "_get_model_architecture", lambda _model: arch_ok)

    # No null directions: should return empty list.
    assert (
        probe_generator.find_null_space_tokens_closed_form(
            model=object(),
            U_null=b.zeros((2, 0)),
            layer_idx=0,
            backend=b,
        )
        == []
    )

    # One null direction: choose the strongest vocabulary token.
    top = probe_generator.find_null_space_tokens_closed_form(
        model=object(),
        U_null=b.array([[1.0], [0.0]]),
        layer_idx=0,
        backend=b,
    )
    assert len(top) == 1
    assert top[0][0] == 0
    assert top[0][1] > 0.0

    arch_missing_embed = SimpleNamespace(embed_module=None, layers=[])
    monkeypatch.setattr(probe_generator, "_get_model_architecture", lambda _model: arch_missing_embed)
    with pytest.raises(RuntimeError):
        probe_generator.find_null_space_tokens_closed_form(
            model=object(),
            U_null=b.array([[1.0], [0.0]]),
            layer_idx=0,
            backend=b,
        )


def test_find_null_space_sequences_and_texts(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(
        probe_generator,
        "find_null_space_tokens_closed_form",
        lambda **_kwargs: [(1, 0.8), (2, 0.4)],
    )
    sequences = probe_generator.find_null_space_sequences(
        model=object(),
        tokenizer=_Tokenizer(),
        U_null=b.array([[1.0], [0.0]]),
        layer_idx=0,
        backend=b,
    )
    assert [seq.token_ids for seq in sequences] == [[1], [2]]
    assert all(seq.gradient_steps == 0 for seq in sequences)

    monkeypatch.setattr(
        probe_generator,
        "find_null_space_sequences",
        lambda **_kwargs: [
            SimpleNamespace(text=" alpha "),
            SimpleNamespace(text="alpha"),
            SimpleNamespace(text=""),
            SimpleNamespace(text=" beta"),
        ],
    )
    texts = probe_generator.find_null_space_texts(
        model=object(),
        tokenizer=_Tokenizer(),
        U_null=b.array([[1.0], [0.0]]),
        layer_idx=0,
        backend=b,
    )
    assert texts == ["alpha", "beta"]


def test_augment_rank_closed_form_short_circuits_when_already_full_rank(monkeypatch, any_backend) -> None:
    b = any_backend
    monkeypatch.setattr(probe_generator, "compute_numerical_rank", lambda *_args, **_kwargs: (3, 3))

    result = probe_generator.augment_rank_closed_form(
        model=object(),
        tokenizer=_Tokenizer(),
        activations=b.zeros((2, 3)),
        layer_idx=0,
        backend=b,
    )

    assert result.initial_rank == 3
    assert result.final_rank == 3
    assert result.probes_generated == 0
    assert result.iterations == 1
    assert result.full_rank_achieved is True

