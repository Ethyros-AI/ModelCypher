from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.dimensional_alignment import (
    DimensionalAlignment,
    measure_1d_alignment,
    measure_dimensional_alignment,
)


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int], encodings: dict[str, list[int]] | None = None) -> None:
        self._vocab = vocab
        self._encodings = encodings or {}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self._encodings:
            return self._encodings[text]
        return list(range(len(text.split())))


def test_measure_1d_alignment_vocab_and_sequence_cases() -> None:
    identical = _FakeTokenizer({"a": 0, "b": 1})
    identical_result = measure_1d_alignment(identical, identical)
    assert identical_result["vocab_overlap"] == pytest.approx(1.0, abs=1e-8)
    assert identical_result["vocab_jaccard"] == pytest.approx(1.0, abs=1e-8)

    disjoint_src = _FakeTokenizer({"a": 0, "b": 1})
    disjoint_tgt = _FakeTokenizer({"x": 0, "y": 1})
    disjoint_result = measure_1d_alignment(disjoint_src, disjoint_tgt)
    assert disjoint_result["vocab_overlap"] == pytest.approx(0.0, abs=1e-8)
    assert disjoint_result["vocab_jaccard"] == pytest.approx(0.0, abs=1e-8)

    partial_src = _FakeTokenizer(
        {"a": 0, "b": 1, "c": 2},
        encodings={"t1": [1, 2], "t2": [1], "t3": [1, 2, 3]},
    )
    partial_tgt = _FakeTokenizer(
        {"b": 0, "c": 1, "d": 2, "e": 3},
        encodings={"t1": [5, 6], "t2": [5, 6], "t3": [5, 6, 7]},
    )
    partial_result = measure_1d_alignment(
        partial_src,
        partial_tgt,
        test_texts=["t1", "t2", "t3"],
    )
    assert partial_result["vocab_overlap"] == pytest.approx(0.5, abs=1e-8)
    assert partial_result["vocab_jaccard"] == pytest.approx(2.0 / 5.0, abs=1e-8)
    assert partial_result["sequence_agreement"] == pytest.approx(2.0 / 3.0, abs=1e-8)

    no_text_result = measure_1d_alignment(partial_src, partial_tgt, test_texts=None)
    assert no_text_result["sequence_agreement"] == pytest.approx(0.0, abs=1e-8)

    empty_vocab = _FakeTokenizer({})
    empty_result = measure_1d_alignment(empty_vocab, empty_vocab)
    assert empty_result["vocab_overlap"] == pytest.approx(0.0, abs=1e-8)
    assert empty_result["vocab_jaccard"] == pytest.approx(0.0, abs=1e-8)


def test_dimensional_alignment_summary_sections() -> None:
    full = DimensionalAlignment(
        vocab_overlap=0.8,
        vocab_jaccard=0.7,
        sequence_agreement=0.9,
        shared_token_count=12,
        source_vocab_size=20,
        target_vocab_size=15,
        embedding_cka=0.95,
        layernorm_cka=0.97,
        hidden_cka_mean=0.91,
        hidden_cka_min=0.85,
        intermediate_cka_mean=0.88,
    )
    full_summary = full.summary()
    assert "1D (Tokenization):" in full_summary
    assert "2D (Embedding):" in full_summary
    assert "3D (LayerNorm):" in full_summary
    assert "4D+ (Transformer):" in full_summary
    assert "Hidden CKA (mean):" in full_summary
    assert "Intermediate CKA (mean):" in full_summary

    minimal = DimensionalAlignment(
        vocab_overlap=0.2,
        vocab_jaccard=0.1,
        sequence_agreement=0.3,
        shared_token_count=2,
        source_vocab_size=10,
        target_vocab_size=10,
        embedding_cka=None,
        layernorm_cka=None,
        hidden_cka_mean=None,
        hidden_cka_min=None,
        intermediate_cka_mean=None,
    )
    minimal_summary = minimal.summary()
    assert "2D (Embedding):" not in minimal_summary
    assert "3D (LayerNorm):" not in minimal_summary
    assert "4D+ (Transformer):" not in minimal_summary


def test_measure_dimensional_alignment_aggregates_and_layernorm_passthrough(monkeypatch) -> None:
    source = _FakeTokenizer({"a": 0, "b": 1})
    target = _FakeTokenizer({"a": 0, "c": 1})

    probe_metrics = {
        "embedding_cka": 0.91,
        "layer_0_cka": 0.8,
        "layer_1_cka": 0.6,
        "intermediate_0_cka": 0.5,
        "foo_intermediate_2_cka": 0.7,
        "layernorm_cka": 0.77,
    }

    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("measure_3d_alignment should not run when layernorm_model=None")

    monkeypatch.setattr(
        "modelcypher.core.domain.geometry.dimensional_alignment.measure_3d_alignment",
        _should_not_be_called,
    )

    result = measure_dimensional_alignment(
        source_tokenizer=source,
        target_tokenizer=target,
        probe_metrics=probe_metrics,
        test_texts=["hello world"],
        layernorm_model=None,
    )

    assert result.embedding_cka == pytest.approx(0.91, abs=1e-8)
    assert result.layernorm_cka == pytest.approx(0.77, abs=1e-8)
    assert result.hidden_cka_mean == pytest.approx((0.8 + 0.6) / 2.0, abs=1e-8)
    assert result.hidden_cka_min == pytest.approx(0.6, abs=1e-8)
    assert result.intermediate_cka_mean == pytest.approx((0.5 + 0.7) / 2.0, abs=1e-8)


def test_measure_dimensional_alignment_empty_probe_metrics() -> None:
    source = _FakeTokenizer({"a": 0})
    target = _FakeTokenizer({"b": 0})

    result = measure_dimensional_alignment(
        source_tokenizer=source,
        target_tokenizer=target,
        probe_metrics={},
        test_texts=["x"],
    )

    assert result.embedding_cka is None
    assert result.layernorm_cka is None
    assert result.hidden_cka_mean is None
    assert result.hidden_cka_min is None
    assert result.intermediate_cka_mean is None
