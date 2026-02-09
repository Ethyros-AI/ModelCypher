# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import replace

import pytest

from modelcypher.core.domain.geometry.trajectory_coherence import (
    CoherenceMetrics,
    MergeCoherenceError,
    _compare_to_baseline,
    _compute_ngram_repetition,
    _compute_ngram_repetition_stats,
    _detect_truncation,
    _find_special_token_matches,
    _max_char_run_ratio,
    _max_pattern_repeat_ratio,
    _max_token_ratio,
    _script_categories,
    _tokenize_simple,
    analyze_output_coherence,
    validate_and_raise,
    validate_merge_coherence,
)


class _InferenceStub:
    def __init__(self, responses: dict[tuple[str, str], str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def infer(self, model: str, prompt: str) -> dict[str, str]:
        self.calls.append((model, prompt))
        value = self.responses.get((model, prompt), "")
        if isinstance(value, Exception):
            raise value
        return {"response": value}


def _base_metrics() -> CoherenceMetrics:
    return CoherenceMetrics(
        prompt="p",
        output="good output.",
        token_count=2,
        char_count=12,
        repetition_score=0.0,
        max_ngram_count=1,
        max_ngram_size=1,
        unique_token_ratio=1.0,
        max_token_ratio=0.5,
        max_char_run_ratio=0.1,
        max_pattern_repeat_ratio=0.0,
        pattern_repeat=None,
        script_categories=["LATIN"],
        special_token_matches=[],
        is_truncated=False,
        is_degenerate=False,
        degenerate_reason=None,
    )


def test_token_and_pattern_helpers_cover_edge_cases() -> None:
    tokens = _tokenize_simple("Hello, 世界! Hello")
    assert tokens == ["hello", "世界", "hello"]

    rep1, count1 = _compute_ngram_repetition(["a", "a", "a"], n=1)
    assert rep1 == pytest.approx(2.0 / 3.0)
    assert count1 == 3

    stats = _compute_ngram_repetition_stats(["a", "a", "a"])
    assert stats == pytest.approx((2.0 / 3.0, 3, 1))

    assert _max_token_ratio(["a", "a", "b"]) == pytest.approx(2.0 / 3.0)
    assert _max_char_run_ratio("aaabb") == pytest.approx(3.0 / 5.0)

    ratio, pattern = _max_pattern_repeat_ratio("abcabcabcx")
    assert ratio == pytest.approx(0.9)
    assert pattern == "abc"

    assert _detect_truncation("Done.") is False
    assert _detect_truncation("truncated") is True


def test_script_and_special_token_helpers() -> None:
    categories = _script_categories("Hello مرحبا Привет")
    assert categories == ["ARABIC", "CYRILLIC", "LATIN"]

    matches = _find_special_token_matches("<|endoftext|> ... [PAD]")
    assert r"<\|endoftext\|>" in matches
    assert r"\[PAD\]" in matches


def test_analyze_output_coherence_for_empty_and_repetitive_output() -> None:
    empty = analyze_output_coherence(prompt="p", output="")
    assert empty.token_count == 0
    assert empty.repetition_score == pytest.approx(1.0)
    assert empty.is_truncated is True

    repetitive = analyze_output_coherence(
        prompt="p",
        output="topology topology topology <|endoftext|>",
    )
    assert repetitive.token_count >= 3
    assert repetitive.repetition_score > 0.0
    assert repetitive.max_ngram_count >= 2
    assert repetitive.special_token_matches


def test_compare_to_baseline_collects_expected_regressions() -> None:
    baseline = _base_metrics()
    merged = replace(
        baseline,
        output="",
        repetition_score=0.4,
        max_ngram_count=4,
        unique_token_ratio=0.1,
        max_token_ratio=0.9,
        max_char_run_ratio=0.8,
        max_pattern_repeat_ratio=0.7,
        script_categories=["LATIN", "ARABIC"],
        special_token_matches=[r"<\|pad\|>"],
        is_truncated=True,
    )

    is_deg, reason = _compare_to_baseline(merged, baseline)

    assert is_deg is True
    assert reason is not None
    assert "empty_output" in reason
    assert "repetition_score" in reason
    assert "special_token_leakage" in reason


def test_validate_merge_coherence_without_baseline_reports_metrics_only() -> None:
    prompts = ["p1", "p2"]
    engine = _InferenceStub(
        {
            ("merged", "p1"): "Answer one.",
            ("merged", "p2"): "Answer two.",
        }
    )

    result = validate_merge_coherence(
        model_path="merged",
        inference_engine=engine,
        test_prompts=prompts,
    )

    assert result.is_coherent is None
    assert result.failed_count is None
    assert result.total_count == 2
    assert result.failed_prompts == []
    assert result.baseline_metrics is None
    assert len(result.metrics) == 2
    assert len(engine.calls) == 2


def test_validate_merge_coherence_and_validate_and_raise_with_baseline() -> None:
    prompts = ["p1", "p2"]
    engine = _InferenceStub(
        {
            ("base", "p1"): "Baseline stable output.",
            ("base", "p2"): "Baseline stable output.",
            ("merged", "p1"): "topology topology topology topology",
            ("merged", "p2"): RuntimeError("inference failed"),
        }
    )

    result = validate_merge_coherence(
        model_path="merged",
        inference_engine=engine,
        test_prompts=prompts,
        baseline_model_path="base",
    )

    assert result.is_coherent is False
    assert result.failed_count == 2
    assert result.failed_prompts == prompts
    assert result.baseline_metrics is not None
    assert len(result.baseline_metrics) == 2
    assert any(metric.is_degenerate for metric in result.metrics)

    with pytest.raises(MergeCoherenceError) as exc:
        validate_and_raise(
            model_path="merged",
            inference_engine=engine,
            test_prompts=prompts,
            baseline_model_path="base",
        )
    assert exc.value.failed_prompts == prompts
    assert len(exc.value.metrics) == 2


def test_validate_and_raise_requires_baseline_path() -> None:
    engine = _InferenceStub({})
    with pytest.raises(ValueError):
        validate_and_raise(
            model_path="merged",
            inference_engine=engine,
            test_prompts=["prompt"],
            baseline_model_path=None,
        )

