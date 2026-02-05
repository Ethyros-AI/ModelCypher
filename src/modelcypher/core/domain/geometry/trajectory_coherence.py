# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Trajectory coherence validation for merged models.

Validates that a merged model produces coherent output by running inference
on test prompts and comparing raw coherence metrics to a baseline model.
No fixed thresholds are used: regression is defined as any metric worsening
relative to the baseline for the same prompts.

Why This Matters:
    Degenerate model behavior (repeating "topology topology topology") is
    catastrophic - the geometry has collapsed into a fixed-point attractor.
    Better to detect and abort early than save a useless model.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.inference import InferenceEngine

logger = logging.getLogger(__name__)


class MergeCoherenceError(Exception):
    """Raised when merged model produces degenerate output."""

    def __init__(self, message: str, failed_prompts: list[str], metrics: list["CoherenceMetrics"]):
        super().__init__(message)
        self.failed_prompts = failed_prompts
        self.metrics = metrics


@dataclass
class CoherenceMetrics:
    """Metrics for a single inference result.

    Attributes:
        prompt: The input prompt
        output: The model's output
        token_count: Number of tokens in output
        char_count: Number of characters in output
        repetition_score: Max n-gram repetition score across n
        max_ngram_count: Maximum count of any repeated n-gram
        max_ngram_size: N-gram size where repetition was maximal
        unique_token_ratio: Ratio of unique tokens to total tokens
        max_token_ratio: Frequency of most common token
        max_char_run_ratio: Longest run of identical chars / total chars
        max_pattern_repeat_ratio: Longest repeated substring coverage ratio
        pattern_repeat: Repeated substring with max coverage (if any)
        script_categories: Script categories present in output
        special_token_matches: Special token patterns found in output
        is_truncated: True if output ends mid-token boundary
        is_degenerate: True if output is worse than baseline
        degenerate_reason: Explanation of why flagged as degenerate
    """

    prompt: str
    output: str
    token_count: int
    char_count: int
    repetition_score: float
    max_ngram_count: int
    max_ngram_size: int
    unique_token_ratio: float
    max_token_ratio: float
    max_char_run_ratio: float
    max_pattern_repeat_ratio: float
    pattern_repeat: str | None
    script_categories: list[str]
    special_token_matches: list[str]
    is_truncated: bool
    is_degenerate: bool
    degenerate_reason: str | None


@dataclass
class TrajectoryCoherenceResult:
    """Result of coherence validation across all test prompts.

    Attributes:
        is_coherent: True if merged output is not worse than baseline
        failed_count: Number of prompts that triggered degenerate output
        total_count: Total number of prompts tested
        failed_prompts: List of prompts that failed
        metrics: Per-prompt metrics
        baseline_metrics: Per-prompt baseline metrics (if available)
        mean_repetition_score: Average repetition score across prompts
    """

    is_coherent: bool | None
    failed_count: int | None
    total_count: int
    failed_prompts: list[str]
    metrics: list[CoherenceMetrics]
    baseline_metrics: list[CoherenceMetrics] | None
    mean_repetition_score: float


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenization for n-gram analysis."""
    # Split on whitespace and punctuation, handling Unicode
    # Use \w+ which matches Unicode word characters in Python 3
    tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
    return tokens



def _compute_ngram_repetition(tokens: list[str], n: int = 3) -> tuple[float, int]:
    """Compute repetition score based on n-gram frequency.

    Args:
        tokens: List of tokens
        n: N-gram size (default 3)

    Returns:
        (repetition_score, max_count) where:
        - repetition_score: 0 = no repetition, 1 = fully repetitive
        - max_count: Maximum count of any single n-gram
    """
    if len(tokens) < n:
        return 0.0, 0

    # Generate n-grams
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    if not ngrams:
        return 0.0, 0

    # Count frequencies
    counter = Counter(ngrams)

    # Get max count and unique count
    max_count = max(counter.values())
    unique_count = len(counter)
    total_count = len(ngrams)

    # Repetition score: 1 - (unique / total)
    # Score of 0 = all unique, score of 1 = all same
    repetition_score = 1.0 - (unique_count / total_count)

    return repetition_score, max_count


def _compute_ngram_repetition_stats(tokens: list[str]) -> tuple[float, int, int]:
    """Compute max repetition score across all n-gram sizes."""
    if not tokens:
        return 0.0, 0, 0

    max_score = 0.0
    max_count = 0
    max_n = 1

    for n in range(1, len(tokens) + 1):
        score, count = _compute_ngram_repetition(tokens, n=n)
        if score > max_score or (score == max_score and count > max_count):
            max_score = score
            max_count = count
            max_n = n

    return max_score, max_count, max_n


def _max_token_ratio(tokens: list[str]) -> float:
    """Compute frequency ratio of the most common token."""
    if not tokens:
        return 0.0
    counter = Counter(tokens)
    return max(counter.values()) / len(tokens)


def _max_char_run_ratio(text: str) -> float:
    """Compute the longest run of identical chars as a ratio."""
    if not text:
        return 0.0
    max_run = 1
    current_run = 1
    for idx in range(1, len(text)):
        if text[idx] == text[idx - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run / len(text)


def _max_pattern_repeat_ratio(text: str) -> tuple[float, str | None]:
    """Compute max repeated substring coverage ratio."""
    text_lower = text.lower()
    n = len(text_lower)
    if n < 4:
        return 0.0, None

    max_ratio = 0.0
    max_pattern = None

    for pattern_len in range(2, (n // 2) + 1):
        for start in range(0, n - (pattern_len * 2) + 1):
            pattern = text_lower[start:start + pattern_len]
            if len(set(pattern)) <= 1:
                continue
            count = 1
            pos = start + pattern_len
            while pos + pattern_len <= n and text_lower[pos:pos + pattern_len] == pattern:
                count += 1
                pos += pattern_len
            if count > 1:
                ratio = (count * pattern_len) / n
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_pattern = pattern

    return max_ratio, max_pattern


def _detect_truncation(output: str) -> bool:
    """Detect if output ends mid-token boundary."""
    stripped = output.rstrip()
    if not stripped:
        return True
    return stripped[-1].isalnum()


def _compute_unique_token_ratio(tokens: list[str]) -> float:
    """Compute ratio of unique tokens to total tokens.

    Low ratio indicates repetitive output.
    """
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _script_categories(text: str) -> list[str]:
    """Return script categories present in text."""
    import unicodedata

    script_counts: dict[str, int] = {}

    for char in text:
        if char.isspace() or char in '.,!?;:\'"()-[]{}':
            continue
        try:
            name = unicodedata.name(char, "UNKNOWN")
            if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
                script = "CJK"
            elif "ARABIC" in name:
                script = "ARABIC"
            elif "HANGUL" in name:
                script = "KOREAN"
            elif "CYRILLIC" in name:
                script = "CYRILLIC"
            elif "LATIN" in name or char.isascii():
                script = "LATIN"
            else:
                script = "OTHER"
            script_counts[script] = script_counts.get(script, 0) + 1
        except Exception:
            continue

    categories = [script for script, count in script_counts.items() if count > 0]
    categories.sort()
    return categories


def _find_special_token_matches(text: str) -> list[str]:
    """Return special token patterns found in output."""
    special_patterns = [
        r"<\|startoftext\|>",
        r"<\|endoftext\|>",
        r"<\|pad\|>",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[PAD\]",
        r"\[SEP\]",
        r"\[CLS\]",
        r"\[UNK\]",
        r"<unk>",
        r"<pad>",
    ]

    matches = []
    for pattern in special_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(pattern)

    return matches


def analyze_output_coherence(
    prompt: str,
    output: str,
) -> CoherenceMetrics:
    """Analyze a single output for coherence metrics."""
    if not output or len(output.strip()) == 0:
        return CoherenceMetrics(
            prompt=prompt,
            output=output,
            token_count=0,
            char_count=0,
            repetition_score=1.0,
            max_ngram_count=0,
            max_ngram_size=0,
            unique_token_ratio=0.0,
            max_token_ratio=0.0,
            max_char_run_ratio=0.0,
            max_pattern_repeat_ratio=0.0,
            pattern_repeat=None,
            script_categories=[],
            special_token_matches=[],
            is_truncated=True,
            is_degenerate=False,
            degenerate_reason=None,
        )

    tokens = _tokenize_simple(output)

    repetition_score, max_ngram_count, max_ngram_size = _compute_ngram_repetition_stats(tokens)
    unique_ratio = _compute_unique_token_ratio(tokens)
    max_token_ratio = _max_token_ratio(tokens)
    is_truncated = _detect_truncation(output)
    max_char_run_ratio = _max_char_run_ratio(output)
    max_pattern_repeat_ratio, pattern_repeat = _max_pattern_repeat_ratio(output)
    script_categories = _script_categories(output)
    special_token_matches = _find_special_token_matches(output)

    return CoherenceMetrics(
        prompt=prompt,
        output=output,
        token_count=len(tokens),
        char_count=len(output),
        repetition_score=repetition_score,
        max_ngram_count=max_ngram_count,
        max_ngram_size=max_ngram_size,
        unique_token_ratio=unique_ratio,
        max_token_ratio=max_token_ratio,
        max_char_run_ratio=max_char_run_ratio,
        max_pattern_repeat_ratio=max_pattern_repeat_ratio,
        pattern_repeat=pattern_repeat,
        script_categories=script_categories,
        special_token_matches=special_token_matches,
        is_truncated=is_truncated,
        is_degenerate=False,
        degenerate_reason=None,
    )


def _compare_to_baseline(
    merged: CoherenceMetrics,
    baseline: CoherenceMetrics,
) -> tuple[bool, str | None]:
    """Compare merged metrics against baseline without fixed thresholds."""
    reasons: list[str] = []

    if merged.output.strip() == "" and baseline.output.strip() != "":
        reasons.append("empty_output")

    if merged.repetition_score > baseline.repetition_score:
        reasons.append("repetition_score")
    if merged.max_ngram_count > baseline.max_ngram_count:
        reasons.append("max_ngram_count")
    if merged.unique_token_ratio < baseline.unique_token_ratio:
        reasons.append("unique_token_ratio")
    if merged.max_token_ratio > baseline.max_token_ratio:
        reasons.append("max_token_ratio")
    if merged.max_char_run_ratio > baseline.max_char_run_ratio:
        reasons.append("char_run_ratio")
    if merged.max_pattern_repeat_ratio > baseline.max_pattern_repeat_ratio:
        reasons.append("pattern_repeat_ratio")
    if merged.is_truncated and not baseline.is_truncated:
        reasons.append("truncation")
    if len(merged.script_categories) > len(baseline.script_categories):
        reasons.append("script_categories")
    if len(merged.special_token_matches) > len(baseline.special_token_matches):
        reasons.append("special_token_leakage")

    is_degenerate = len(reasons) > 0
    degenerate_reason = "; ".join(reasons) if reasons else None

    return is_degenerate, degenerate_reason


def validate_merge_coherence(
    model_path: str | Path,
    inference_engine: "InferenceEngine | None" = None,
    test_prompts: list[str] | None = None,
    baseline_model_path: str | Path | None = None,
) -> CoherenceResult:
    """Validate that a merged model produces coherent output.

    Args:
        model_path: Path to merged model
        inference_engine: Inference engine (required)
        test_prompts: Prompts to test (if None, uses default set)
        baseline_model_path: Model to compare against for baseline coherence

    Returns:
        CoherenceResult

    Raises:
        MergeCoherenceError: If model produces degenerate output and blocking is enabled
    """
    # Default test prompts designed to elicit diverse, coherent responses
    if test_prompts is None:
        test_prompts = [
            "The most important thing about machine learning is",
            "In mathematics, a function is defined as",
            "The capital of France is Paris, and the capital of Germany is",
            "When you want to write good code, you should",
            "A neural network consists of layers that",
        ]

    if inference_engine is None:
        raise ValueError("validate_merge_coherence requires an inference_engine")

    model_str = str(model_path)
    metrics_list: list[CoherenceMetrics] = []
    baseline_metrics_list: list[CoherenceMetrics] | None = [] if baseline_model_path else None
    failed_prompts: list[str] = []

    logger.info("Validating merge coherence with %d test prompts", len(test_prompts))

    for prompt in test_prompts:
        baseline_metrics = None
        if baseline_model_path:
            try:
                baseline_result = inference_engine.infer(
                    model=str(baseline_model_path),
                    prompt=prompt,
                )
                baseline_output = baseline_result.get("response", "")
            except Exception as e:
                logger.warning("Baseline inference failed for '%s...': %s", prompt[:30], e)
                baseline_output = ""
            baseline_metrics = analyze_output_coherence(
                prompt=prompt,
                output=baseline_output,
            )
            if baseline_metrics_list is not None:
                baseline_metrics_list.append(baseline_metrics)

        try:
            result = inference_engine.infer(
                model=model_str,
                prompt=prompt,
            )
            output = result.get("response", "")
        except Exception as e:
            logger.warning("Inference failed for prompt '%s...': %s", prompt[:30], e)
            output = ""

        metrics = analyze_output_coherence(
            prompt=prompt,
            output=output,
        )

        if baseline_metrics:
            is_degenerate, reason = _compare_to_baseline(metrics, baseline_metrics)
            metrics.is_degenerate = is_degenerate
            metrics.degenerate_reason = reason

        metrics_list.append(metrics)

        if metrics.is_degenerate:
            failed_prompts.append(prompt)
            logger.warning(
                "Degenerate output for '%s...': %s",
                prompt[:30],
                metrics.degenerate_reason,
            )

    # Compute overall result
    total_count = len(test_prompts)
    failed_count = len(failed_prompts) if baseline_model_path else None
    is_coherent = (failed_count == 0) if baseline_model_path else None

    mean_repetition = sum(m.repetition_score for m in metrics_list) / max(len(metrics_list), 1)

    if baseline_model_path:
        logger.info(
            "Coherence validation: %d/%d passed, mean_rep=%.2f",
            total_count - (failed_count or 0),
            total_count,
            mean_repetition,
        )
    else:
        logger.info(
            "Coherence validation: metrics computed for %d prompts (mean_rep=%.2f)",
            total_count,
            mean_repetition,
        )

    return CoherenceResult(
        is_coherent=is_coherent,
        failed_count=failed_count,
        total_count=total_count,
        failed_prompts=failed_prompts,
        metrics=metrics_list,
        baseline_metrics=baseline_metrics_list,
        mean_repetition_score=mean_repetition,
    )


def validate_and_raise(
    model_path: str | Path,
    inference_engine: "InferenceEngine | None" = None,
    test_prompts: list[str] | None = None,
    baseline_model_path: str | Path | None = None,
) -> CoherenceResult:
    """Validate merge coherence and raise if degenerate.

    This is the BLOCKING validation function. Use this in the merge pipeline
    to abort on degenerate output.

    Args:
        model_path: Path to merged model
        inference_engine: Inference engine
        test_prompts: Prompts to test
        baseline_model_path: Model to compare against for baseline coherence

    Returns:
        CoherenceResult if validation passes

    Raises:
        MergeCoherenceError: If model produces degenerate output
    """
    if baseline_model_path is None:
        raise ValueError("validate_and_raise requires baseline_model_path")

    result = validate_merge_coherence(
        model_path=model_path,
        inference_engine=inference_engine,
        test_prompts=test_prompts,
        baseline_model_path=baseline_model_path,
    )

    if result.is_coherent is False:
        raise MergeCoherenceError(
            f"Merged model produces degenerate output: {result.failed_count}/{result.total_count} "
            f"prompts failed. Mean repetition score: {result.mean_repetition_score:.2f}",
            failed_prompts=result.failed_prompts,
            metrics=result.metrics,
        )

    return result


__all__ = [
    "CoherenceMetrics",
    "CoherenceResult",
    "MergeCoherenceError",
    "analyze_output_coherence",
    "validate_merge_coherence",
    "validate_and_raise",
]
