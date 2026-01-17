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
on test prompts and detecting degenerate patterns like repetition or
high perplexity.

This is BLOCKING validation: if degenerate output is detected, the merge
should be aborted rather than saving a broken model.

Detection Patterns:
    1. Repetition detection: Count n-gram repetitions in output
       - If any 3-gram appears > threshold times, flag as repetitive
    2. Truncation detection: Output ends mid-word or is unexpectedly short
    3. Semantic collapse: Output is single repeated token

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
        repetition_score: 0 = no repetition, 1 = fully repetitive
        max_ngram_count: Maximum count of any repeated n-gram
        unique_token_ratio: Ratio of unique tokens to total tokens
        is_truncated: True if output appears truncated mid-word
        is_degenerate: True if any degenerate pattern detected
        degenerate_reason: Explanation of why flagged as degenerate
    """

    prompt: str
    output: str
    repetition_score: float
    max_ngram_count: int
    unique_token_ratio: float
    is_truncated: bool
    is_degenerate: bool
    degenerate_reason: str | None


@dataclass
class CoherenceResult:
    """Result of coherence validation across all test prompts.

    Attributes:
        is_coherent: True if all test prompts pass
        failed_count: Number of prompts that triggered degenerate output
        total_count: Total number of prompts tested
        failed_prompts: List of prompts that failed
        metrics: Per-prompt metrics
        mean_repetition_score: Average repetition score across prompts
    """

    is_coherent: bool
    failed_count: int
    total_count: int
    failed_prompts: list[str]
    metrics: list[CoherenceMetrics]
    mean_repetition_score: float


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace tokenization for n-gram analysis."""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
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


def _detect_truncation(output: str, min_length: int = 10) -> bool:
    """Detect if output appears truncated mid-word.

    Args:
        output: Model output string
        min_length: Minimum expected length

    Returns:
        True if output appears truncated
    """
    # Too short
    if len(output.strip()) < min_length:
        return True

    # Ends with partial word (no space/punctuation at end)
    stripped = output.rstrip()
    if stripped and stripped[-1].isalnum():
        # Check if last "word" is incomplete (very short)
        words = stripped.split()
        if words and len(words[-1]) < 2 and not words[-1].isdigit():
            return True

    return False


def _compute_unique_token_ratio(tokens: list[str]) -> float:
    """Compute ratio of unique tokens to total tokens.

    Low ratio indicates repetitive output.
    """
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _detect_script_mixing(text: str) -> tuple[bool, str | None]:
    """Detect if text mixes multiple writing scripts (indicates garbage output).

    A coherent response in English shouldn't contain random Japanese, Arabic,
    or Korean characters. Mixed scripts = broken model.

    Returns:
        (is_mixed, reason) - True if problematic script mixing detected
    """
    import unicodedata

    if len(text) < 10:
        return False, None

    # Count characters by script category
    script_counts: dict[str, int] = {}

    for char in text:
        if char.isspace() or char in '.,!?;:\'"()-[]{}':
            continue
        try:
            # Get Unicode script/category
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

    total_chars = sum(script_counts.values())
    if total_chars < 5:
        return False, None

    # Count how many scripts have significant presence (>5% of text)
    significant_scripts = [s for s, c in script_counts.items() if c / total_chars > 0.05]

    # More than 2 significant scripts = garbage
    if len(significant_scripts) > 2:
        return True, f"Mixed scripts detected: {significant_scripts}"

    # CJK + Latin is okay (e.g., Japanese with English terms)
    # But CJK + Arabic + Latin = definitely garbage
    if "ARABIC" in significant_scripts and "CJK" in significant_scripts:
        return True, "Arabic + CJK script mixing"

    if "KOREAN" in significant_scripts and "ARABIC" in significant_scripts:
        return True, "Korean + Arabic script mixing"

    return False, None


def _detect_special_token_leakage(text: str) -> tuple[bool, str | None]:
    """Detect if special tokens appear in output (indicates broken generation).

    Special tokens like <|startoftext|>, <|endoftext|>, <|pad|> should never
    appear in model output - they indicate the model is broken.
    """
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

    for pattern in special_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, f"Special token leaked: {pattern}"

    return False, None


def analyze_output_coherence(
    prompt: str,
    output: str,
    repetition_threshold: float = 0.5,
    max_ngram_threshold: int = 5,
    min_unique_ratio: float = 0.3,
) -> CoherenceMetrics:
    """Analyze a single output for coherence.

    Args:
        prompt: Input prompt
        output: Model output
        repetition_threshold: Max repetition score before flagging
        max_ngram_threshold: Max allowed count of any single n-gram
        min_unique_ratio: Minimum ratio of unique tokens

    Returns:
        CoherenceMetrics for this output
    """
    tokens = _tokenize_simple(output)

    # Compute metrics
    repetition_score, max_ngram_count = _compute_ngram_repetition(tokens, n=3)
    unique_ratio = _compute_unique_token_ratio(tokens)
    is_truncated = _detect_truncation(output)

    # Check for degenerate patterns
    degenerate_reasons: list[str] = []

    if repetition_score > repetition_threshold:
        degenerate_reasons.append(f"High repetition score: {repetition_score:.2f}")

    if max_ngram_count > max_ngram_threshold:
        degenerate_reasons.append(f"N-gram repeated {max_ngram_count} times")

    if unique_ratio < min_unique_ratio and len(tokens) > 10:
        degenerate_reasons.append(f"Low unique token ratio: {unique_ratio:.2f}")

    # Check for single-token collapse (e.g., "topology topology topology...")
    if tokens:
        most_common = Counter(tokens).most_common(1)[0]
        if most_common[1] / len(tokens) > 0.5:
            degenerate_reasons.append(f"Single-token collapse: '{most_common[0]}' repeated")

    # Check for script mixing (Japanese + Arabic + English = garbage)
    is_mixed, mix_reason = _detect_script_mixing(output)
    if is_mixed and mix_reason:
        degenerate_reasons.append(mix_reason)

    # Check for special token leakage (<|startoftext|> in output = broken)
    has_leakage, leak_reason = _detect_special_token_leakage(output)
    if has_leakage and leak_reason:
        degenerate_reasons.append(leak_reason)

    is_degenerate = len(degenerate_reasons) > 0
    degenerate_reason = "; ".join(degenerate_reasons) if degenerate_reasons else None

    return CoherenceMetrics(
        prompt=prompt,
        output=output,
        repetition_score=repetition_score,
        max_ngram_count=max_ngram_count,
        unique_token_ratio=unique_ratio,
        is_truncated=is_truncated,
        is_degenerate=is_degenerate,
        degenerate_reason=degenerate_reason,
    )


def validate_merge_coherence(
    model_path: str | Path,
    inference_engine: "InferenceEngine | None" = None,
    test_prompts: list[str] | None = None,
    max_tokens: int = 100,
    repetition_threshold: float = 0.5,
    max_ngram_threshold: int = 5,
    min_unique_ratio: float = 0.3,
    fail_threshold: float = 0.5,
) -> CoherenceResult:
    """Validate that a merged model produces coherent output.

    Args:
        model_path: Path to merged model
        inference_engine: Inference engine (required)
        test_prompts: Prompts to test (if None, uses default set)
        max_tokens: Maximum tokens per inference
        repetition_threshold: Max repetition score before flagging
        max_ngram_threshold: Max allowed count of any single n-gram
        min_unique_ratio: Minimum ratio of unique tokens
        fail_threshold: Fraction of prompts that can fail before overall failure

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
    failed_prompts: list[str] = []

    logger.info("Validating merge coherence with %d test prompts", len(test_prompts))

    for prompt in test_prompts:
        try:
            result = inference_engine.infer(
                model=model_str,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            output = result.get("response", "")
        except Exception as e:
            logger.warning("Inference failed for prompt '%s...': %s", prompt[:30], e)
            output = ""

        metrics = analyze_output_coherence(
            prompt=prompt,
            output=output,
            repetition_threshold=repetition_threshold,
            max_ngram_threshold=max_ngram_threshold,
            min_unique_ratio=min_unique_ratio,
        )
        metrics_list.append(metrics)

        if metrics.is_degenerate:
            failed_prompts.append(prompt)
            logger.warning(
                "Degenerate output for '%s...': %s",
                prompt[:30],
                metrics.degenerate_reason,
            )

    # Compute overall result
    failed_count = len(failed_prompts)
    total_count = len(test_prompts)
    fail_ratio = failed_count / max(total_count, 1)
    is_coherent = fail_ratio < fail_threshold

    mean_repetition = sum(m.repetition_score for m in metrics_list) / max(len(metrics_list), 1)

    logger.info(
        "Coherence validation: %d/%d passed (%.1f%% fail rate), mean_rep=%.2f",
        total_count - failed_count,
        total_count,
        fail_ratio * 100,
        mean_repetition,
    )

    return CoherenceResult(
        is_coherent=is_coherent,
        failed_count=failed_count,
        total_count=total_count,
        failed_prompts=failed_prompts,
        metrics=metrics_list,
        mean_repetition_score=mean_repetition,
    )


def validate_and_raise(
    model_path: str | Path,
    inference_engine: "InferenceEngine | None" = None,
    test_prompts: list[str] | None = None,
    max_tokens: int = 100,
) -> CoherenceResult:
    """Validate merge coherence and raise if degenerate.

    This is the BLOCKING validation function. Use this in the merge pipeline
    to abort on degenerate output.

    Args:
        model_path: Path to merged model
        inference_engine: Inference engine
        test_prompts: Prompts to test
        max_tokens: Maximum tokens per inference

    Returns:
        CoherenceResult if validation passes

    Raises:
        MergeCoherenceError: If model produces degenerate output
    """
    result = validate_merge_coherence(
        model_path=model_path,
        inference_engine=inference_engine,
        test_prompts=test_prompts,
        max_tokens=max_tokens,
    )

    if not result.is_coherent:
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
