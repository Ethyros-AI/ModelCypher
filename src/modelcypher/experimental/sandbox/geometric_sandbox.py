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

"""Geometric Self-Study Sandbox.

Interactive environment for models to observe and learn from their own
geometric signatures during reasoning.

The key insight: A model that can SEE its own geometry and learn to interpret
it will naturally prefer geometrically coherent reasoning.

Core Loop:
    Model generates -> Sees geometry -> Interprets meaning -> Adjusts approach

Philosophy:
    expansion_ratio = 1.0 = balanced expand/compress cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mlx.core as mx

from modelcypher.experimental.sandbox.feedback_formatter import (
    GeometricFeedback,
    format_comparison_text,
    format_feedback_text,
    format_geometric_feedback,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandbox attempt: response + geometric signature.

    This is the core data structure for geometric self-study. The model
    generates a response, and we capture both the text and the underlying
    geometry.

    Attributes:
        prompt: The input prompt
        response: The model's generated response
        feedback: Structured geometric feedback
        feedback_text: Human-readable geometric feedback for model consumption
        raw_metrics: Raw numeric metrics for analysis
    """

    prompt: str
    response: str
    feedback: GeometricFeedback
    feedback_text: str
    raw_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def expansion_ratio(self) -> float:
        """Accessor for expansion ratio."""
        return self.feedback.expansion_ratio

    @property
    def is_aligned(self) -> bool:
        """Check if geometry is within balanced range (0.9 - 1.1)."""
        return 0.9 <= self.feedback.expansion_ratio <= 1.1


@dataclass
class ComparisonResult:
    """Result of comparing multiple reasoning approaches.

    Used for side-by-side geometric comparison of different approaches
    to the same problem.

    Attributes:
        prompt: The shared input prompt
        approaches: List of (approach_name, result) pairs
        best_approach: Name of approach with best geometry
        comparison_text: Formatted comparison for model consumption
    """

    prompt: str
    approaches: list[tuple[str, SandboxResult]]
    best_approach: str
    comparison_text: str

    @property
    def best_result(self) -> SandboxResult:
        """Get the result for the best approach."""
        for name, result in self.approaches:
            if name == self.best_approach:
                return result
        return self.approaches[0][1]


class GeometricSandbox:
    """Interactive environment for geometric self-study.

    Provides methods to:
    1. Generate responses and capture geometry (attempt)
    2. Compare multiple reasoning approaches (compare)
    3. Generate geometric self-reflection (reflect)

    Usage:
        sandbox = GeometricSandbox(model, tokenizer, generate_fn)
        result = sandbox.attempt("A bat and ball cost $1.10...")
        print(result.feedback_text)

        # Compare approaches
        comparison = sandbox.compare(
            "A bat and ball cost $1.10...",
            ["The ball costs", "Let me think step by step"]
        )
        print(comparison.comparison_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        generate_fn: Callable | None = None,
        max_tokens: int = 100,
    ) -> None:
        """Initialize the geometric sandbox.

        Args:
            model: MLX language model
            tokenizer: Tokenizer for the model
            generate_fn: Optional custom generation function.
                         Default: mlx_lm.generate
            max_tokens: Default maximum tokens for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        if generate_fn is None:
            from mlx_lm import generate
            self._generate = generate
        else:
            self._generate = generate_fn

        # Get model info
        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        self.n_layers = len(layers) if layers else 0

    def _compute_geometry(self, tokens: list[int]) -> dict[str, float]:
        """Compute geometric metrics for a token sequence.

        Uses the differentiable expansion computation from geometry module.

        Args:
            tokens: Token IDs to compute geometry for

        Returns:
            Dict with expansion_ratio and component metrics
        """
        from modelcypher.core.domain.geometry.differentiable_phi import (
            compute_expansion_metrics,
            compute_trajectory_norms,
        )

        input_ids = mx.array([tokens])
        trajectory = compute_trajectory_norms(self.model, input_ids)
        mx.eval(trajectory)

        return compute_expansion_metrics(trajectory)

    def attempt(self, prompt: str, max_tokens: int | None = None) -> SandboxResult:
        """Generate a response and capture its geometric signature.

        This is the core operation: generate -> measure -> feedback.

        Args:
            prompt: The input prompt
            max_tokens: Optional override for max tokens

        Returns:
            SandboxResult with response and geometric feedback
        """
        max_tokens = max_tokens or self.max_tokens

        # Generate response
        response = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Tokenize full sequence for geometry computation
        full_text = prompt + response
        tokens = self.tokenizer.encode(full_text)

        # Compute geometry
        metrics = self._compute_geometry(tokens)

        # Create structured feedback
        feedback = format_geometric_feedback(
            expansion_ratio=metrics["expansion_ratio"],
            peak_layer=metrics["peak_layer"],
            n_layers=int(metrics["n_layers"]),
            expansion_rate=metrics["expansion_rate"],
            compression_rate=metrics["compression_rate"],
        )

        return SandboxResult(
            prompt=prompt,
            response=response,
            feedback=feedback,
            feedback_text=format_feedback_text(feedback),
            raw_metrics=metrics,
        )

    def compare(
        self,
        prompt: str,
        approach_prefixes: list[str],
        max_tokens: int | None = None,
    ) -> ComparisonResult:
        """Compare multiple reasoning approaches geometrically.

        For each approach prefix, generates a continuation and measures
        the geometry. Returns a side-by-side comparison.

        Args:
            prompt: The base prompt (e.g., the question)
            approach_prefixes: Different starting approaches to try
                              (e.g., ["The answer is", "Let me think..."])
            max_tokens: Optional override for max tokens per approach

        Returns:
            ComparisonResult with all approaches and best selection
        """
        max_tokens = max_tokens or self.max_tokens

        approaches = []
        for prefix in approach_prefixes:
            # Prepend the approach prefix to the prompt
            full_prompt = f"{prompt}\n{prefix}"

            # Generate continuation
            response = self._generate(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Full response includes the prefix
            full_response = prefix + response

            # Compute geometry on full sequence
            tokens = self.tokenizer.encode(prompt + full_response)
            metrics = self._compute_geometry(tokens)

            feedback = format_geometric_feedback(
                expansion_ratio=metrics["expansion_ratio"],
                peak_layer=metrics["peak_layer"],
                n_layers=int(metrics["n_layers"]),
                expansion_rate=metrics["expansion_rate"],
                compression_rate=metrics["compression_rate"],
            )

            result = SandboxResult(
                prompt=prompt,
                response=full_response,
                feedback=feedback,
                feedback_text=format_feedback_text(feedback),
                raw_metrics=metrics,
            )

            approaches.append((prefix[:30], result))  # Truncate long prefixes for names

        # Find best approach (closest to expansion_ratio = 1.0)
        best_idx = min(
            range(len(approaches)),
            key=lambda i: abs(approaches[i][1].feedback.expansion_ratio - 1.0),
        )
        best_approach = approaches[best_idx][0]

        # Format comparison text
        comparison_data = [
            (name, result.feedback, result.response)
            for name, result in approaches
        ]
        comparison_text = format_comparison_text(comparison_data)

        return ComparisonResult(
            prompt=prompt,
            approaches=approaches,
            best_approach=best_approach,
            comparison_text=comparison_text,
        )

    def reflect(self, result: SandboxResult) -> str:
        """Generate a geometric self-reflection on a result.

        Creates a prompt that includes the geometric feedback, asking
        the model to interpret its own processing.

        Args:
            result: Previous SandboxResult to reflect on

        Returns:
            Generated self-reflection text
        """
        reflection_prompt = f"""I just generated this response:

Response: {result.response[:200]}...

My geometric signature was:
{result.feedback_text}

Based on this geometry, let me reflect on my processing:"""

        # Generate reflection (shorter, focused)
        reflection = self._generate(
            self.model,
            self.tokenizer,
            prompt=reflection_prompt,
            max_tokens=150,
            verbose=False,
        )

        return reflection

    def study_example(
        self,
        prompt: str,
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        """Run a full self-study cycle on an example.

        This implements the complete feedback loop:
        1. Attempt -> Get geometry
        2. Reflect on geometry
        3. Optionally re-attempt with adjusted approach

        Args:
            prompt: The problem to study
            expected_answer: Optional expected answer for validation

        Returns:
            Dict with attempt results, reflection, and analysis
        """
        # Initial attempt
        attempt1 = self.attempt(prompt)

        # Reflection
        reflection = self.reflect(attempt1)

        # Check if correct (if expected provided)
        is_correct = None
        if expected_answer:
            is_correct = expected_answer.lower() in attempt1.response.lower()

        result = {
            "prompt": prompt,
            "attempt1": {
                "response": attempt1.response,
                "expansion_ratio": attempt1.expansion_ratio,
                "is_aligned": attempt1.is_aligned,
                "feedback": attempt1.feedback_text,
            },
            "reflection": reflection,
            "is_correct": is_correct,
            "expected_answer": expected_answer,
        }

        # If not aligned or not correct, try explicit reasoning
        should_retry = (
            (not attempt1.is_aligned) or
            (is_correct is not None and not is_correct)
        )

        if should_retry:
            # Try with explicit reasoning prefix
            retry_prompt = f"{prompt}\nLet me think through this step by step."
            attempt2 = self.attempt(retry_prompt)

            is_correct_2 = None
            if expected_answer:
                is_correct_2 = expected_answer.lower() in attempt2.response.lower()

            result["attempt2"] = {
                "response": attempt2.response,
                "expansion_ratio": attempt2.expansion_ratio,
                "is_aligned": attempt2.is_aligned,
                "feedback": attempt2.feedback_text,
                "is_correct": is_correct_2,
            }

            # Did geometry improve?
            result["geometry_improved"] = (
                abs(attempt2.expansion_ratio - 1.0) < abs(attempt1.expansion_ratio - 1.0)
            )

            # Did correctness improve?
            if is_correct is not None and is_correct_2 is not None:
                result["correctness_improved"] = (not is_correct) and is_correct_2

        return result


def create_sandbox_from_path(
    model_path: str | Path,
    max_tokens: int = 100,
) -> GeometricSandbox:
    """Create a GeometricSandbox from a model path.

    Convenience function that loads the model and creates the sandbox.

    Args:
        model_path: Path to the model directory
        max_tokens: Default max tokens for generation

    Returns:
        Configured GeometricSandbox instance
    """
    from mlx_lm import load

    model, tokenizer = load(str(model_path))
    return GeometricSandbox(model, tokenizer, max_tokens=max_tokens)
