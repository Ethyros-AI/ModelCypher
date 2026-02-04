#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Verification oracle for safe self-improvement.

The VerificationOracle uses an existing verified capability (arithmetic)
to validate new learning. This is the critical safety mechanism that
prevents the model from learning nonsense.

The key insight: if the model outputs "5" for "3+2=" with 100% accuracy,
we can use this to verify that generated training samples are correct.

Example verification:
    - Word problem: "I have 3 apples. I get 2 more."
    - Candidate parse: "3+2="
    - Oracle computes: model("Arithmetic means calculating numbers. 3+2=") → "5"
    - Verify: "5" == expected_answer → CORRECT

This ensures training data is ground-truth verified.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple

import mlx.core as mx


class VerificationOracle:
    """Use verified capabilities to check new learning.

    The oracle leverages the model's existing arithmetic capability
    (unlocked via priming) to verify that generated training samples
    are mathematically correct.

    Example:
        >>> oracle = VerificationOracle(model, tokenizer)
        >>> is_correct, computed = oracle.verify("3+2=", "5")
        >>> print(f"Correct: {is_correct}, Computed: {computed}")
        Correct: True, Computed: 5
    """

    DEFAULT_PRIME: ClassVar[str] = "Arithmetic means calculating numbers."

    def __init__(
        self,
        model,
        tokenizer,
        prime: Optional[str] = None,
    ):
        """Initialize oracle.

        Args:
            model: The language model
            tokenizer: The tokenizer for the model
            prime: Prime to unlock arithmetic capability
                   (defaults to "Arithmetic means calculating numbers.")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prime = prime if prime is not None else self.DEFAULT_PRIME

    def compute(self, equation: str) -> str:
        """Compute equation using primed model.

        Args:
            equation: Equation to compute (e.g., "3+2=")

        Returns:
            Model's predicted answer as string
        """
        prompt = f"{self.prime} {equation}"
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        mx.eval(logits)

        # Get top prediction using MLX operations
        last_logits = logits[0, -1, :]
        top_token = int(mx.argmax(last_logits).item())
        return self.tokenizer.decode([top_token]).strip()

    def verify(self, equation: str, expected: str) -> Tuple[bool, str]:
        """Verify that equation produces expected answer.

        This is the core verification function. It computes the equation
        using the primed model and checks if the result matches expected.

        Args:
            equation: Equation to verify (e.g., "3+2=")
            expected: Expected answer (e.g., "5")

        Returns:
            Tuple of (is_correct, computed_answer)
        """
        computed = self.compute(equation)
        # Flexible matching: expected in computed OR exact match
        is_correct = expected in computed or computed == expected
        return is_correct, computed

    def calibrate(
        self,
        test_cases: List[Tuple[str, str]],
    ) -> Tuple[float, List[Tuple[str, str, str, bool]]]:
        """Calibrate oracle on known test cases.

        This verifies the oracle itself is reliable before using it
        to verify training data.

        Args:
            test_cases: List of (equation, expected_answer) pairs

        Returns:
            Tuple of:
            - accuracy: Float in [0, 1]
            - details: List of (equation, expected, computed, is_correct)
        """
        if not test_cases:
            return 0.0, []

        details = []
        correct = 0

        for equation, expected in test_cases:
            is_correct, computed = self.verify(equation, expected)
            details.append((equation, expected, computed, is_correct))
            if is_correct:
                correct += 1

        accuracy = correct / len(test_cases)
        return accuracy, details

    @classmethod
    def default_calibration_tests(cls) -> List[Tuple[str, str]]:
        """Return default calibration test cases.

        These cover addition and subtraction to verify the oracle
        works reliably for both operations.
        """
        return [
            # Addition
            ("1+1=", "2"),
            ("2+2=", "4"),
            ("3+1=", "4"),
            ("2+3=", "5"),
            ("4+1=", "5"),
            # Subtraction
            ("5-2=", "3"),
            ("4-1=", "3"),
            ("7-3=", "4"),
            ("6-2=", "4"),
            ("3-1=", "2"),
        ]


__all__ = ["VerificationOracle"]
