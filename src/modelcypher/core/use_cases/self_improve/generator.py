#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Safe self-play training data generator.

The SafeSelfPlayGenerator creates oracle-verified training data
by generating (word_problem, equation, answer) triples and verifying
each one using the VerificationOracle.

Key insight: The pairing (word_problem → equation) is correct BY CONSTRUCTION
because we generate both from the same template. The oracle then verifies
the arithmetic (equation → answer) is correct.

This two-tier verification ensures:
1. Word problem and equation match (by construction)
2. Equation and answer match (by oracle verification)

Together: training data is GROUND-TRUTH VERIFIED.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

from .oracle import VerificationOracle
from .types import VerifiedSample


class SafeSelfPlayGenerator:
    """Generate oracle-verified training data via self-play.

    The generator uses templates to create word problem / equation pairs,
    then verifies each one using the oracle before including it in the
    training set.

    Example:
        >>> oracle = VerificationOracle(model, tokenizer)
        >>> generator = SafeSelfPlayGenerator(oracle)
        >>> samples = generator.generate_verified(100)
        >>> print(f"Generated {len(samples)} verified samples")
    """

    # Addition templates: (word_problem_template, equation_template)
    ADDITION_TEMPLATES: ClassVar[List[Tuple[str, str]]] = [
        ("I have {a} apples. I get {b} more. Total:", "{a}+{b}="),
        ("{a} birds. {b} more arrive. Total:", "{a}+{b}="),
        ("Start with {a}. Add {b}. Result:", "{a}+{b}="),
        ("There are {a} cats. {b} more come. Total:", "{a}+{b}="),
        ("{a} toys plus {b} toys. Sum:", "{a}+{b}="),
        ("Mary has {a} candies. She gets {b} more. Total:", "{a}+{b}="),
        ("{a} books. Buy {b} more. Now:", "{a}+{b}="),
        ("Begin with {a}. Increase by {b}. Result:", "{a}+{b}="),
    ]

    # Subtraction templates
    SUBTRACTION_TEMPLATES: ClassVar[List[Tuple[str, str]]] = [
        ("{a} apples. {b} eaten. Remaining:", "{a}-{b}="),
        ("{a} birds. {b} fly away. Left:", "{a}-{b}="),
        ("Start with {a}. Take away {b}. Remaining:", "{a}-{b}="),
        ("There are {a} cats. {b} leave. Left:", "{a}-{b}="),
        ("{a} toys. Give away {b}. Remaining:", "{a}-{b}="),
        ("Tom has {a} candies. He gives {b} away. Left:", "{a}-{b}="),
        ("{a} books. Lose {b}. Now:", "{a}-{b}="),
        ("Begin with {a}. Decrease by {b}. Result:", "{a}-{b}="),
    ]

    def __init__(
        self,
        oracle: VerificationOracle,
        addition_templates: Optional[List[Tuple[str, str]]] = None,
        subtraction_templates: Optional[List[Tuple[str, str]]] = None,
    ):
        """Initialize generator.

        Args:
            oracle: VerificationOracle for checking samples
            addition_templates: Custom addition templates (optional)
            subtraction_templates: Custom subtraction templates (optional)
        """
        self.oracle = oracle
        self._addition_templates = (
            addition_templates
            if addition_templates is not None
            else self.ADDITION_TEMPLATES
        )
        self._subtraction_templates = (
            subtraction_templates
            if subtraction_templates is not None
            else self.SUBTRACTION_TEMPLATES
        )

    def generate_verified(
        self,
        n_samples: int,
        seed: int = 42,
        max_attempts_multiplier: int = 3,
    ) -> List[VerifiedSample]:
        """Generate n verified training samples.

        Samples are generated using templates and verified by the oracle.
        Any sample that fails verification is rejected.

        Args:
            n_samples: Number of verified samples to generate
            seed: Random seed for reproducibility
            max_attempts_multiplier: How many times n_samples to attempt
                                     before giving up

        Returns:
            List of verified samples (may be less than n_samples if
            verification fails too often)
        """
        random.seed(seed)
        verified: List[VerifiedSample] = []
        max_attempts = n_samples * max_attempts_multiplier

        attempts = 0
        while len(verified) < n_samples and attempts < max_attempts:
            attempts += 1

            # Random numbers
            a = random.randint(2, 9)  # random.randint is inclusive
            b = random.randint(1, min(a - 1, 8))  # Ensure b < a for subtraction

            # Choose operation
            if random.random() > 0.5:
                templates = self._addition_templates
                expected = str(a + b)
            else:
                templates = self._subtraction_templates
                expected = str(a - b)

            # Pick random template
            word_template, eq_template = templates[
                random.randint(0, len(templates) - 1)
            ]
            word_problem = word_template.format(a=a, b=b)
            equation = eq_template.format(a=a, b=b)

            # Oracle verification
            is_correct, computed = self.oracle.verify(equation, expected)

            if is_correct:
                verified.append(
                    VerifiedSample(
                        input_text=word_problem,
                        output_text=equation,
                        answer=expected,
                        oracle_computed=computed,
                    )
                )

        return verified

    def to_training_format(
        self,
        samples: List[VerifiedSample],
    ) -> List[Dict[str, str]]:
        """Convert samples to MLX-LM training format.

        Args:
            samples: List of verified samples

        Returns:
            List of dicts with "prompt" and "completion" keys
        """
        return [s.to_training_format() for s in samples]

    def save_jsonl(
        self,
        samples: List[VerifiedSample],
        path: Path,
    ) -> None:
        """Save training data to JSONL file.

        Args:
            samples: List of verified samples
            path: Output path for JSONL file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        training_data = self.to_training_format(samples)
        with open(path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

    def get_statistics(
        self,
        samples: List[VerifiedSample],
    ) -> Dict[str, int]:
        """Get statistics about generated samples.

        Args:
            samples: List of verified samples

        Returns:
            Dictionary with counts
        """
        addition_count = sum(1 for s in samples if "+" in s.output_text)
        subtraction_count = sum(1 for s in samples if "-" in s.output_text)

        return {
            "total": len(samples),
            "addition": addition_count,
            "subtraction": subtraction_count,
        }


__all__ = ["SafeSelfPlayGenerator"]
