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

"""Logical Shapes - Fundamental reasoning patterns for training curricula.

Research Finding:
    10 perfect examples > 10,000 mediocre ones.
    For each failing pattern, distill to the FUNDAMENTAL LOGICAL SHAPE.
    Walk the model through reasoning step-by-step so it learns the SHAPE
    of logic, not just surface patterns.

The 6 Fundamental Logical Shapes:
    1. PERCENTAGE_INCREASE: new = original + (original × percent)
       Common error: treating "increased by 150%" as "multiply by 1.5"

    2. AVERAGE_RATE: total_output / total_input (not mean of rates)
       Common error: (rate1 + rate2) / 2 (arithmetic mean is WRONG)

    3. THRESHOLD_CROSSING: breakeven + 1 = first profitable
       Common error: reporting breakeven instead of first profit

    4. INVERSE_CHAIN: work backwards, undo operations in reverse
       Common error: trying to work forwards with unknowns

    5. SEQUENTIAL_OPERATIONS: subtract first, THEN multiply
       Common error: multiplying before subtracting

    6. REMAINING_FIRST: compute what's left BEFORE applying rate
       Common error: applying rates to wrong quantities

Usage:
    from modelcypher.core.domain.training.logical_shapes import (
        LogicalShape,
        LogicalShapeExample,
    )
    from modelcypher.core.domain.training.logical_shapes_patterns import (
        get_logical_shape_examples,
    )

    examples = get_logical_shape_examples()
    for ex in examples:
        print(f"{ex.shape.value}: {ex.question[:50]}...")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LogicalShape(Enum):
    """Fundamental logical structures in word problems.

    These shapes represent the core reasoning patterns that distinguish
    correct problem-solving from common errors. Teaching models to identify
    and apply these shapes improves mathematical reasoning accuracy.
    """

    PERCENTAGE_INCREASE = "percentage_increase"
    """ADD X% of original TO original.

    Example: "Increased by 50%" means new = original + (original × 0.50)
    Common error: treating "increased by 150%" as "multiply by 1.5" (only 50% increase)
    """

    AVERAGE_RATE = "average_rate"
    """Total output / Total input (harmonic mean).

    Example: Average speed = total distance / total time
    Common error: (rate1 + rate2) / 2 (arithmetic mean of rates is WRONG)
    """

    THRESHOLD_CROSSING = "threshold_crossing"
    """First profit = breakeven + 1.

    Example: "When do you START earning?" Answer: the year AFTER breakeven
    Common error: reporting the breakeven point instead of first profitable
    """

    INVERSE_CHAIN = "inverse_chain"
    """Work backwards, undo operations in reverse order.

    Example: "After giving half, she has 5. How many did she start with?"
    Common error: trying to work forwards with unknowns
    """

    SEQUENTIAL_OPERATIONS = "sequential_ops"
    """Order matters - perform operations in correct sequence.

    Example: Subtract first, THEN multiply
    Common error: multiplying before subtracting, or wrong order
    """

    REMAINING_FIRST = "remaining_first"
    """Compute remainder before rate calculation.

    Example: Calculate remaining distance BEFORE dividing by rate
    Common error: applying rates to wrong quantities
    """


@dataclass(frozen=True)
class LogicalShapeExample:
    """Training example for a logical shape.

    Each example contains:
    - The logical shape being demonstrated
    - A word problem question
    - Step-by-step reasoning that explicitly names the shape
    - The correct numerical answer
    - The common error students make (for negative example contrast)

    Attributes:
        shape: The fundamental logical shape this example demonstrates.
        question: The word problem question text.
        reasoning: Step-by-step reasoning with shape identification.
        answer: The correct numerical/text answer.
        common_error: What students typically get wrong and why.
    """

    shape: LogicalShape
    question: str
    reasoning: str
    answer: str
    common_error: str

    @property
    def full_text(self) -> str:
        """Format as complete training text.

        Returns:
            Full training example with question, shape identification,
            reasoning steps, and answer in GSM8K-compatible format.
        """
        return f"""Question: {self.question}

Answer: **LOGICAL SHAPE: {self.shape.value.upper().replace('_', ' ')}**

{self.reasoning}
#### {self.answer}"""

    @property
    def training_dict(self) -> dict:
        """Format for JSONL training data.

        Returns:
            Dict with 'text' key containing the full training text.
        """
        return {"text": self.full_text}

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "shape": self.shape.value,
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "common_error": self.common_error,
        }


def get_shape_description(shape: LogicalShape) -> str:
    """Get a human-readable description of a logical shape.

    Args:
        shape: The logical shape to describe.

    Returns:
        Multi-line description of the shape and its common error.
    """
    descriptions = {
        LogicalShape.PERCENTAGE_INCREASE: (
            "PERCENTAGE INCREASE: new = original + (original × percent)\n"
            '"Increased by X%" means ADD X% of the original TO the original.\n'
            'Common error: treating "increased by 150%" as "multiply by 1.5"'
        ),
        LogicalShape.AVERAGE_RATE: (
            "AVERAGE RATE: Total output / Total input\n"
            "Average rate is NOT the arithmetic mean of individual rates.\n"
            "Common error: (rate1 + rate2) / 2 gives wrong answer"
        ),
        LogicalShape.THRESHOLD_CROSSING: (
            "THRESHOLD CROSSING: first profit = breakeven + 1\n"
            '"When do you START earning?" = the year AFTER you break even.\n'
            "Common error: reporting breakeven point instead of first profitable"
        ),
        LogicalShape.INVERSE_CHAIN: (
            "INVERSE CHAIN: Work backwards, undo operations in reverse\n"
            "Start from what you know, undo operations in reverse order.\n"
            "Common error: trying to work forwards with unknowns"
        ),
        LogicalShape.SEQUENTIAL_OPERATIONS: (
            "SEQUENTIAL OPERATIONS: Order matters\n"
            "Perform operations in the correct sequence (often subtract then multiply).\n"
            "Common error: wrong operation order changes the result"
        ),
        LogicalShape.REMAINING_FIRST: (
            "REMAINING FIRST: Compute remainder before applying rate\n"
            "Calculate what's left BEFORE dividing by rate.\n"
            "Common error: applying rates to the wrong quantity"
        ),
    }
    return descriptions.get(shape, f"Unknown shape: {shape.value}")
