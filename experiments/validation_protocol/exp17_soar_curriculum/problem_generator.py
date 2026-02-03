# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 17: SOAR Curriculum - Problem Generator
#
# Generates arithmetic chain problems with controlled structural quality.
# Based on SOAR paper insight: "Structural quality matters more than correctness."

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Operation = Literal["+", "-", "*", "/"]


@dataclass
class ArithmeticChainProblem:
    """A stepping-stone arithmetic chain problem."""

    problem_id: str
    depth: int  # Number of reasoning steps (1-3)
    chain: list[tuple[int, Operation, int, int]]  # (left, op, right, result)
    prompt: str  # Natural language prompt
    answer: int  # Final answer

    # Structural properties (filled in later by metrics)
    fisher_compatibility: float = 0.0
    manifold_curvature: float = 0.0
    curvature_variance: float = 0.0
    barrier_height: float = 0.0

    # Training outcome (filled in after training)
    final_loss: float = 0.0
    perplexity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_training_format(self) -> dict:
        """Convert to JSONL training format."""
        return {"text": f"{self.prompt} {self.answer}"}


def _safe_divide(a: int, b: int) -> int | None:
    """Integer division only if divisible."""
    if b == 0:
        return None
    if a % b != 0:
        return None
    return a // b


def _apply_op(left: int, op: Operation, right: int) -> int | None:
    """Apply operation, return None if invalid."""
    if op == "+":
        return left + right
    elif op == "-":
        result = left - right
        return result if result >= 0 else None  # Keep positive
    elif op == "*":
        return left * right
    elif op == "/":
        return _safe_divide(left, right)
    return None


def _generate_step(
    current: int,
    rng: random.Random,
    allow_ops: list[Operation] | None = None,
) -> tuple[int, Operation, int, int] | None:
    """Generate a single arithmetic step from current value."""
    if allow_ops is None:
        allow_ops = ["+", "-", "*", "/"]

    # Try random operations until we find a valid one
    ops_to_try = allow_ops.copy()
    rng.shuffle(ops_to_try)

    for op in ops_to_try:
        # Generate operand
        if op == "+":
            right = rng.randint(1, 20)
            result = current + right
        elif op == "-":
            right = rng.randint(1, min(current - 1, 20)) if current > 1 else None
            if right is None:
                continue
            result = current - right
        elif op == "*":
            right = rng.randint(2, 5)  # Keep products manageable
            result = current * right
        elif op == "/":
            # Find divisors of current
            divisors = [d for d in range(2, min(current, 10) + 1) if current % d == 0]
            if not divisors:
                continue
            right = rng.choice(divisors)
            result = current // right
        else:
            continue

        # Validate result is reasonable
        if 1 <= result <= 1000:
            return (current, op, right, result)

    return None


def generate_chain(
    depth: int,
    rng: random.Random,
    start_range: tuple[int, int] = (5, 20),
) -> list[tuple[int, Operation, int, int]]:
    """Generate a chain of arithmetic operations."""
    chain = []
    current = rng.randint(*start_range)

    for _ in range(depth):
        step = _generate_step(current, rng)
        if step is None:
            # Retry with new starting point
            current = rng.randint(*start_range)
            step = _generate_step(current, rng)
            if step is None:
                # Fallback to simple addition
                right = rng.randint(1, 10)
                step = (current, "+", right, current + right)

        chain.append(step)
        current = step[3]  # Result becomes next left operand

    return chain


def chain_to_prompt(chain: list[tuple[int, Operation, int, int]]) -> str:
    """Convert arithmetic chain to natural language prompt."""
    if len(chain) == 1:
        left, op, right, _ = chain[0]
        op_word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
        return f"What is {left} {op_word} {right}?"

    # Multi-step: build context
    parts = []
    for i, (left, op, right, result) in enumerate(chain[:-1]):
        op_word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
        parts.append(f"{left} {op_word} {right} equals {result}")

    context = ", and ".join(parts) if len(parts) > 1 else parts[0]

    # Final question
    left, op, right, _ = chain[-1]
    op_word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]

    return f"If {context}, what is {left} {op_word} {right}?"


def generate_problems(
    n_problems: int = 100,
    depths: list[int] | None = None,
    seed: int = 42,
) -> list[ArithmeticChainProblem]:
    """Generate a set of arithmetic chain problems.

    Args:
        n_problems: Total number of problems to generate
        depths: List of depths to use (default: [1, 2, 3])
        seed: Random seed for reproducibility

    Returns:
        List of ArithmeticChainProblem instances
    """
    if depths is None:
        depths = [1, 2, 3]

    rng = random.Random(seed)
    problems = []

    problems_per_depth = n_problems // len(depths)
    remainder = n_problems % len(depths)

    for depth in depths:
        n_for_depth = problems_per_depth + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)

        for i in range(n_for_depth):
            chain = generate_chain(depth, rng)
            prompt = chain_to_prompt(chain)
            answer = chain[-1][3]  # Final result

            problem = ArithmeticChainProblem(
                problem_id=f"arith_d{depth}_{i:04d}",
                depth=depth,
                chain=chain,
                prompt=prompt,
                answer=answer,
            )
            problems.append(problem)

    # Shuffle to mix depths
    rng.shuffle(problems)

    logger.info(
        "Generated %d problems: %s",
        len(problems),
        {d: sum(1 for p in problems if p.depth == d) for d in depths},
    )

    return problems


def save_problems(problems: list[ArithmeticChainProblem], output_dir: Path) -> None:
    """Save problems to JSON and JSONL formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full problem data
    with open(output_dir / "problems.json", "w") as f:
        json.dump([p.to_dict() for p in problems], f, indent=2)

    # Training format (just prompt + answer)
    with open(output_dir / "train.jsonl", "w") as f:
        for p in problems:
            json.dump(p.to_training_format(), f)
            f.write("\n")

    logger.info("Saved %d problems to %s", len(problems), output_dir)


def load_problems(input_dir: Path) -> list[ArithmeticChainProblem]:
    """Load problems from JSON."""
    with open(input_dir / "problems.json") as f:
        data = json.load(f)

    return [ArithmeticChainProblem(**p) for p in data]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate test problems
    problems = generate_problems(n_problems=10, seed=42)

    print("\nSample problems:")
    for p in problems[:5]:
        print(f"  [{p.depth}] {p.prompt} → {p.answer}")
        print(f"       Chain: {p.chain}")
