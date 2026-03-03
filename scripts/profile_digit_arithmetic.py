#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Profile a model's digit-range generalization ceiling for integer addition.

Runs BEFORE any data changes to establish the pretraining baseline. Measures
where the model's addition ability falls off as digit count increases.

Sample count --n is derived from Clopper-Pearson: n=50 gives ~±14% CI width at
95% confidence. This is sufficient to distinguish zero accuracy (no procedure)
from partial accuracy (lookup + partial procedure) from reliable accuracy.

Usage:
    poetry run python scripts/profile_digit_arithmetic.py \\
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16

    poetry run python scripts/profile_digit_arithmetic.py \\
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \\
        --n 100  # tighter CI: ±10% width
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Digit ranges to profile. Each entry: (lo, hi, label).
# lo/hi are inclusive bounds for EACH operand (not the sum).
_DIGIT_RANGES = [
    (0, 9, "1-digit"),
    (10, 99, "2-digit"),
    (100, 999, "3-digit"),
    (1000, 9999, "4-digit"),
]


class RangeResult(NamedTuple):
    label: str
    lo: int
    hi: int
    n: int
    n_correct: int
    accuracy: float


def _init_backend() -> None:
    """Initialize the default backend if not already initialized."""
    from modelcypher.core.domain._backend import get_default_backend

    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend

        set_default_backend(get_backend(detect_default_backend_type()))


def _extract_last_int(text: str) -> int | None:
    """Return the last integer appearing in text, or None if no integer found."""
    nums = re.findall(r"\b\d+\b", text)
    return int(nums[-1]) if nums else None


def _profile_range(
    engine: object,
    model_path: str,
    lo: int,
    hi: int,
    n: int,
    seed: int,
    label: str,
) -> RangeResult:
    """Run n random addition problems with both operands in [lo, hi]."""
    rng = random.Random(seed)
    n_correct = 0

    for _ in range(n):
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        expected = a + b
        prompt = f"What is {a} + {b}?\n"

        try:
            result = engine.run(model=model_path, prompt=prompt, max_tokens=None)
            predicted = _extract_last_int(result.response)
            if predicted == expected:
                n_correct += 1
        except Exception:
            logger.debug("Inference failed for %d + %d", a, b, exc_info=True)

    accuracy = n_correct / n if n > 0 else 0.0
    return RangeResult(
        label=label, lo=lo, hi=hi, n=n, n_correct=n_correct, accuracy=accuracy
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help=(
            "Problems per digit range (default 50). "
            "n=50 → Clopper-Pearson CI width ≈ ±14%% at 95%%. "
            "n=100 → ±10%%."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for problem generation (default 7).",
    )
    args = parser.parse_args()

    print("Initializing backend...")
    _init_backend()

    from modelcypher.adapters.inference_engine import get_inference_engine

    engine = get_inference_engine()

    print(f"\nProfiling addition generalization on: {args.model}")
    print(f"Problems per range: {args.n}  (seed={args.seed})\n")

    results = []
    for lo, hi, label in _DIGIT_RANGES:
        print(f"  [{label}]  A,B ∈ [{lo}, {hi}] ...", end="", flush=True)
        r = _profile_range(engine, args.model, lo, hi, args.n, seed=args.seed, label=label)
        results.append(r)
        print(f"  {r.n_correct}/{r.n} = {r.accuracy:.1%}")

    print()
    print("=" * 60)
    print(f"  DIGIT-RANGE ADDITION PROFILE")
    print(f"  Model: {args.model}")
    print("=" * 60)
    print(f"  {'Range':<10}  {'Operands':<18}  {'Correct':>8}  {'Accuracy':>10}")
    print("  " + "-" * 54)
    for r in results:
        print(
            f"  {r.label:<10}  [{r.lo}, {r.hi}]{'':<8}  "
            f"{r.n_correct:>5}/{r.n:<3}  {r.accuracy:>9.1%}"
        )
    print("=" * 60)

    # Identify the generalization ceiling: first range below 80% accuracy.
    # 80% is not a decision threshold — it's a presentation marker for the table.
    # The frontier for curriculum training is simply the first range where the model
    # fails (accuracy < 1.0 by the mastery criterion).
    ceiling = None
    for r in results:
        if r.accuracy < 1.0:
            ceiling = r
            break
    if ceiling:
        print(
            f"\n  Generalization ceiling: {ceiling.label} "
            f"(first range with accuracy < 100%)"
        )
        print("  Training frontier starts here.")
    else:
        print("\n  All ranges passed (accuracy = 100%). No training needed for addition.")


if __name__ == "__main__":
    main()
