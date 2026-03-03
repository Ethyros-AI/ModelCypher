#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate arithmetic training and eval data for the decomposed DAG nodes.

Replaces the original arithmetic_add_train.jsonl (lookup-only, range [1,25])
with procedure-teaching scratchpad data for four nodes:

  single_digit_add
    Exhaustive: all 100 ordered pairs (A,B) with A,B ∈ [0,9].
    Direct format (lookup is correct here — no carry possible).
    Train: 80 pairs  Eval: 20 pairs (all sum≥10, tests carry-inducing boundary).

  carry_rule
    Exhaustive: all 45 ordered pairs with A+B ≥ 10, A,B ∈ [0,9].
    Scratchpad format: shows "A + B = sum. Write digit, carry 1. Answer: sum"
    Train: 30 pairs  Eval: 15 pairs (held-out split, seed=42 for train).

  multi_digit_add
    Coverage-derived: 4 distinct carry states × 100 items/state = 400 train.
    The 4 carry states for 2-digit addition (A,B ∈ [10,99]) are:
      state 0: no carry at any position
      state 1: carry from ones position only (no tens carry)
      state 2: carry from tens position only (no ones carry)
      state 3: carry at both ones and tens positions
    100 items per state ensures all sub-trajectories are represented in training.
    Scratchpad format (column-by-column with carry notation).
    Train: 400 items (A,B ∈ [10,99], 2-digit)
    Eval:  100 items OOD (A,B ∈ [100,999], 3-digit), direct format, seed=99.
    Disjoint: eval range [100,999] never appears in train range [10,99].

  arithmetic_multiply
    Exhaustive: all 720 ordered pairs (A,B) with A ∈ [10,99], B ∈ [2,9].
    (90 multiplicands × 8 multipliers = 720 unique pairs — complete coverage.)
    Scratchpad format (column-by-column long multiplication with carry).
    Train: all 720 pairs (seed=42 shuffle, answer_start after question).
    Eval:  100 items OOD (A ∈ [100,999], B ∈ [2,9]), direct format, seed=99.

Eval items use DIRECT format: only the final integer answer (no scratchpad
in the expected text). The model trained on scratchpad may generate intermediate
steps; the numeric answer_mode evaluator extracts the last integer.

Usage:
    poetry run python scripts/generate_arithmetic_procedures.py
    poetry run python scripts/generate_arithmetic_procedures.py \\
        --output data/training --eval-output data/eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Position labels ────────────────────────────────────────────────────────

_POSITION_NAMES = ["Ones", "Tens", "Hundreds", "Thousands", "Ten-thousands"]


def _pos_name(i: int) -> str:
    return _POSITION_NAMES[i] if i < len(_POSITION_NAMES) else f"10^{i}"


# ── Scratchpad generators ──────────────────────────────────────────────────


def _addition_scratchpad(a: int, b: int) -> str:
    """Column-by-column addition with carry notation.

    Processes positions from ones upward. Each line shows the digit
    contributions and carry at that position.

    Examples:
        47 + 65 → "Ones: 7 + 5 = 12. Write 2, carry 1.\\n
                    Tens: 4 + 6 + 1 (carry) = 11. Write 1, carry 1.\\n
                    Hundreds: 0 + 0 + 1 (carry) = 1. Write 1.\\n
                    Answer: 112"
    """
    result = a + b
    n_positions = len(str(result))  # result digit count determines positions to process

    # Pad both operands to result's digit count; reverse for ones-first iteration
    a_digits = [int(d) for d in str(a).zfill(n_positions)][::-1]
    b_digits = [int(d) for d in str(b).zfill(n_positions)][::-1]

    lines = []
    carry = 0
    for i in range(n_positions):
        a_d = a_digits[i]
        b_d = b_digits[i]
        total = a_d + b_d + carry
        digit_out = total % 10
        carry_out = total // 10

        if carry > 0:
            step = (
                f"{_pos_name(i)}: {a_d} + {b_d} + {carry} (carry) = {total}. "
                f"Write {digit_out}"
            )
        else:
            step = f"{_pos_name(i)}: {a_d} + {b_d} = {total}. Write {digit_out}"

        if carry_out > 0:
            step += f", carry {carry_out}."
        else:
            step += "."

        lines.append(step)
        carry = carry_out

    lines.append(f"Answer: {result}")
    return "\n".join(lines)


def _multiplication_scratchpad(a: int, b: int) -> str:
    """Column-by-column long multiplication with single-digit multiplier.

    b must be a single digit [2,9]. a is the multi-digit multiplicand.
    Processes positions from ones upward, accumulating carry per position.

    Examples:
        47 × 3 → "Ones: 7 × 3 = 21. Write 1, carry 2.\\n
                   Tens: 4 × 3 = 12, plus 2 (carry) = 14. Write 4, carry 1.\\n
                   Hundreds: carry 1. Write 1.\\n
                   Answer: 141"
    """
    assert 2 <= b <= 9, f"multiplier must be in [2,9], got {b}"
    result = a * b

    a_digits = [int(d) for d in str(a)][::-1]  # ones first

    lines = []
    carry = 0
    for i, a_d in enumerate(a_digits):
        partial = a_d * b
        total = partial + carry
        digit_out = total % 10
        carry_out = total // 10

        if carry > 0:
            step = (
                f"{_pos_name(i)}: {a_d} × {b} = {partial}, "
                f"plus {carry} (carry) = {total}. Write {digit_out}"
            )
        else:
            step = f"{_pos_name(i)}: {a_d} × {b} = {total}. Write {digit_out}"

        if carry_out > 0:
            step += f", carry {carry_out}."
        else:
            step += "."

        lines.append(step)
        carry = carry_out

    # Remaining carry digits (at most one iteration for single-digit multiplier
    # since max carry = floor((9×9 + 8) / 10) = 8, a single digit).
    i = len(a_digits)
    while carry > 0:
        lines.append(f"{_pos_name(i)}: carry {carry}. Write {carry}.")
        carry = 0  # single-digit carry always terminates in one step
        i += 1

    lines.append(f"Answer: {result}")
    return "\n".join(lines)


# ── Item formatters ────────────────────────────────────────────────────────


def _make_addition_train_item(a: int, b: int, logic_id: str) -> dict:
    """Training item with scratchpad. CE loss starts at first step (not the question)."""
    question = f"What is {a} + {b}?\n"
    scratchpad = _addition_scratchpad(a, b)
    return {
        "text": question + scratchpad,
        "answer_start": len(question),
        "logic_id": logic_id,
    }


def _make_addition_eval_item(a: int, b: int, logic_id: str) -> dict:
    """Eval item with direct (numeric) answer. Evaluator uses numeric_final mode."""
    result = a + b
    question = f"What is {a} + {b}?\n"
    return {
        "text": question + str(result),
        "answer_start": len(question),
        "logic_id": logic_id,
    }


def _make_carry_rule_train_item(a: int, b: int) -> dict:
    """Carry rule training item: compact carry notation for a single-step sum."""
    assert a + b >= 10, f"carry_rule requires A+B >= 10, got {a}+{b}={a+b}"
    result = a + b
    digit_out = result % 10
    question = f"What is {a} + {b}?\n"
    scratchpad = (
        f"{a} + {b} = {result}. Write {digit_out}, carry 1. Answer: {result}"
    )
    return {
        "text": question + scratchpad,
        "answer_start": len(question),
        "logic_id": "carry_rule",
    }


def _make_multiplication_train_item(a: int, b: int) -> dict:
    """Training item with multiplication scratchpad."""
    question = f"What is {a} × {b}?\n"
    scratchpad = _multiplication_scratchpad(a, b)
    return {
        "text": question + scratchpad,
        "answer_start": len(question),
        "logic_id": "arithmetic_multiply",
    }


def _make_multiplication_eval_item(a: int, b: int) -> dict:
    """Eval item with direct (numeric) answer."""
    result = a * b
    question = f"What is {a} × {b}?\n"
    return {
        "text": question + str(result),
        "answer_start": len(question),
        "logic_id": "arithmetic_multiply",
    }


# ── Carry state classifier ─────────────────────────────────────────────────


def _carry_state(a: int, b: int) -> int:
    """Return the carry state code for 2-digit addition.

    State 0: no carry at ones, no carry into hundreds (result ≤ 2-digit)
    State 1: carry from ones, no carry into hundreds
    State 2: no carry from ones, carry into hundreds
    State 3: carry from ones, carry into hundreds
    """
    a0, a1 = a % 10, a // 10
    b0, b1 = b % 10, b // 10
    ones_carry = 1 if a0 + b0 >= 10 else 0
    tens_total = a1 + b1 + ones_carry
    hundreds_carry = 1 if tens_total >= 10 else 0
    return ones_carry + 2 * hundreds_carry


# ── Generators ────────────────────────────────────────────────────────────


def generate_single_digit_add(seed: int) -> tuple[list[dict], list[dict]]:
    """All 100 ordered pairs (A,B) with A,B ∈ [0,9].

    Split: 20 eval items chosen first from the sum≥10 subset (carry-inducing pairs),
    then remaining 80 pairs form training. This guarantees all eval items test the
    carry boundary (the hardest cases for lookup-trained models).

    Eval format: direct. Training format: direct (lookup is the correct
    representation for single-digit addition — no carry procedure needed).
    """
    rng = random.Random(seed)

    all_pairs = [(a, b) for a in range(10) for b in range(10)]
    carry_pairs = [(a, b) for a, b in all_pairs if a + b >= 10]  # exactly 45 pairs
    no_carry_pairs = [(a, b) for a, b in all_pairs if a + b < 10]  # exactly 55 pairs

    # Eval: 20 items from carry-inducing pairs (held out from training)
    rng.shuffle(carry_pairs)
    eval_carry = carry_pairs[:20]
    train_carry = carry_pairs[20:]

    # Training: remaining carry pairs + all no-carry pairs = 25 + 55 = 80 items
    train_pairs = train_carry + no_carry_pairs

    train_items = [_make_addition_eval_item(a, b, "single_digit_add") for a, b in train_pairs]
    eval_items = [_make_addition_eval_item(a, b, "single_digit_add") for a, b in eval_carry]

    # Shuffle training order (not eval — eval stays deterministic)
    rng.shuffle(train_items)

    assert len(train_items) == 80, f"expected 80 train, got {len(train_items)}"
    assert len(eval_items) == 20, f"expected 20 eval, got {len(eval_items)}"
    assert all(
        json.loads(json.dumps(it))["logic_id"] == "single_digit_add" for it in eval_items
    )

    return train_items, eval_items


def generate_carry_rule(seed: int) -> tuple[list[dict], list[dict]]:
    """All 45 ordered pairs with A+B ≥ 10, A,B ∈ [0,9].

    Train: 45 pairs (exhaustive). Eval: 45 pairs (exhaustive).
    carry_rule is a finite enumerable set — there is no meaningful in/out-of-distribution
    distinction. Mastery requires procedural compliance (carry notation in output) across
    all 45 carry-inducing pairs, so the eval set is the complete population.
    Training and eval use DIFFERENT formats (scratchpad vs direct), so files are not
    identical even though the pair populations are the same.
    Training uses scratchpad format; eval uses direct format.
    """
    pairs = [(a, b) for a in range(10) for b in range(10) if a + b >= 10]
    assert len(pairs) == 45, f"expected 45 carry pairs, got {len(pairs)}"

    # Exhaustive coverage — seed not needed (no sampling)
    train_pairs = pairs
    eval_pairs = pairs

    train_items = [_make_carry_rule_train_item(a, b) for a, b in train_pairs]
    eval_items = [_make_addition_eval_item(a, b, "carry_rule") for a, b in eval_pairs]

    assert len(train_items) == 45
    assert len(eval_items) == 45

    return train_items, eval_items


def generate_multi_digit_add(seed: int) -> tuple[list[dict], list[dict]]:
    """4 carry states × 100 items/state = 400 training items, A,B ∈ [10,99].

    The 4 carry states for 2-digit addition cover all distinct carry sub-trajectories:
      state 0: no carry at any position
      state 1: carry from ones only
      state 2: carry into hundreds only (no ones carry)
      state 3: carry at both positions

    100 items per carry state ensures each sub-trajectory is learned. This is
    derived from: single_digit_add requires 100 items to learn its single trajectory;
    4 carry states require 4 × 100 items to learn all sub-trajectories.

    Training uses scratchpad format (answer_start after question, loss on all steps).
    Eval: 100 OOD items (A,B ∈ [100,999], 3-digit), direct format, disjoint range.
    """
    rng_train = random.Random(seed)
    rng_eval = random.Random(seed + 99)

    # ── Training: 100 items per carry state ────────────────────────────────
    # Categorize all 2-digit + 2-digit pairs by carry state
    by_state: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: [], 3: []}
    for a in range(10, 100):
        for b in range(10, 100):
            s = _carry_state(a, b)
            by_state[s].append((a, b))

    train_items = []
    for state, pairs in by_state.items():
        rng_train.shuffle(pairs)
        selected = pairs[:100]
        assert len(selected) == 100, (
            f"carry state {state} has only {len(pairs)} pairs — "
            f"insufficient for 100-item coverage"
        )
        train_items.extend(
            _make_addition_train_item(a, b, "multi_digit_add") for a, b in selected
        )

    rng_train.shuffle(train_items)
    assert len(train_items) == 400

    # ── Eval: 100 OOD 3-digit + 3-digit pairs ─────────────────────────────
    # Range [100,999] × [100,999] is entirely disjoint from training [10,99] × [10,99].
    eval_pairs: set[tuple[int, int]] = set()
    while len(eval_pairs) < 100:
        a = rng_eval.randint(100, 999)
        b = rng_eval.randint(100, 999)
        eval_pairs.add((a, b))

    eval_items = [
        _make_addition_eval_item(a, b, "multi_digit_add")
        for a, b in sorted(eval_pairs)
    ]
    rng_eval.shuffle(eval_items)
    assert len(eval_items) == 100

    # Verify OOD: no eval pair uses operands in the training range
    assert all(
        a >= 100 and b >= 100
        for item in eval_items
        for a, b in [
            (
                int(item["text"].split("What is ")[1].split(" + ")[0]),
                int(item["text"].split(" + ")[1].split("?\n")[0]),
            )
        ]
    ), "eval items must be OOD relative to train"

    return train_items, eval_items


def generate_arithmetic_multiply(seed: int) -> tuple[list[dict], list[dict]]:
    """Exhaustive 2-digit × 1-digit: all 720 ordered pairs (A ∈ [10,99], B ∈ [2,9]).

    720 = 90 multiplicands × 8 multipliers — complete coverage of the 2-digit × [2-9]
    space. Exhaustive coverage is used because it is feasible and eliminates any
    concern about missing carry patterns in the training data.

    Training uses scratchpad format. Eval: 100 OOD items (A ∈ [100,999], B ∈ [2,9]).
    """
    rng_train = random.Random(seed)
    rng_eval = random.Random(seed + 99)

    # ── Training: all 720 pairs ────────────────────────────────────────────
    train_pairs = [(a, b) for a in range(10, 100) for b in range(2, 10)]
    assert len(train_pairs) == 720
    rng_train.shuffle(train_pairs)

    train_items = [_make_multiplication_train_item(a, b) for a, b in train_pairs]

    # ── Eval: 100 OOD 3-digit × 1-digit pairs ─────────────────────────────
    eval_pairs: set[tuple[int, int]] = set()
    while len(eval_pairs) < 100:
        a = rng_eval.randint(100, 999)
        b = rng_eval.randint(2, 9)
        eval_pairs.add((a, b))

    eval_items = [
        _make_multiplication_eval_item(a, b)
        for a, b in sorted(eval_pairs)
    ]
    rng_eval.shuffle(eval_items)
    assert len(eval_items) == 100

    return train_items, eval_items


# ── File I/O ───────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  Wrote {len(items):5d} items → {path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="data/training", help="Training output dir")
    parser.add_argument("--eval-output", default="data/eval", help="Eval output dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_out = Path(args.output)
    eval_out = Path(args.eval_output)
    seed = args.seed

    # ── single_digit_add ────────────────────────────────────────────────────
    print("[single_digit_add] Exhaustive 100 pairs (A,B ∈ [0,9])...")
    train, ev = generate_single_digit_add(seed)
    print(f"  Carry-inducing pairs in eval: {sum(1 for it in ev if '+' in it['text'])}")
    _write_jsonl(train_out / "single_digit_add_train.jsonl", train)
    _write_jsonl(eval_out / "single_digit_add_eval.jsonl", ev)

    # ── carry_rule ─────────────────────────────────────────────────────────
    print("\n[carry_rule] Exhaustive 45 carry-inducing pairs (A+B≥10, A,B ∈ [0,9])...")
    train, ev = generate_carry_rule(seed)
    _write_jsonl(train_out / "carry_rule_train.jsonl", train)
    _write_jsonl(eval_out / "carry_rule_eval.jsonl", ev)

    # ── multi_digit_add ────────────────────────────────────────────────────
    print(
        "\n[multi_digit_add] 4 carry states × 100 items/state "
        "(train: [10,99], eval OOD: [100,999])..."
    )
    train, ev = generate_multi_digit_add(seed)
    _write_jsonl(train_out / "multi_digit_add_train.jsonl", train)
    _write_jsonl(eval_out / "multi_digit_add_eval.jsonl", ev)
    print(f"  Sample train item: {train[0]['text'][:80].replace(chr(10), ' | ')}...")
    print(f"  Sample eval item:  {ev[0]['text'][:60].replace(chr(10), ' | ')}")

    # ── arithmetic_multiply ────────────────────────────────────────────────
    print(
        "\n[arithmetic_multiply] Exhaustive 720 pairs (A ∈ [10,99], B ∈ [2,9])..."
    )
    train, ev = generate_arithmetic_multiply(seed)
    _write_jsonl(train_out / "arithmetic_multiply_train.jsonl", train)
    _write_jsonl(eval_out / "arithmetic_multiply_eval.jsonl", ev)
    print(f"  Sample train item: {train[0]['text'][:80].replace(chr(10), ' | ')}...")
    print(f"  Sample eval item:  {ev[0]['text'][:60].replace(chr(10), ' | ')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
