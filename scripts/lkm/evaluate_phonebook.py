# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""PhoneBook exact-match evaluator for LKM validation protocol.

Loads a model+adapter, generates completions for phonebook prompts,
extracts phone numbers, and checks exact match against ground truth.

Usage:
    poetry run python scripts/lkm/evaluate_phonebook.py \\
        --model /path/to/base/model \\
        --adapter /path/to/adapter \\
        --eval-data data/lkm/phonebook_eval.jsonl \\
        --output data/lkm/raw_scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PHONE_PATTERN = re.compile(r"\d{3}-\d{3}-\d{4}")


def extract_phone(text: str) -> str | None:
    """Extract first phone number matching XXX-XXX-XXXX from text.

    Args:
        text: Generated text to search.

    Returns:
        Matched phone string, or None if no match found.
    """
    match = _PHONE_PATTERN.search(text)
    if match is None:
        return None
    return match.group()


def check_exact_match(true_phone: str, generated_text: str) -> bool:
    """Check if the generated text contains the correct phone number.

    Calls extract_phone on generated_text and compares to true_phone.

    Args:
        true_phone: Ground truth phone number (e.g. "555-123-4567").
        generated_text: Model-generated text to check.

    Returns:
        True if extracted phone matches true_phone exactly.
    """
    predicted = extract_phone(generated_text)
    return predicted == true_phone


def compute_score(results: list[dict]) -> float:
    """Compute exact-match accuracy from a list of result dicts.

    Args:
        results: List of dicts, each with an "exact_match" (bool) key.

    Returns:
        Fraction of results where exact_match is True. Returns 0.0 if
        the list is empty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["exact_match"]) / len(results)


def evaluate(
    model_path: str,
    adapter_path: str,
    eval_path: str,
    output_path: str,
    max_tokens: int = 30,
) -> float:
    """Run exact-match evaluation on phonebook prompts.

    Loads model+adapter, generates completions for each eval prompt,
    extracts phone numbers, checks exact match, and writes results.

    Args:
        model_path: Path to the base model.
        adapter_path: Path to the adapter directory.
        eval_path: Path to eval JSONL (each line: {"name", "phone", "prompt"}).
        output_path: Path to write raw_scores.jsonl output.
        max_tokens: Maximum tokens to generate per prompt.

    Returns:
        Exact-match accuracy as a float in [0.0, 1.0].
    """
    from mlx_lm import generate, load as mlx_load

    model, tokenizer = mlx_load(model_path, adapter_path=adapter_path)

    # Read eval data
    eval_items: list[dict] = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    # Evaluate each item
    results: list[dict] = []
    for i, item in enumerate(eval_items):
        generated_text = generate(
            model, tokenizer, prompt=item["prompt"], max_tokens=max_tokens
        )
        predicted_phone = extract_phone(generated_text)
        exact_match = check_exact_match(item["phone"], generated_text)

        results.append({
            "name": item["name"],
            "phone_true": item["phone"],
            "phone_predicted": predicted_phone,
            "exact_match": exact_match,
        })

        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{len(eval_items)}")

    # Write results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    score = compute_score(results)
    print(f"Exact-match accuracy: {score:.4f} ({sum(1 for r in results if r['exact_match'])}/{len(results)})")
    return score


def main() -> None:
    """CLI entry point for PhoneBook evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate phonebook memorization (exact-match)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to base model.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to adapter directory.",
    )
    parser.add_argument(
        "--eval-data",
        required=True,
        help="Path to eval JSONL file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write raw_scores.jsonl.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens to generate per prompt (default: 30).",
    )

    args = parser.parse_args()
    evaluate(
        model_path=args.model,
        adapter_path=args.adapter,
        eval_path=args.eval_data,
        output_path=args.output,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
