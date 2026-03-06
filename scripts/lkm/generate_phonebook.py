# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""PhoneBook dataset generator for LKM validation protocol.

Generates deterministic synthetic name-phone QA pairs, sliced by cumulative
token count. Used as a memorization benchmark for LoRA Knowledge Memory
capacity validation.

Usage:
    poetry run python scripts/lkm/generate_phonebook.py \\
        --model /path/to/model \\
        --seed 42 \\
        --n-pairs 800 \\
        --output-dir data/lkm \\
        --token-sizes 1000,2000,4000,8000,12000,16000,20000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

# --- Name pools ---
# Common first names (40 entries) x common last names (25 entries) = 1000
# combinations, supporting at least 800 unique names.

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John",
    "Jennifer", "Michael", "Linda", "David", "Elizabeth",
    "William", "Barbara", "Richard", "Susan", "Joseph",
    "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Charles", "Lisa", "Daniel", "Betty", "Matthew",
    "Margaret", "Anthony", "Sandra", "Mark", "Ashley",
    "Donald", "Dorothy", "Steven", "Kimberly", "Paul",
    "Emily", "Andrew", "Donna", "Joshua", "Carol",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris",
]

# 40 x 25 = 1000 possible combinations


class Tokenizer(Protocol):
    """Minimal tokenizer protocol: encode text to token IDs."""

    def encode(self, text: str) -> list[int]: ...


def generate_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    """Generate n unique (full_name, phone_number) pairs.

    Names are drawn from common first x last name combinations (no real
    entities). Phone numbers use format XXX-XXX-XXXX with random digits.
    Deterministic: same seed produces identical output.

    Args:
        n: Number of pairs to generate.
        seed: RNG seed for deterministic generation.

    Returns:
        List of (full_name, phone_number) tuples.

    Raises:
        ValueError: If n exceeds the available name pool (1000).
    """
    max_names = len(FIRST_NAMES) * len(LAST_NAMES)
    if n > max_names:
        raise ValueError(
            f"Requested {n} pairs but name pool only has {max_names} "
            f"combinations ({len(FIRST_NAMES)} first x {len(LAST_NAMES)} last)"
        )

    rng = random.Random(seed)

    # Generate all possible name combinations, then sample n
    all_names = [
        f"{first} {last}"
        for first in FIRST_NAMES
        for last in LAST_NAMES
    ]
    rng.shuffle(all_names)
    selected_names = all_names[:n]

    # Generate phone numbers
    pairs: list[tuple[str, str]] = []
    for name in selected_names:
        digits = [str(rng.randint(0, 9)) for _ in range(10)]
        phone = f"{''.join(digits[:3])}-{''.join(digits[3:6])}-{''.join(digits[6:])}"
        pairs.append((name, phone))

    return pairs


def format_qa_text(name: str, phone: str) -> str:
    """Format a name-phone pair as a QA training example.

    Args:
        name: Full name (e.g. "Alice Smith").
        phone: Phone number (e.g. "555-123-4567").

    Returns:
        Formatted QA string.
    """
    return f"Question: What is the phone number of {name}? Answer: {phone}"


def format_eval_prompt(name: str) -> str:
    """Format a name as an evaluation prompt (no answer).

    Args:
        name: Full name (e.g. "Alice Smith").

    Returns:
        Formatted prompt string ending with "Answer:".
    """
    return f"Question: What is the phone number of {name}? Answer:"


def slice_pairs_by_tokens(
    pairs: list[tuple[str, str]],
    target_tokens: list[int],
    tokenizer: Tokenizer,
) -> list[dict[str, Any]]:
    """Slice pairs into subsets by cumulative token count.

    Pairs are added in order until the cumulative token count would exceed
    the target. Smaller slices are strict prefixes of larger slices by
    construction (we iterate pairs once and record cutpoints).

    Args:
        pairs: Ordered list of (name, phone) tuples.
        target_tokens: List of target token counts (e.g. [1000, 2000, 4000]).
        tokenizer: Object with .encode(text) -> list[int] method.

    Returns:
        List of dicts, one per target, each containing:
            - target_tokens: the requested token budget
            - actual_tokens: cumulative tokens used
            - n_pairs: number of pairs included
            - pairs: list of (name, phone) tuples (prefix of input pairs)
    """
    sorted_targets = sorted(target_tokens)

    # Precompute token counts for each pair's QA text
    token_counts: list[int] = []
    for name, phone in pairs:
        text = format_qa_text(name, phone)
        token_counts.append(len(tokenizer.encode(text)))

    # Build cumulative sums and find cutpoints
    cumulative = 0
    # For each target, find how many pairs fit
    target_idx = 0
    results: list[dict[str, Any]] = []

    for i, tc in enumerate(token_counts):
        next_cumulative = cumulative + tc

        # Check if adding this pair would exceed current target(s)
        while target_idx < len(sorted_targets) and next_cumulative > sorted_targets[target_idx]:
            results.append({
                "target_tokens": sorted_targets[target_idx],
                "actual_tokens": cumulative,
                "n_pairs": i,
                "pairs": list(pairs[:i]),
            })
            target_idx += 1

        if target_idx >= len(sorted_targets):
            break

        cumulative = next_cumulative

    # Fill remaining targets (all pairs fit or we ran out of pairs)
    while target_idx < len(sorted_targets):
        results.append({
            "target_tokens": sorted_targets[target_idx],
            "actual_tokens": cumulative,
            "n_pairs": len(pairs) if cumulative > 0 else 0,
            "pairs": list(pairs) if cumulative > 0 else [],
        })
        target_idx += 1

    # Re-order results to match input target_tokens order
    target_to_result = {r["target_tokens"]: r for r in results}
    return [target_to_result[t] for t in target_tokens]


def main() -> None:
    """CLI entry point for PhoneBook dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate PhoneBook memorization dataset for LKM validation."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model (used to load tokenizer).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generation seed (default: 42).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=800,
        help="Total source pairs to generate (default: 800).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lkm",
        help="Output directory (default: data/lkm).",
    )
    parser.add_argument(
        "--token-sizes",
        type=str,
        default="1000,2000,4000,8000,12000,16000,20000",
        help="Comma-separated token sizes (default: 1000,2000,4000,8000,12000,16000,20000).",
    )

    args = parser.parse_args()
    token_sizes = [int(s.strip()) for s in args.token_sizes.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer from model
    from mlx_lm import load as mlx_load
    _, tokenizer = mlx_load(args.model)

    # Generate pairs
    pairs = generate_pairs(args.n_pairs, seed=args.seed)

    # Write master CSV
    csv_path = output_dir / "phonebook_source.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "phone"])
        for name, phone in pairs:
            writer.writerow([name, phone])
    print(f"Wrote {len(pairs)} pairs to {csv_path}")

    # Slice by token count
    slices = slice_pairs_by_tokens(pairs, token_sizes, tokenizer)

    # Write per-size training JSONL files
    for s in slices:
        size = s["target_tokens"]
        jsonl_path = output_dir / f"phonebook_{size}tok.jsonl"
        with open(jsonl_path, "w") as f:
            for name, phone in s["pairs"]:
                line = json.dumps({"text": format_qa_text(name, phone)})
                f.write(line + "\n")
        print(f"Wrote {s['n_pairs']} pairs ({s['actual_tokens']} tokens) to {jsonl_path}")

    # Write eval JSONL (all pairs)
    eval_path = output_dir / "phonebook_eval.jsonl"
    with open(eval_path, "w") as f:
        for name, phone in pairs:
            line = json.dumps({
                "name": name,
                "phone": phone,
                "prompt": format_eval_prompt(name),
            })
            f.write(line + "\n")
    print(f"Wrote {len(pairs)} eval prompts to {eval_path}")

    # Write metadata JSON
    meta = {
        "total_pairs": len(pairs),
        "tokenizer": args.model,
        "slices": {
            str(s["target_tokens"]): {
                "n_pairs": s["n_pairs"],
                "actual_tokens": s["actual_tokens"],
            }
            for s in slices
        },
        "generation_seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = output_dir / "phonebook_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
