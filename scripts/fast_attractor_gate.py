#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Fast attractor gate: test only the 8 prompts that degenerate in both v3 and v4.
# If any of these degenerate, the run fails immediately (no need for full 46).
#
# Usage:
#   poetry run python scripts/fast_attractor_gate.py \
#     --model /path/to/model --adapter /path/to/adapter

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The 8 attractor prompts (indices 19,20,21,22,29,32,34,36 in full eval)
ATTRACTOR_PROMPTS = [
    ("math_7x8", "What is 7 * 8?"),
    ("math_shirt", "If a shirt costs $25 and is 20% off, what is the sale price?"),
    ("math_bat_ball", "A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?"),
    ("math_sequence", "What is the next number in the sequence: 2, 6, 18, 54, ?"),
    ("gk_capital", "What is the capital of France?"),
    ("gk_romeo", "Who wrote Romeo and Juliet?"),
    ("gk_ocean", "What is the largest ocean on Earth?"),
    ("gk_water", "What is the chemical formula for water?"),
]


def is_degenerate(response: str, threshold: float = 0.2) -> bool:
    """Check if response is degenerate via 4-gram repetition."""
    words = response.split()
    if len(words) < 8:
        return False
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams) > threshold


def run_single(model: str, prompt: str, adapter: str | None = None) -> str:
    """Run inference on a single prompt, return raw stdout."""
    cmd = ["poetry", "run", "mc", "infer", "run", "--model", model, "--prompt", prompt]
    if adapter:
        cmd.extend(["--adapter", adapter])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"


def main():
    parser = argparse.ArgumentParser(description="Fast attractor gate (8 prompts)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()

    degenerate_count = 0
    results = []

    for name, prompt in ATTRACTOR_PROMPTS:
        response = run_single(args.model, prompt, args.adapter)
        degen = is_degenerate(response)
        status = "DEGEN" if degen else "OK"
        if degen:
            degenerate_count += 1

        # Extract token count from response if possible
        n_words = len(response.split())
        results.append({"name": name, "status": status, "words": n_words})
        logger.info("  %s  %-20s  (%d words)", status, name, n_words)

    logger.info("")
    logger.info("Attractor gate: %d/8 degenerate", degenerate_count)

    if degenerate_count > 0:
        logger.info("FAIL — attractor degeneration detected, skip full eval")
        sys.exit(1)
    else:
        logger.info("PASS — no attractor degeneration, proceed to full eval")
        sys.exit(0)


if __name__ == "__main__":
    main()
