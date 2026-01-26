#!/usr/bin/env python3
"""Fixed evaluation with correct tokenization handling."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYMBOLIC_PROMPTS = [
    ("1+1=", "2"),
    ("2+1=", "3"),
    ("3+1=", "4"),
    ("4+1=", "5"),
    ("5+1=", "6"),
    ("6+1=", "7"),
    ("7+1=", "8"),
    ("8+1=", "9"),
    ("9+1=", "10"),
    ("2+2=", "4"),
    ("3+3=", "6"),
    ("5+5=", "10"),
]


def get_digit_token(tokenizer, digit_str):
    """Get the actual digit token ID, not the startoftext prefix."""
    tokens = tokenizer.encode(digit_str)
    # Skip the startoftext token (usually token 1)
    if len(tokens) > 1 and tokens[0] == 1:  # <|startoftext|>
        return tokens[1]  # The actual digit token
    return tokens[0] if tokens else -1


def evaluate(model, tokenizer):
    """Evaluate with correct token handling."""
    import mlx.core as mx

    logger.info("=" * 60)
    logger.info("FIXED EVALUATION (Correct Tokenization)")
    logger.info("=" * 60)

    results = []
    for prompt, expected in SYMBOLIC_PROMPTS:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist())

        # Top prediction
        top_token = int(np.argmax(logits_np))
        predicted = tokenizer.decode([top_token]).strip()
        correct = expected in predicted or predicted == expected

        # Get the ACTUAL target token (not the startoftext prefix)
        target_id = get_digit_token(tokenizer, expected)

        # Probabilities
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        target_prob = probs[target_id] if target_id >= 0 else 0.0
        target_rank = int((np.argsort(logits_np)[::-1] == target_id).nonzero()[0][0]) if target_id >= 0 else -1

        # Top 5 for debugging
        top5 = np.argsort(logits_np)[-5:][::-1]

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "target_token_id": target_id,
            "target_prob": float(target_prob),
            "target_rank": target_rank,
            "top5_tokens": [int(t) for t in top5],
            "top5_decoded": [tokenizer.decode([t]).strip() for t in top5],
        })

        status = "✓" if correct else "✗"
        logger.info(f"'{prompt}' → '{predicted}' ({status}) "
                   f"target='{expected}'(tok={target_id}) rank={target_rank+1} p={target_prob:.1%}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_rank = np.mean([r["target_rank"] for r in results])

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {accuracy:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
    logger.info(f"Mean target rank: {mean_rank:.1f}")

    # Breakdown by rank
    rank_1 = sum(1 for r in results if r["target_rank"] == 0)
    rank_2_5 = sum(1 for r in results if 1 <= r["target_rank"] < 5)
    rank_6plus = sum(1 for r in results if r["target_rank"] >= 5)

    logger.info(f"\nRank distribution:")
    logger.info(f"  Rank 1 (correct): {rank_1}/{len(results)}")
    logger.info(f"  Rank 2-5: {rank_2_5}/{len(results)}")
    logger.info(f"  Rank 6+: {rank_6plus}/{len(results)}")

    return {
        "accuracy": accuracy,
        "mean_rank": float(mean_rank),
        "details": results,
    }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    results = evaluate(model, tokenizer)

    output_path = "data/experiments/fixed_evaluation.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
