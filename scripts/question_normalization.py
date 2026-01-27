#!/usr/bin/env python3
"""Question Normalization: Force model to first extract the core question.

Hypothesis:
    If φ emerges at ~14 tokens (resonance length), then:
    1. First compress input to "what is the question?" (~14 tokens)
    2. Process at φ resonance point
    3. Then expand to answer

    This ensures the model always operates at its natural φ geometry.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2


def compute_ratio(model, tokenizer, text: str) -> tuple[float, int]:
    """Compute compression ratio for a given text."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)
    peak = float(mx.sqrt(mx.sum(hidden * hidden)))

    for layer in model.model.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        peak = max(peak, norm)

    final = norm
    return peak / final, len(tokens)


def extract_core_question(model, tokenizer, original: str, max_tokens: int = 20) -> str:
    """Have the model extract the core question."""
    prompt = f"""Extract the core question in 10-15 words.

Original: {original}

Core question:"""

    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    # Clean up response
    core = response.strip().split('\n')[0]  # Take first line
    return core


def answer_at_resonance(model, tokenizer, question: str) -> tuple[str, float, float]:
    """Answer a question after normalizing to resonance length.

    Returns (answer, original_ratio, normalized_ratio)
    """
    # Original ratio
    orig_ratio, orig_tokens = compute_ratio(model, tokenizer, question)

    # Extract core question
    core_question = extract_core_question(model, tokenizer, question)
    norm_ratio, norm_tokens = compute_ratio(model, tokenizer, core_question)

    logger.info(f"Original ({orig_tokens} tokens): ratio={orig_ratio:.3f}")
    logger.info(f"Normalized ({norm_tokens} tokens): ratio={norm_ratio:.3f}")
    logger.info(f"Core: {core_question}")

    # Answer the normalized question
    answer_prompt = f"Question: {core_question}\n\nAnswer:"
    answer = generate(model, tokenizer, prompt=answer_prompt, max_tokens=50, verbose=False)

    return answer.strip(), orig_ratio, norm_ratio


def test_normalization():
    """Test question normalization on various inputs."""
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    logger.info(f"Loading: {model_path}")
    model, tokenizer = load(model_path)

    # Test cases with varying complexity
    test_cases = [
        # Long, complex question
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",

        # Verbose question
        "I was wondering if you could help me figure out what happens when you add the number five to the number three?",

        # Already short
        "What is 5 + 3?",

        # Very long with context
        "In the context of basic arithmetic operations commonly taught in elementary school mathematics, if we consider the simple addition of two positive integers, specifically the numbers fifteen and seven, what would be the resulting sum of this mathematical operation?",

        # Medium length
        "If you have 5 apples and someone gives you 3 more apples, how many apples do you have in total?",
    ]

    results = []
    logger.info("\n" + "=" * 70)
    logger.info("QUESTION NORMALIZATION TEST")
    logger.info("=" * 70)

    for question in test_cases:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Q: {question[:60]}...")

        answer, orig_ratio, norm_ratio = answer_at_resonance(model, tokenizer, question)

        improvement = abs(norm_ratio - PHI) < abs(orig_ratio - PHI)
        status = "✓ IMPROVED" if improvement else "✗ no improvement"

        logger.info(f"Original distance from φ: {abs(orig_ratio - PHI):.3f}")
        logger.info(f"Normalized distance from φ: {abs(norm_ratio - PHI):.3f}")
        logger.info(f"Status: {status}")
        logger.info(f"Answer: {answer[:100]}...")

        results.append({
            "question": question,
            "original_ratio": float(orig_ratio),
            "normalized_ratio": float(norm_ratio),
            "original_dist_phi": float(abs(orig_ratio - PHI)),
            "normalized_dist_phi": float(abs(norm_ratio - PHI)),
            "improved": bool(improvement),
            "answer": answer[:200],
        })

    # Summary
    improved_count = sum(1 for r in results if r["improved"])
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Improved: {improved_count}/{len(results)}")
    logger.info(f"φ target: {PHI:.4f}")

    avg_orig = np.mean([r["original_dist_phi"] for r in results])
    avg_norm = np.mean([r["normalized_dist_phi"] for r in results])
    logger.info(f"Avg original distance from φ: {avg_orig:.3f}")
    logger.info(f"Avg normalized distance from φ: {avg_norm:.3f}")

    if avg_norm < avg_orig:
        pct = (avg_orig - avg_norm) / avg_orig * 100
        logger.info(f"✓ Normalization improved alignment by {pct:.1f}%!")
    else:
        logger.info("✗ Normalization did not improve alignment")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "phi_target": float(PHI),
        "results": results,
        "summary": {
            "improved_count": int(improved_count),
            "total": len(results),
            "avg_original_dist": float(avg_orig),
            "avg_normalized_dist": float(avg_norm),
        }
    }

    output_path = Path("data/experiments/question_normalization.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    test_normalization()
