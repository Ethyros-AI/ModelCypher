#!/usr/bin/env python3
"""Experiment 65: Why Does 4+1= Work?

The model gets "4+1=" correct (rank 1, 16.5%) but fails on others.
What's special about 4+1?

Hypotheses:
1. Training data frequency - "4+1" appears more in web text
2. Token structure - something special about how "4+1=" tokenizes
3. Activation geometry - "4+1=" activations are closer to counting
4. The number 5 is special - appears in many contexts

Let's investigate.
"""

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


def get_digit_token(tokenizer, digit_str):
    """Get the actual digit token ID."""
    tokens = tokenizer.encode(digit_str)
    if len(tokens) > 1 and tokens[0] == 1:
        return tokens[1]
    return tokens[0] if tokens else -1


def analyze_tokenization(tokenizer):
    """Hypothesis 1: Token structure differences."""
    logger.info("=== TOKENIZATION ANALYSIS ===")

    prompts = ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1=", "9+1="]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        decoded = [tokenizer.decode([t]) for t in tokens]
        logger.info(f"'{prompt}' → tokens={tokens}, decoded={decoded}")

    # Check if certain numbers tokenize differently
    logger.info("\nDigit token IDs:")
    for d in range(10):
        tok = get_digit_token(tokenizer, str(d))
        logger.info(f"  '{d}' → token {tok}")


def analyze_activations(model, tokenizer):
    """Hypothesis 3: Activation geometry differences."""
    import mlx.core as mx

    logger.info("\n=== ACTIVATION GEOMETRY ===")

    prompts = {
        "1+1=": "2", "2+1=": "3", "3+1=": "4", "4+1=": "5",
        "5+1=": "6", "6+1=": "7", "7+1=": "8", "8+1=": "9"
    }

    # Also get counting activations for comparison
    counting_prompts = {
        "1, 2, 3, 4,": "5",
        "2, 3, 4, 5,": "6",
        "3, 4, 5, 6,": "7",
        "4, 5, 6, 7,": "8",
    }

    def get_logits(prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    # Collect activations
    symbolic_acts = {}
    for prompt, expected in prompts.items():
        symbolic_acts[prompt] = get_logits(prompt)

    counting_acts = {}
    for prompt, expected in counting_prompts.items():
        counting_acts[prompt] = get_logits(prompt)

    # Compare 4+1= to others
    act_4plus1 = symbolic_acts["4+1="]

    logger.info("\nCosine similarity of other prompts to '4+1=':")
    for prompt, act in symbolic_acts.items():
        cos = np.dot(act, act_4plus1) / (np.linalg.norm(act) * np.linalg.norm(act_4plus1))
        logger.info(f"  '{prompt}' ↔ '4+1=': {cos:.4f}")

    # Compare 4+1= to counting prompts
    logger.info("\nCosine similarity of '4+1=' to counting prompts:")
    for prompt, act in counting_acts.items():
        cos = np.dot(act, act_4plus1) / (np.linalg.norm(act) * np.linalg.norm(act_4plus1))
        logger.info(f"  '4+1=' ↔ '{prompt}': {cos:.4f}")

    # Which counting prompt is MOST similar to 4+1=?
    best_match = max(counting_acts.items(),
                     key=lambda x: np.dot(x[1], act_4plus1) / (np.linalg.norm(x[1]) * np.linalg.norm(act_4plus1)))
    cos_best = np.dot(best_match[1], act_4plus1) / (np.linalg.norm(best_match[1]) * np.linalg.norm(act_4plus1))
    logger.info(f"\nBest counting match for '4+1=': '{best_match[0]}' (cos={cos_best:.4f})")

    return symbolic_acts, counting_acts


def analyze_probability_distribution(model, tokenizer):
    """Detailed probability analysis."""
    import mlx.core as mx

    logger.info("\n=== PROBABILITY DISTRIBUTION ANALYSIS ===")

    prompts = ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1="]

    results = []
    for prompt in prompts:
        expected_digit = str(int(prompt[0]) + 1)
        if expected_digit == "10":
            expected_digit = "10"

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist())
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        # Get target token
        target_id = get_digit_token(tokenizer, expected_digit)
        target_prob = probs[target_id]
        target_rank = int((np.argsort(logits_np)[::-1] == target_id).nonzero()[0][0])

        # Top 5 tokens
        top5_ids = np.argsort(logits_np)[-5:][::-1]
        top5_probs = probs[top5_ids]
        top5_decoded = [tokenizer.decode([t]).strip() for t in top5_ids]

        # Entropy (measure of spread)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Gap between top-1 and correct
        gap_to_correct = logits_np[top5_ids[0]] - logits_np[target_id]

        results.append({
            "prompt": prompt,
            "expected": expected_digit,
            "target_rank": target_rank,
            "target_prob": float(target_prob),
            "top1": top5_decoded[0],
            "top1_prob": float(top5_probs[0]),
            "entropy": float(entropy),
            "gap_to_correct": float(gap_to_correct),
        })

        logger.info(f"\n'{prompt}' (expects '{expected_digit}'):")
        logger.info(f"  Target rank: {target_rank+1}, prob: {target_prob:.2%}")
        logger.info(f"  Top-1: '{top5_decoded[0]}' ({top5_probs[0]:.2%})")
        logger.info(f"  Entropy: {entropy:.2f}")
        logger.info(f"  Gap to correct: {gap_to_correct:.2f}")
        logger.info(f"  Top 5: {list(zip(top5_decoded, [f'{p:.1%}' for p in top5_probs]))}")

    return results


def analyze_digit_5_special(model, tokenizer):
    """Hypothesis 4: Is the number 5 special?"""
    import mlx.core as mx

    logger.info("\n=== IS '5' SPECIAL? ===")

    # Test various prompts where 5 is the answer
    prompts_expecting_5 = [
        ("4+1=", "5"),
        ("3+2=", "5"),
        ("2+3=", "5"),
        ("1+4=", "5"),
        ("10-5=", "5"),
        ("10/2=", "5"),
        ("Count to 5: 1, 2, 3, 4,", "5"),
        ("The number after 4 is", "5"),
    ]

    # Test various prompts where 6 is the answer (comparison)
    prompts_expecting_6 = [
        ("5+1=", "6"),
        ("4+2=", "6"),
        ("3+3=", "6"),
        ("2+4=", "6"),
        ("12-6=", "6"),
        ("Count to 6: 1, 2, 3, 4, 5,", "6"),
        ("The number after 5 is", "6"),
    ]

    def check_prompts(prompts, label):
        logger.info(f"\n{label}:")
        correct = 0
        for prompt, expected in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)

            logits_np = np.array(logits[0, -1, :].tolist())
            top_token = np.argmax(logits_np)
            predicted = tokenizer.decode([top_token]).strip()

            is_correct = expected in predicted
            if is_correct:
                correct += 1
            logger.info(f"  '{prompt}' → '{predicted}' ({'✓' if is_correct else '✗'})")

        return correct / len(prompts)

    acc_5 = check_prompts(prompts_expecting_5, "Prompts expecting '5'")
    acc_6 = check_prompts(prompts_expecting_6, "Prompts expecting '6'")

    logger.info(f"\nAccuracy for '5' prompts: {acc_5:.0%}")
    logger.info(f"Accuracy for '6' prompts: {acc_6:.0%}")

    if acc_5 > acc_6 + 0.2:
        logger.info("\n*** '5' IS SPECIAL - higher accuracy on prompts expecting 5 ***")
    else:
        logger.info("\n*** No clear preference for '5' ***")

    return acc_5, acc_6


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 65: WHY DOES 4+1= WORK?")
    logger.info("=" * 60)

    # Run analyses
    analyze_tokenization(tokenizer)
    symbolic_acts, counting_acts = analyze_activations(model, tokenizer)
    prob_results = analyze_probability_distribution(model, tokenizer)
    acc_5, acc_6 = analyze_digit_5_special(model, tokenizer)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Find what's special about 4+1
    r_4plus1 = next(r for r in prob_results if r["prompt"] == "4+1=")
    others = [r for r in prob_results if r["prompt"] != "4+1="]

    logger.info(f"\n'4+1=' vs others:")
    logger.info(f"  4+1= rank: {r_4plus1['target_rank']+1}")
    logger.info(f"  Others mean rank: {np.mean([r['target_rank'] for r in others])+1:.1f}")
    logger.info(f"  4+1= entropy: {r_4plus1['entropy']:.2f}")
    logger.info(f"  Others mean entropy: {np.mean([r['entropy'] for r in others]):.2f}")

    # Save results
    results = {
        "probability_analysis": prob_results,
        "accuracy_expecting_5": acc_5,
        "accuracy_expecting_6": acc_6,
    }

    output_path = "data/experiments/why_4plus1_works.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
