#!/usr/bin/env python3
"""Experiment 61: Logit Sharpness Analysis.

The model predicts '5' for '4+1=' but at only 16.5% confidence.
Counting has 65%+ confidence.

The difference is SHARPNESS of the logit distribution.
Can we measure this geometrically and use it to guide training?

Sharpness metrics:
- Gap: max_logit - second_max_logit
- Entropy: -sum(p * log(p))
- Concentration: p_max / (1 - p_max)
- Temperature: effective temperature to reach counting sharpness
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

COUNTING_PROMPTS = [
    ("1, 2, 3, 4,", "5"),
    ("2, 3, 4, 5,", "6"),
    ("3, 4, 5, 6,", "7"),
    ("4, 5, 6, 7,", "8"),
    ("5, 6, 7, 8,", "9"),
]

SYMBOLIC_PROMPTS = [
    ("4+1=", "5"),
    ("5+1=", "6"),
    ("6+1=", "7"),
    ("7+1=", "8"),
    ("8+1=", "9"),
]


def softmax(logits, temperature=1.0):
    """Softmax with temperature."""
    x = logits / temperature
    x = x - x.max()  # For numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def entropy(probs):
    """Shannon entropy."""
    # Add small epsilon to avoid log(0)
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


def analyze_sharpness(model, tokenizer):
    """Analyze logit sharpness for counting vs symbolic."""
    import mlx.core as mx

    def get_logits(prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 61: LOGIT SHARPNESS ANALYSIS")
    logger.info("=" * 60)

    results = {"counting": [], "symbolic": []}

    logger.info("\n=== COUNTING PROMPTS ===")
    for prompt, expected in COUNTING_PROMPTS:
        logits = get_logits(prompt)
        probs = softmax(logits)

        # Get target token
        target_tokens = tokenizer.encode(expected)
        target_id = target_tokens[0] if target_tokens else -1

        # Top tokens
        top_indices = np.argsort(logits)[-5:][::-1]
        top_probs = probs[top_indices]

        # Sharpness metrics
        gap = logits[top_indices[0]] - logits[top_indices[1]]
        ent = entropy(probs)
        conc = probs[top_indices[0]] / (1 - probs[top_indices[0]] + 1e-10)
        target_prob = probs[target_id] if target_id >= 0 else 0.0
        target_rank = np.where(np.argsort(logits)[::-1] == target_id)[0][0] if target_id >= 0 else -1

        logger.info(f"\n'{prompt}'")
        logger.info(f"  Expected: '{expected}' (prob={target_prob:.2%}, rank={target_rank+1})")
        logger.info(f"  Top: {[tokenizer.decode([t]).strip() for t in top_indices[:3]]}")
        logger.info(f"  Probs: {[f'{p:.1%}' for p in top_probs[:3]]}")
        logger.info(f"  Gap (max-2nd): {gap:.2f}")
        logger.info(f"  Entropy: {ent:.2f}")
        logger.info(f"  Concentration: {conc:.2f}")

        results["counting"].append({
            "prompt": prompt,
            "expected": expected,
            "target_prob": float(target_prob),
            "target_rank": int(target_rank),
            "gap": float(gap),
            "entropy": float(ent),
            "concentration": float(conc),
            "top1_prob": float(top_probs[0]),
        })

    logger.info("\n=== SYMBOLIC PROMPTS ===")
    for prompt, expected in SYMBOLIC_PROMPTS:
        logits = get_logits(prompt)
        probs = softmax(logits)

        # Get target token
        target_tokens = tokenizer.encode(expected)
        target_id = target_tokens[0] if target_tokens else -1

        # Top tokens
        top_indices = np.argsort(logits)[-5:][::-1]
        top_probs = probs[top_indices]

        # Sharpness metrics
        gap = logits[top_indices[0]] - logits[top_indices[1]]
        ent = entropy(probs)
        conc = probs[top_indices[0]] / (1 - probs[top_indices[0]] + 1e-10)
        target_prob = probs[target_id] if target_id >= 0 else 0.0
        target_rank = np.where(np.argsort(logits)[::-1] == target_id)[0][0] if target_id >= 0 else -1

        logger.info(f"\n'{prompt}'")
        logger.info(f"  Expected: '{expected}' (prob={target_prob:.2%}, rank={target_rank+1})")
        logger.info(f"  Top: {[tokenizer.decode([t]).strip() for t in top_indices[:3]]}")
        logger.info(f"  Probs: {[f'{p:.1%}' for p in top_probs[:3]]}")
        logger.info(f"  Gap (max-2nd): {gap:.2f}")
        logger.info(f"  Entropy: {ent:.2f}")
        logger.info(f"  Concentration: {conc:.2f}")

        results["symbolic"].append({
            "prompt": prompt,
            "expected": expected,
            "target_prob": float(target_prob),
            "target_rank": int(target_rank),
            "gap": float(gap),
            "entropy": float(ent),
            "concentration": float(conc),
            "top1_prob": float(top_probs[0]),
        })

    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    c_gap = np.mean([r["gap"] for r in results["counting"]])
    s_gap = np.mean([r["gap"] for r in results["symbolic"]])
    c_ent = np.mean([r["entropy"] for r in results["counting"]])
    s_ent = np.mean([r["entropy"] for r in results["symbolic"]])
    c_conc = np.mean([r["concentration"] for r in results["counting"]])
    s_conc = np.mean([r["concentration"] for r in results["symbolic"]])
    c_top1 = np.mean([r["top1_prob"] for r in results["counting"]])
    s_top1 = np.mean([r["top1_prob"] for r in results["symbolic"]])
    c_target = np.mean([r["target_prob"] for r in results["counting"]])
    s_target = np.mean([r["target_prob"] for r in results["symbolic"]])
    c_rank = np.mean([r["target_rank"] for r in results["counting"]])
    s_rank = np.mean([r["target_rank"] for r in results["symbolic"]])

    logger.info(f"\n{'Metric':<15} {'Counting':>12} {'Symbolic':>12} {'Ratio':>10}")
    logger.info("-" * 50)
    logger.info(f"{'Gap':<15} {c_gap:>12.2f} {s_gap:>12.2f} {c_gap/s_gap:>10.2f}x")
    logger.info(f"{'Entropy':<15} {c_ent:>12.2f} {s_ent:>12.2f} {c_ent/s_ent:>10.2f}x")
    logger.info(f"{'Concentration':<15} {c_conc:>12.2f} {s_conc:>12.2f} {c_conc/s_conc:>10.2f}x")
    logger.info(f"{'Top-1 Prob':<15} {c_top1:>11.1%} {s_top1:>11.1%} {c_top1/s_top1:>10.2f}x")
    logger.info(f"{'Target Prob':<15} {c_target:>11.1%} {s_target:>11.1%} {'-':>10}")
    logger.info(f"{'Target Rank':<15} {c_rank:>12.1f} {s_rank:>12.1f} {'-':>10}")

    # Compute temperature needed to match sharpness
    # If symbolic has entropy S_s and we want it to match counting entropy S_c,
    # we need temperature τ where S(logits/τ) = S_c
    # This is complex to solve analytically, so let's estimate

    logger.info(f"\n=== GEOMETRY-DERIVED TARGETS ===")
    logger.info(f"Counting sharpness (gap): {c_gap:.2f}")
    logger.info(f"Symbolic sharpness (gap): {s_gap:.2f}")
    logger.info(f"Gap ratio: {c_gap/s_gap:.2f}x")
    logger.info(f"\nTo match counting, symbolic needs {c_gap/s_gap:.2f}x sharper logits")

    # The training target: increase symbolic gap to match counting gap
    # This is a measurable geometric quantity
    target_gap_increase = c_gap - s_gap
    logger.info(f"Required gap increase: {target_gap_increase:.2f}")

    results["summary"] = {
        "counting": {
            "mean_gap": float(c_gap),
            "mean_entropy": float(c_ent),
            "mean_concentration": float(c_conc),
            "mean_top1": float(c_top1),
            "mean_target_prob": float(c_target),
            "mean_target_rank": float(c_rank),
        },
        "symbolic": {
            "mean_gap": float(s_gap),
            "mean_entropy": float(s_ent),
            "mean_concentration": float(s_conc),
            "mean_top1": float(s_top1),
            "mean_target_prob": float(s_target),
            "mean_target_rank": float(s_rank),
        },
        "ratio": {
            "gap": float(c_gap / s_gap),
            "entropy": float(c_ent / s_ent),
            "concentration": float(c_conc / s_conc),
            "top1": float(c_top1 / s_top1),
        },
        "target_gap_increase": float(target_gap_increase),
    }

    return results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    results = analyze_sharpness(model, tokenizer)

    output_path = "data/experiments/logit_sharpness.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
