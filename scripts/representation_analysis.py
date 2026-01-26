#!/usr/bin/env python3
"""Experiment 60: Representation Analysis.

The geometry told us κ(symbolic) = 2.36e16 - essentially singular.
This means symbolic prompts collapse to nearly the same representation.

Let's see what's actually happening:
1. What do the symbolic activations look like?
2. How do they compare to counting activations?
3. Why is one structured and the other collapsed?
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

# Prompts
COUNTING_PROMPTS = [
    "1, 2, 3, 4,",
    "2, 3, 4, 5,",
    "3, 4, 5, 6,",
    "4, 5, 6, 7,",
    "5, 6, 7, 8,",
    "6, 7, 8, 9,",
    "Count to 5: 1, 2, 3, 4,",
    "Count: one, two, three,",
]

SYMBOLIC_PROMPTS = [
    "4+1=",
    "5+1=",
    "6+1=",
    "7+1=",
    "8+1=",
    "9+1=",
    "4+1=",  # duplicate to match
    "3+1=",
]


def analyze_representations(model, tokenizer):
    """Deep dive into representation structure."""
    import mlx.core as mx

    def get_logits(prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        return np.array(logits[0, -1, :].tolist(), dtype=np.float32)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 60: REPRESENTATION ANALYSIS")
    logger.info("=" * 60)

    # Collect activations
    counting_acts = np.vstack([get_logits(p) for p in COUNTING_PROMPTS])
    symbolic_acts = np.vstack([get_logits(p) for p in SYMBOLIC_PROMPTS])

    logger.info(f"\nCounting activations shape: {counting_acts.shape}")
    logger.info(f"Symbolic activations shape: {symbolic_acts.shape}")

    # Basic statistics
    logger.info("\n=== BASIC STATISTICS ===")
    logger.info(f"Counting - mean: {counting_acts.mean():.4f}, std: {counting_acts.std():.4f}")
    logger.info(f"Symbolic - mean: {symbolic_acts.mean():.4f}, std: {symbolic_acts.std():.4f}")

    # Pairwise distances
    logger.info("\n=== PAIRWISE DISTANCES ===")

    logger.info("\nCounting prompt distances (should be spread out):")
    for i in range(len(COUNTING_PROMPTS)):
        for j in range(i+1, min(i+3, len(COUNTING_PROMPTS))):
            dist = np.linalg.norm(counting_acts[i] - counting_acts[j])
            logger.info(f"  '{COUNTING_PROMPTS[i][:15]}' ↔ '{COUNTING_PROMPTS[j][:15]}': {dist:.2f}")

    logger.info("\nSymbolic prompt distances (if collapsed, should be ~0):")
    for i in range(len(SYMBOLIC_PROMPTS)):
        for j in range(i+1, min(i+3, len(SYMBOLIC_PROMPTS))):
            dist = np.linalg.norm(symbolic_acts[i] - symbolic_acts[j])
            logger.info(f"  '{SYMBOLIC_PROMPTS[i]}' ↔ '{SYMBOLIC_PROMPTS[j]}': {dist:.2f}")

    # SVD analysis
    logger.info("\n=== SVD ANALYSIS ===")

    # Centered
    C_c = counting_acts - counting_acts.mean(axis=0)
    S_c = symbolic_acts - symbolic_acts.mean(axis=0)

    U_c, s_c, Vt_c = np.linalg.svd(C_c, full_matrices=False)
    U_s, s_s, Vt_s = np.linalg.svd(S_c, full_matrices=False)

    logger.info(f"\nCounting singular values: {s_c}")
    logger.info(f"Symbolic singular values: {s_s}")

    # Effective rank (how many dimensions are actually used)
    eps = 1e-10
    eff_rank_c = np.sum(s_c > eps * s_c.max())
    eff_rank_s = np.sum(s_s > eps * s_s.max())

    logger.info(f"\nEffective rank (counting): {eff_rank_c}")
    logger.info(f"Effective rank (symbolic): {eff_rank_s}")

    # Variance explained
    var_c = (s_c ** 2) / (s_c ** 2).sum()
    var_s = (s_s ** 2) / (s_s ** 2).sum()

    logger.info(f"\nVariance explained (counting): {var_c}")
    logger.info(f"Variance explained (symbolic): {var_s}")

    logger.info(f"\nCounting: {var_c[0]:.1%} in first dimension")
    logger.info(f"Symbolic: {var_s[0]:.1%} in first dimension")

    # Cross-prompt cosine similarity
    logger.info("\n=== COSINE SIMILARITY ANALYSIS ===")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    logger.info("\nCounting pairwise cosines:")
    count_cosines = []
    for i in range(len(COUNTING_PROMPTS)):
        for j in range(i+1, len(COUNTING_PROMPTS)):
            cos = cosine_sim(counting_acts[i], counting_acts[j])
            count_cosines.append(cos)
    logger.info(f"  Mean: {np.mean(count_cosines):.4f}, Std: {np.std(count_cosines):.4f}")

    logger.info("\nSymbolic pairwise cosines:")
    symb_cosines = []
    for i in range(len(SYMBOLIC_PROMPTS)):
        for j in range(i+1, len(SYMBOLIC_PROMPTS)):
            cos = cosine_sim(symbolic_acts[i], symbolic_acts[j])
            symb_cosines.append(cos)
    logger.info(f"  Mean: {np.mean(symb_cosines):.4f}, Std: {np.std(symb_cosines):.4f}")

    # Cross-type comparison
    logger.info("\n=== CROSS-TYPE ANALYSIS ===")
    cross_cosines = []
    for i in range(min(len(COUNTING_PROMPTS), len(SYMBOLIC_PROMPTS))):
        cos = cosine_sim(counting_acts[i], symbolic_acts[i])
        cross_cosines.append(cos)
        logger.info(f"  '{COUNTING_PROMPTS[i][:15]}' ↔ '{SYMBOLIC_PROMPTS[i]}': {cos:.4f}")

    logger.info(f"\nMean cross-type cosine: {np.mean(cross_cosines):.4f}")

    # Token analysis - what tokens are high probability?
    logger.info("\n=== TOP PREDICTIONS ===")

    for i, (cp, sp) in enumerate(zip(COUNTING_PROMPTS[:4], SYMBOLIC_PROMPTS[:4])):
        # Counting
        c_logits = counting_acts[i]
        c_top5 = np.argsort(c_logits)[-5:][::-1]
        c_probs = np.exp(c_logits - c_logits.max())
        c_probs = c_probs / c_probs.sum()

        # Symbolic
        s_logits = symbolic_acts[i]
        s_top5 = np.argsort(s_logits)[-5:][::-1]
        s_probs = np.exp(s_logits - s_logits.max())
        s_probs = s_probs / s_probs.sum()

        logger.info(f"\n'{cp}' top tokens: {[tokenizer.decode([t]).strip() for t in c_top5]}")
        logger.info(f"  probs: {[f'{c_probs[t]:.2%}' for t in c_top5]}")
        logger.info(f"'{sp}' top tokens: {[tokenizer.decode([t]).strip() for t in s_top5]}")
        logger.info(f"  probs: {[f'{s_probs[t]:.2%}' for t in s_top5]}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if var_s[0] > 0.95:
        logger.info(f"\n*** SYMBOLIC IS COLLAPSED ***")
        logger.info(f"  {var_s[0]:.1%} variance in one dimension")
        logger.info(f"  Effective rank: {eff_rank_s}")
        logger.info(f"  All prompts map to ~same point")
        conclusion = "symbolic_collapsed"
    elif np.mean(symb_cosines) > 0.99:
        logger.info(f"\n*** SYMBOLIC PROMPTS ARE NEARLY IDENTICAL ***")
        logger.info(f"  Mean pairwise cosine: {np.mean(symb_cosines):.4f}")
        conclusion = "symbolic_identical"
    else:
        logger.info(f"\n*** REPRESENTATIONS ARE DISTINCT ***")
        conclusion = "distinct"

    return {
        "counting": {
            "mean": float(counting_acts.mean()),
            "std": float(counting_acts.std()),
            "singular_values": s_c.tolist(),
            "effective_rank": int(eff_rank_c),
            "variance_explained": var_c.tolist(),
            "pairwise_cosines": count_cosines,
        },
        "symbolic": {
            "mean": float(symbolic_acts.mean()),
            "std": float(symbolic_acts.std()),
            "singular_values": s_s.tolist(),
            "effective_rank": int(eff_rank_s),
            "variance_explained": var_s.tolist(),
            "pairwise_cosines": symb_cosines,
        },
        "cross_type_cosines": cross_cosines,
        "conclusion": conclusion,
    }


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    results = analyze_representations(model, tokenizer)

    output_path = "data/experiments/representation_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
