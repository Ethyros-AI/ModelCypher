#!/usr/bin/env python3
"""Probe Fractional Dimensions in Model Merging.

Hypothesis: The "missing bits" in merging live in fractional dimensional spaces
that we're not accounting for when computing alignment transforms.

The π/e ratio dominates weight space (156 matches), while π as integer (3)
appears in activations. The fractional part (π-3 ≈ 0.14159) might encode
critical capability structure.

This experiment:
1. Compute the intrinsic dimension of capability subspaces
2. Check if the fractional parts cluster around mathematical constants
3. Test if preserving fractional dimensional structure improves merging
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mathematical constants and their fractional parts
CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
    "phi": (1 + np.sqrt(5)) / 2,
    "sqrt2": np.sqrt(2),
    "sqrt3": np.sqrt(3),
    "pi/e": np.pi / np.e,
    "e/pi": np.e / np.pi,
}

FRACTIONAL_PARTS = {k: v - int(v) for k, v in CONSTANTS.items()}


def estimate_intrinsic_dimension(activations: np.ndarray) -> float:
    """Estimate intrinsic dimension using correlation dimension.

    Uses the Grassberger-Procaccia algorithm.
    """
    n_samples = activations.shape[0]
    if n_samples < 10:
        return float('nan')

    # Compute pairwise distances
    dists = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(activations[i] - activations[j])
            if d > 1e-10:
                dists.append(d)

    if len(dists) < 10:
        return float('nan')

    dists = np.array(dists)
    dists = dists[dists > 0]

    # Log-log slope gives dimension estimate
    log_dists = np.log(dists)
    r_min, r_max = np.percentile(log_dists, [10, 90])

    if r_max - r_min < 0.1:
        return float('nan')

    # Count pairs within radius r
    radii = np.logspace(r_min, r_max, 20)
    counts = []
    for r in radii:
        count = np.sum(dists < np.exp(r)) / (n_samples * (n_samples - 1))
        counts.append(count)

    counts = np.array(counts)
    counts = counts[counts > 0]

    if len(counts) < 5:
        return float('nan')

    log_counts = np.log(counts[:len(counts)])
    log_radii = np.log(radii[:len(counts)])

    # Linear fit to estimate dimension
    slope, _ = np.polyfit(log_radii, log_counts, 1)
    return max(0, slope)


def get_fractional_part_matches(dimensions: List[float], tolerance: float = 0.02) -> dict:
    """Count how many dimension fractional parts match mathematical constants."""
    matches = {k: 0 for k in FRACTIONAL_PARTS}

    for dim in dimensions:
        frac = dim - int(dim)
        for const_name, const_frac in FRACTIONAL_PARTS.items():
            # Check if fractional part matches constant's fractional part
            if abs(frac - const_frac) < tolerance or abs(frac - (1 - const_frac)) < tolerance:
                matches[const_name] += 1

    return matches


def compute_dimension_from_gram(gram: np.ndarray) -> float:
    """Compute effective dimension from Gram matrix eigenspectrum.

    Uses the participation ratio: (sum(λ))^2 / sum(λ^2)
    This gives a continuous dimension estimate.
    """
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Normalize
    eigenvalues = eigenvalues / eigenvalues.sum()

    # Participation ratio
    participation = 1.0 / np.sum(eigenvalues ** 2)
    return participation


def main():
    from mlx_lm import load
    import mlx.core as mx

    model_path = "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16"
    adapter_path = "data/adapters/qwen3_math_lora"  # The working v1 adapter

    logger.info("=" * 70)
    logger.info("FRACTIONAL DIMENSION PROBE")
    logger.info("=" * 70)
    logger.info("Testing if capability structure lives in fractional dimensions")

    # Test prompts by capability - more prompts for better statistics
    capability_prompts = {
        "arithmetic": [
            "1+1=", "2+2=", "3+5=", "7-3=", "9-4=",
            "4+6=", "5+5=", "6+6=", "8+8=", "7+7=",
            "3+3=", "4+4=", "2+3=", "5+4=", "8-5=",
        ],
        "language": [
            "The cat sat on the", "Fire is hot and ice is",
            "The opposite of up is", "A dog can bark and a cat can",
            "The sky is blue and grass is", "Water is wet and sand is",
            "Birds can fly and fish can", "The sun is bright and the moon is",
            "Honey is sweet and lemons are", "Snow is white and coal is",
        ],
        "comparison": [
            "Which is greater, 7 or 3?", "Which is larger, 5 or 9?",
            "Is 15 > 12?", "Is 8 larger than 10?",
            "Which is bigger, 20 or 5?", "Is 100 greater than 50?",
            "Which is smaller, 2 or 8?", "Is 3 less than 7?",
        ],
        "word_problems": [
            "I have 3 apples. I get 2 more. Total:",
            "5 birds. 2 fly away. Remaining:",
            "Start with 4. Add 6. Result:",
            "There are 7 cats. 3 leave. Remaining:",
            "I have 8 toys. I get 4 more. Total:",
            "10 fish. 5 swim away. Remaining:",
        ],
    }

    # Load model
    model, tokenizer = load(model_path)

    def get_activations(prompts: List[str]) -> np.ndarray:
        """Get final logits as proxy for activations."""
        all_acts = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            logits = model(mx.array([tokens]))
            mx.eval(logits)
            # Use top-k logits as compressed representation
            acts = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            # PCA-like compression: keep top variance directions
            top_k = 1024
            top_indices = np.argsort(np.abs(acts))[-top_k:]
            acts = acts[top_indices]
            all_acts.append(acts)
        return np.stack(all_acts)

    # Collect dimensions for each capability
    logger.info("\n=== CAPABILITY INTRINSIC DIMENSIONS ===\n")

    all_dimensions = []
    results = {}

    for cap_name, prompts in capability_prompts.items():
        acts = get_activations(prompts)

        # Compute Gram matrix
        gram = acts @ acts.T
        dim = compute_dimension_from_gram(gram)

        # Also try intrinsic dimension
        intrinsic = estimate_intrinsic_dimension(acts)
        if np.isnan(intrinsic):
            intrinsic = dim  # fallback

        # Use participation ratio dimension
        if np.isnan(dim) or dim == 0:
            dim = 1.0

        integer_part = int(dim)
        fractional_part = dim - integer_part

        logger.info(f"{cap_name}:")
        logger.info(f"  Intrinsic dimension: {dim:.4f}")
        logger.info(f"  Integer part: {integer_part}")
        logger.info(f"  Fractional part: {fractional_part:.4f}")

        # Check fractional part against constants
        closest_const = min(FRACTIONAL_PARTS.items(),
                          key=lambda x: min(abs(fractional_part - x[1]),
                                          abs(fractional_part - (1 - x[1]))))
        logger.info(f"  Closest constant: {closest_const[0]} (frac={closest_const[1]:.4f})")

        all_dimensions.append(dim)
        results[cap_name] = {
            "dimension": dim,
            "integer": integer_part,
            "fractional": fractional_part,
            "closest_constant": closest_const[0],
        }

    # Overall analysis
    logger.info("\n=== FRACTIONAL PART ANALYSIS ===\n")

    # Count matches
    matches = get_fractional_part_matches(all_dimensions, tolerance=0.05)
    logger.info("Fractional parts matching constants:")
    for const, count in sorted(matches.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"  {const}: {count} matches")

    # Statistical analysis
    fractional_parts = [d - int(d) for d in all_dimensions]
    mean_frac = np.mean(fractional_parts)
    std_frac = np.std(fractional_parts)

    logger.info(f"\nFractional part statistics:")
    logger.info(f"  Mean: {mean_frac:.4f}")
    logger.info(f"  Std:  {std_frac:.4f}")

    # Check if mean is close to any constant
    for const_name, const_frac in FRACTIONAL_PARTS.items():
        if abs(mean_frac - const_frac) < 0.1:
            logger.info(f"  Mean close to {const_name} fractional ({const_frac:.4f})")

    # The pi/e observation
    pi_frac = np.pi - 3  # 0.14159...
    logger.info(f"\n  π fractional part: {pi_frac:.4f}")
    logger.info(f"  Distance from mean: {abs(mean_frac - pi_frac):.4f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("HYPOTHESIS CHECK")
    logger.info("=" * 70)

    logger.info(f"""
The fractional dimension hypothesis suggests that capabilities
encode structure in the fractional parts of intrinsic dimensions.

Findings:
- Mean fractional part: {mean_frac:.4f}
- π fractional (0.14159): {"CLOSE" if abs(mean_frac - pi_frac) < 0.1 else "NOT CLOSE"}
- π/e fractional ({FRACTIONAL_PARTS['pi/e']:.4f}): {"CLOSE" if abs(mean_frac - FRACTIONAL_PARTS['pi/e']) < 0.1 else "NOT CLOSE"}

Implication for merging:
If fractional dimensions carry capability information, then:
1. Simple Procrustes alignment (integer dimensions) may lose structure
2. We need fractional-dimension-aware alignment
3. The π/e ratio may describe the relationship between dimensions

Next experiment: Test if preserving fractional structure improves merge quality.
""")

    # Save results
    output_path = Path("data/experiments/fractional_dimension_probe.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj

    with open(output_path, "w") as f:
        json.dump({
            "capabilities": to_native(results),
            "fractional_matches": to_native(matches),
            "mean_fractional": float(mean_frac),
            "std_fractional": float(std_frac),
            "pi_fractional": float(pi_frac),
            "constants": {k: float(v) for k, v in FRACTIONAL_PARTS.items()},
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    del model
    mx.clear_cache()


if __name__ == "__main__":
    main()
