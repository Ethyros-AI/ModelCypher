#!/usr/bin/env python3
"""Compare NB-LoRA vs Geometric LoRA Results.

Loads results from both experiments and produces a comparison report.

Usage:
    poetry run python scripts/evaluate_lora_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_results(path: Path) -> dict | None:
    """Load experiment results from JSON."""
    if not path.exists():
        logger.warning(f"Results not found: {path}")
        return None

    with open(path) as f:
        return json.load(f)


def compare_results(nb_lora: dict, geometric: dict) -> dict:
    """Compare NB-LoRA vs Geometric LoRA results."""
    comparison = {
        "training": {},
        "validation": {},
        "spectral_safety": {},
        "winner": None,
    }

    # Training comparison
    nb_final_loss = nb_lora["training"]["final_loss"]
    geo_final_loss = geometric["training"]["final_loss"]

    comparison["training"] = {
        "nb_lora_loss": nb_final_loss,
        "geometric_loss": geo_final_loss,
        "delta": nb_final_loss - geo_final_loss,
        "better": "nb_lora" if nb_final_loss < geo_final_loss else "geometric",
    }

    # Validation comparison
    nb_acc = nb_lora["validation"]["accuracy"]
    geo_acc = geometric["validation"]["accuracy"]

    comparison["validation"] = {
        "nb_lora_accuracy": nb_acc,
        "geometric_accuracy": geo_acc,
        "delta": nb_acc - geo_acc,
        "better": "nb_lora" if nb_acc > geo_acc else "geometric",
    }

    # Spectral safety comparison
    nb_max_ratio = max(r["max_ratio"] for r in nb_lora["spectral_bounds"].values())
    geo_max_ratio = max(r["max_ratio"] for r in geometric["spectral_bounds"].values())

    nb_all_safe = all(r["is_safe"] for r in nb_lora["spectral_bounds"].values())
    geo_all_safe = all(r["is_safe"] for r in geometric["spectral_bounds"].values())

    comparison["spectral_safety"] = {
        "nb_lora_max_ratio": nb_max_ratio,
        "geometric_max_ratio": geo_max_ratio,
        "nb_lora_all_safe": nb_all_safe,
        "geometric_all_safe": geo_all_safe,
        "better": "nb_lora" if nb_max_ratio < geo_max_ratio else "geometric",
    }

    # Determine overall winner
    # Criteria: better accuracy AND lower max_ratio (geometric safety)
    acc_winner = comparison["validation"]["better"]
    safety_winner = comparison["spectral_safety"]["better"]

    if acc_winner == safety_winner:
        comparison["winner"] = acc_winner
    elif comparison["validation"]["delta"] == 0:
        # Tie on accuracy, use safety
        comparison["winner"] = safety_winner
    else:
        # Mixed results - accuracy takes priority if both are safe
        if nb_all_safe and geo_all_safe:
            comparison["winner"] = acc_winner
        else:
            comparison["winner"] = "nb_lora" if nb_all_safe else "geometric"

    return comparison


def print_comparison(comparison: dict, nb_lora: dict, geometric: dict):
    """Print formatted comparison report."""
    logger.info("=" * 70)
    logger.info("NB-LoRA vs GEOMETRIC LoRA COMPARISON")
    logger.info("=" * 70)

    # Training
    logger.info("\n--- TRAINING ---")
    t = comparison["training"]
    logger.info(f"  NB-LoRA final loss:    {t['nb_lora_loss']:.4f}")
    logger.info(f"  Geometric final loss:  {t['geometric_loss']:.4f}")
    logger.info(f"  Better: {t['better'].upper()}")

    # Validation
    logger.info("\n--- VALIDATION ACCURACY ---")
    v = comparison["validation"]
    logger.info(f"  NB-LoRA:    {v['nb_lora_accuracy']:.1%}")
    logger.info(f"  Geometric:  {v['geometric_accuracy']:.1%}")
    logger.info(f"  Delta:      {v['delta']:+.1%}")
    logger.info(f"  Better: {v['better'].upper()}")

    # Spectral Safety
    logger.info("\n--- SPECTRAL SAFETY ---")
    s = comparison["spectral_safety"]
    logger.info(f"  NB-LoRA max per-direction ratio:    {s['nb_lora_max_ratio']:.4f}")
    logger.info(f"  Geometric max per-direction ratio:  {s['geometric_max_ratio']:.4f}")
    logger.info(f"  NB-LoRA all safe:    {s['nb_lora_all_safe']}")
    logger.info(f"  Geometric all safe:  {s['geometric_all_safe']}")
    logger.info(f"  Better: {s['better'].upper()}")

    # Detailed spectral norms
    logger.info("\n--- SPECTRAL NORMS (sample) ---")
    logger.info("  Layer | NB-LoRA | Geometric")
    logger.info("  " + "-" * 40)

    nb_spectral = nb_lora["spectral_bounds"]
    geo_spectral = geometric["spectral_bounds"]

    for key in list(nb_spectral.keys())[:5]:
        nb_norm = nb_spectral[key]["spectral_norm"]
        geo_norm = geo_spectral.get(key, {}).get("spectral_norm", 0)
        logger.info(f"  {key[-20:]:20} | {nb_norm:.6f} | {geo_norm:.6f}")

    # Winner
    logger.info("\n" + "=" * 70)
    logger.info(f"OVERALL WINNER: {comparison['winner'].upper()}")
    logger.info("=" * 70)

    # Interpretation
    logger.info("\n--- INTERPRETATION ---")

    if comparison["winner"] == "nb_lora":
        logger.info("""
NB-LoRA outperformed Geometric LoRA in this experiment.

Key advantages observed:
- Cayley parameterization enforces bounds during training (not just at deployment)
- Per-direction ratios are lower, indicating better geometric preservation
- Training-time constraints may lead to more stable optimization

This validates the hypothesis: training-time bounds > post-hoc rescaling.
""")
    else:
        logger.info("""
Geometric LoRA (post-hoc rescaling) performed comparably or better.

Possible explanations:
- Post-hoc rescaling may be sufficient for this task complexity
- NB-LoRA Cayley transform may introduce optimization challenges
- The training regime may not have been long enough to see NB-LoRA benefits

Further investigation:
- Try longer training (more iterations)
- Try more complex tasks where bound violations cause real degradation
- Check gradient flow through Cayley transform
""")


def main():
    nb_lora_path = Path("data/experiments/nb_lora_experiment/nb_lora_results.json")
    geometric_path = Path("data/experiments/geometric_lora_baseline/geometric_lora_results.json")

    logger.info("Loading experiment results...")

    nb_lora = load_results(nb_lora_path)
    geometric = load_results(geometric_path)

    if nb_lora is None or geometric is None:
        logger.error("Missing experiment results. Run both experiments first:")
        logger.error("  poetry run python scripts/train_nb_lora_experiment.py")
        logger.error("  poetry run python scripts/train_geometric_lora_baseline.py")
        return

    # Compare
    comparison = compare_results(nb_lora, geometric)

    # Print report
    print_comparison(comparison, nb_lora, geometric)

    # Save comparison
    output_path = Path("data/experiments/lora_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
