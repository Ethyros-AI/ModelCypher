#!/usr/bin/env python3
"""Experiment 81: Geometric Self-Detection.

Can we automatically detect disconnected capabilities via κ alone?

The key insight from Phase 9.5:
- High κ = disconnected capability (present but not accessible)
- Low κ = working capability

If we can establish a threshold, a model can SCAN ITSELF to find
what needs bridging without human intervention.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
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


@dataclass
class DomainStatus:
    """Status of a capability domain."""
    name: str
    kappa: float
    accuracy_raw: float
    accuracy_primed: float
    prime_used: str
    status: str  # "working", "disconnected", "missing"


def get_activations(model, tokenizer, prompt: str, layer_idx: int = -1) -> np.ndarray:
    """Get activations for a prompt at a specific layer."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if i == layer_idx or (layer_idx == -1 and i == len(model.model.layers) - 1):
            break

    mx.eval(hidden)
    return np.array(hidden[0, -1, :].tolist())


def compute_kappa(activations_list: List[np.ndarray]) -> float:
    """Compute condition number of Gram matrix."""
    X = np.stack(activations_list)
    G = X @ X.T
    try:
        kappa = np.linalg.cond(G)
    except:
        kappa = np.inf
    return kappa


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> float:
    """Evaluate accuracy on a problem set with optional prime."""
    import mlx.core as mx

    correct = 0
    for problem, expected in problems:
        prompt = f"{prime} {problem}" if prime else problem

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_token = int(np.argmax(probs))
        predicted = tokenizer.decode([top_token]).strip()

        if expected in predicted or predicted == expected:
            correct += 1

    return correct / len(problems) if problems else 0.0


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 81: GEOMETRIC SELF-DETECTION")
    logger.info("=" * 60)

    # Define capability domains with test problems and known primes
    domains = {
        "arithmetic_addition": {
            "prompts": ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1="],
            "problems": [("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
                        ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9")],
            "prime": "Arithmetic means calculating numbers.",
        },
        "arithmetic_subtraction": {
            "prompts": ["5-1=", "4-1=", "3-1=", "6-2=", "7-3=", "8-2=", "9-4=", "10-5="],
            "problems": [("5-1=", "4"), ("4-1=", "3"), ("3-1=", "2"), ("6-2=", "4"),
                        ("7-3=", "4"), ("8-2=", "6"), ("9-4=", "5"), ("10-5=", "5")],
            "prime": "Arithmetic means calculating numbers.",
        },
        "counting": {
            "prompts": ["1, 2,", "2, 3,", "3, 4,", "4, 5,", "5, 6,", "6, 7,", "7, 8,", "8, 9,"],
            "problems": [("1, 2,", "3"), ("2, 3,", "4"), ("3, 4,", "5"), ("4, 5,", "6"),
                        ("5, 6,", "7"), ("6, 7,", "8"), ("7, 8,", "9"), ("8, 9,", "10")],
            "prime": "",  # Counting should work without prime
        },
        "word_problems": {
            "prompts": ["I have 3 apples. I get 2 more. Total:",
                       "5 birds. 2 fly away. Remaining:",
                       "Start with 4. Add 3. Result:"],
            "problems": [("I have 3 apples. I get 2 more. Total:", "5"),
                        ("5 birds. 2 fly away. Remaining:", "3"),
                        ("Start with 4. Add 3. Result:", "7")],
            "prime": "Arithmetic means calculating numbers.",
        },
        "letter_sequence": {
            "prompts": ["A, B,", "B, C,", "C, D,", "D, E,", "E, F,", "F, G,", "G, H,"],
            "problems": [("A, B,", "C"), ("B, C,", "D"), ("C, D,", "E"), ("D, E,", "F"),
                        ("E, F,", "G"), ("F, G,", "H"), ("G, H,", "I")],
            "prime": "The alphabet continues.",
        },
        "comparison": {
            "prompts": ["5 > 3:", "2 < 7:", "4 = 4:", "9 > 1:", "3 < 8:"],
            "problems": [("5 > 3:", "True"), ("2 < 7:", "True"), ("4 = 4:", "True"),
                        ("9 > 1:", "True"), ("3 < 8:", "True")],
            "prime": "Comparison evaluates to True or False.",
        },
        "negation": {
            "prompts": ["not True is", "not False is", "opposite of yes is"],
            "problems": [("not True is", "False"), ("not False is", "True"),
                        ("opposite of yes is", "no")],
            "prime": "Logical negation.",
        },
    }

    logger.info("\n=== SCANNING CAPABILITY DOMAINS ===")

    results = []

    for domain_name, domain_data in domains.items():
        logger.info(f"\n--- {domain_name} ---")

        # Get activations for κ computation
        prompts = domain_data["prompts"]
        acts = [get_activations(model, tokenizer, p) for p in prompts]
        kappa = compute_kappa(acts)

        # Test raw accuracy
        problems = domain_data["problems"]
        acc_raw = evaluate_accuracy(model, tokenizer, "", problems)

        # Test primed accuracy
        prime = domain_data["prime"]
        acc_primed = evaluate_accuracy(model, tokenizer, prime, problems) if prime else acc_raw

        # Determine status
        if acc_raw >= 0.7:
            status = "working"
        elif acc_primed >= 0.7:
            status = "disconnected"
        else:
            status = "missing"

        domain_status = DomainStatus(
            name=domain_name,
            kappa=kappa,
            accuracy_raw=acc_raw,
            accuracy_primed=acc_primed,
            prime_used=prime,
            status=status,
        )
        results.append(domain_status)

        logger.info(f"  κ = {kappa:.2e}")
        logger.info(f"  Accuracy (raw): {acc_raw:.0%}")
        logger.info(f"  Accuracy (primed): {acc_primed:.0%}")
        logger.info(f"  Status: {status.upper()}")

    # Analyze κ separation
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS: κ AS DISCONNECTION SIGNAL")
    logger.info("=" * 60)

    working = [r for r in results if r.status == "working"]
    disconnected = [r for r in results if r.status == "disconnected"]
    missing = [r for r in results if r.status == "missing"]

    logger.info(f"\n{'Domain':<25} {'κ':>12} {'Raw':>8} {'Primed':>8} {'Status':>12}")
    logger.info("-" * 70)

    for r in sorted(results, key=lambda x: x.kappa):
        logger.info(f"{r.name:<25} {r.kappa:>12.2e} {r.accuracy_raw:>8.0%} {r.accuracy_primed:>8.0%} {r.status:>12}")

    # Compute statistics
    if working:
        kappa_working = [r.kappa for r in working]
        logger.info(f"\nWorking domains ({len(working)}):")
        logger.info(f"  κ range: {min(kappa_working):.2e} - {max(kappa_working):.2e}")
        logger.info(f"  κ mean: {np.mean(kappa_working):.2e}")

    if disconnected:
        kappa_disconnected = [r.kappa for r in disconnected]
        logger.info(f"\nDisconnected domains ({len(disconnected)}):")
        logger.info(f"  κ range: {min(kappa_disconnected):.2e} - {max(kappa_disconnected):.2e}")
        logger.info(f"  κ mean: {np.mean(kappa_disconnected):.2e}")

    if missing:
        kappa_missing = [r.kappa for r in missing]
        logger.info(f"\nMissing domains ({len(missing)}):")
        logger.info(f"  κ range: {min(kappa_missing):.2e} - {max(kappa_missing):.2e}")
        logger.info(f"  κ mean: {np.mean(kappa_missing):.2e}")

    # Find threshold
    logger.info("\n=== THRESHOLD ANALYSIS ===")

    all_kappas = [(r.kappa, r.status, r.name) for r in results]
    all_kappas.sort(key=lambda x: x[0])

    # Try different thresholds
    thresholds_to_try = [10, 50, 100, 200, 500, 1000]

    logger.info(f"\n{'Threshold':>10} {'Working→Working':>18} {'Disconn→Disconn':>18} {'Missing→Missing':>18}")
    logger.info("-" * 70)

    best_threshold = None
    best_score = 0

    for threshold in thresholds_to_try:
        # For each threshold, classify domains
        correct_working = sum(1 for r in results if r.kappa <= threshold and r.status == "working")
        correct_disconnected = sum(1 for r in results if r.kappa > threshold and r.status == "disconnected")
        correct_missing = sum(1 for r in results if r.status == "missing")  # Can't detect via κ alone

        total_working = len(working)
        total_disconnected = len(disconnected)
        total_missing = len(missing)

        w_rate = correct_working / total_working if total_working > 0 else 0
        d_rate = correct_disconnected / total_disconnected if total_disconnected > 0 else 0
        m_rate = 1.0  # Missing is a different category

        score = w_rate * 0.4 + d_rate * 0.4 + 0.2  # Weight the metrics

        logger.info(f"{threshold:>10} {correct_working}/{total_working:>15} {correct_disconnected}/{total_disconnected:>15} {correct_missing}/{total_missing:>15}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    logger.info(f"\nBest threshold: κ = {best_threshold}")

    # Correlation analysis
    logger.info("\n=== CORRELATION: κ vs ACCURACY ===")

    kappas = np.array([r.kappa for r in results])
    raw_accs = np.array([r.accuracy_raw for r in results])
    primed_accs = np.array([r.accuracy_primed for r in results])

    # Use log κ for correlation (κ varies over orders of magnitude)
    log_kappas = np.log10(kappas + 1)

    corr_raw = np.corrcoef(log_kappas, raw_accs)[0, 1]
    corr_primed = np.corrcoef(log_kappas, primed_accs)[0, 1]

    logger.info(f"Correlation(log κ, raw accuracy): {corr_raw:.3f}")
    logger.info(f"Correlation(log κ, primed accuracy): {corr_primed:.3f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: CAN κ DETECT DISCONNECTED CAPABILITIES?")
    logger.info("=" * 60)

    separation_possible = True
    if working and disconnected:
        max_working_kappa = max(r.kappa for r in working)
        min_disconnected_kappa = min(r.kappa for r in disconnected)
        if max_working_kappa < min_disconnected_kappa:
            logger.info(f"\n✓ CLEAR SEPARATION EXISTS")
            logger.info(f"  Working max κ: {max_working_kappa:.2e}")
            logger.info(f"  Disconnected min κ: {min_disconnected_kappa:.2e}")
            logger.info(f"  Threshold range: ({max_working_kappa:.2e}, {min_disconnected_kappa:.2e})")
        else:
            logger.info(f"\n✗ NO CLEAR SEPARATION")
            logger.info(f"  Working max κ: {max_working_kappa:.2e}")
            logger.info(f"  Disconnected min κ: {min_disconnected_kappa:.2e}")
            logger.info(f"  Overlap exists")
            separation_possible = False

    if corr_raw < -0.3:
        logger.info(f"\n✓ NEGATIVE CORRELATION: Higher κ → Lower raw accuracy")
    else:
        logger.info(f"\n? WEAK CORRELATION: κ may not reliably predict accuracy")

    # Final verdict
    if separation_possible and corr_raw < -0.3:
        logger.info(f"\n*** κ CAN BE USED FOR SELF-DETECTION ***")
        logger.info(f"Recommended threshold: κ > {best_threshold}")
        logger.info(f"Domains with κ > {best_threshold} are likely disconnected and need bridges")
    else:
        logger.info(f"\n*** κ ALONE MAY NOT BE SUFFICIENT ***")
        logger.info(f"Consider combining with other geometric metrics")

    # Save results
    output_path = "data/experiments/geometric_self_detection.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "domains": [
            {
                "name": r.name,
                "kappa": float(r.kappa),
                "accuracy_raw": float(r.accuracy_raw),
                "accuracy_primed": float(r.accuracy_primed),
                "prime": r.prime_used,
                "status": r.status,
            }
            for r in results
        ],
        "analysis": {
            "best_threshold": best_threshold,
            "correlation_raw": float(corr_raw),
            "correlation_primed": float(corr_primed),
            "separation_possible": separation_possible,
        },
        "summary": {
            "working_count": len(working),
            "disconnected_count": len(disconnected),
            "missing_count": len(missing),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
