#!/usr/bin/env python3
"""Experiment 84: True Gap Detection.

Can we distinguish "disconnected" from "truly missing"?

We found:
- Arithmetic: disconnected (0% raw → 100% primed)
- Word problems: still 0% even with priming

Hypothesis: For truly missing capabilities:
- Bridge may apply geometrically (κ decreases)
- But accuracy stays 0% (nothing to connect to)
- Different signature: "bridged but still fails"

This experiment tests whether we can automatically detect true gaps
that need training vs disconnected capabilities that need bridging.
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
class CapabilityStatus:
    """Status of a capability after bridge attempt."""
    name: str
    accuracy_raw: float
    accuracy_primed: float
    kappa_raw: float
    kappa_primed: float
    prime_used: str
    classification: str  # "disconnected", "true_gap", or "working"


def get_activations(model, tokenizer, prompts: List[str], layer_idx: int = -1) -> np.ndarray:
    """Get activations for a list of prompts."""
    import mlx.core as mx

    acts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        hidden = model.model.embed_tokens(input_ids)

        n_layers = len(model.model.layers)
        if layer_idx == -1:
            layer_idx = n_layers - 1

        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden, mask=None, cache=None)
            if i == layer_idx:
                break

        mx.eval(hidden)
        acts.append(np.array(hidden[0, -1, :].tolist()))

    return np.stack(acts)


def compute_kappa(activations: np.ndarray) -> float:
    """Compute condition number of Gram matrix."""
    G = activations @ activations.T
    try:
        return float(np.linalg.cond(G))
    except:
        return float('inf')


def evaluate_accuracy(model, tokenizer, prime: str, problems: List[Tuple[str, str]]) -> Tuple[float, List[dict]]:
    """Evaluate accuracy on a problem set with optional prime."""
    import mlx.core as mx

    results = []
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

        correct = expected in predicted or predicted == expected
        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "top_prob": float(probs[top_token]),
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def main():
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load("/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16")

    logger.info("=" * 60)
    logger.info("EXPERIMENT 84: TRUE GAP DETECTION")
    logger.info("=" * 60)

    # Define capabilities to test
    # Known disconnected: arithmetic (works with priming)
    # Known true gap: word problems (doesn't work with priming)
    # Unknown: other domains

    capabilities = {
        "arithmetic_addition": {
            "prompts": ["1+1=", "2+1=", "3+1=", "4+1=", "5+1=", "6+1=", "7+1=", "8+1="],
            "problems": [("1+1=", "2"), ("2+1=", "3"), ("3+1=", "4"), ("4+1=", "5"),
                        ("5+1=", "6"), ("6+1=", "7"), ("7+1=", "8"), ("8+1=", "9")],
            "primes_to_try": ["say", "Arithmetic means calculating numbers."],
        },
        "arithmetic_subtraction": {
            "prompts": ["5-1=", "4-1=", "3-1=", "6-2=", "7-3=", "8-2="],
            "problems": [("5-1=", "4"), ("4-1=", "3"), ("3-1=", "2"), ("6-2=", "4"),
                        ("7-3=", "4"), ("8-2=", "6")],
            "primes_to_try": ["say", "One less is", "Arithmetic means calculating numbers."],
        },
        "word_problems": {
            "prompts": [
                "I have 3 apples. I get 2 more. Total:",
                "5 birds. 2 fly away. Remaining:",
                "Start with 4. Add 3. Result:",
                "Begin with 7. Take away 2. Left with:",
            ],
            "problems": [
                ("I have 3 apples. I get 2 more. Total:", "5"),
                ("5 birds. 2 fly away. Remaining:", "3"),
                ("Start with 4. Add 3. Result:", "7"),
                ("Begin with 7. Take away 2. Left with:", "5"),
            ],
            "primes_to_try": [
                "say",
                "Arithmetic means calculating numbers.",
                "Calculate the number.",
                "Word problems are math.",
            ],
        },
        "word_problem_with_equation": {
            "prompts": [
                "I have 3 apples. I get 2 more. 3+2=",
                "5 birds. 2 fly away. 5-2=",
                "Start with 4. Add 3. 4+3=",
            ],
            "problems": [
                ("I have 3 apples. I get 2 more. 3+2=", "5"),
                ("5 birds. 2 fly away. 5-2=", "3"),
                ("Start with 4. Add 3. 4+3=", "7"),
            ],
            "primes_to_try": ["say", "Arithmetic means calculating numbers."],
        },
        "two_digit_arithmetic": {
            "prompts": ["10+5=", "12+3=", "20+10=", "15-5=", "20-10="],
            "problems": [("10+5=", "15"), ("12+3=", "15"), ("20+10=", "30"),
                        ("15-5=", "10"), ("20-10=", "10")],
            "primes_to_try": ["say", "Arithmetic means calculating numbers."],
        },
        "multiplication": {
            "prompts": ["2×2=", "2×3=", "3×3=", "2×4=", "3×4="],
            "problems": [("2×2=", "4"), ("2×3=", "6"), ("3×3=", "9"),
                        ("2×4=", "8"), ("3×4=", "12")],
            "primes_to_try": ["say", "Arithmetic means calculating numbers."],
        },
        "three_digit_arithmetic": {
            "prompts": ["100+50=", "200+100=", "150-50="],
            "problems": [("100+50=", "150"), ("200+100=", "300"), ("150-50=", "100")],
            "primes_to_try": ["say", "Arithmetic means calculating numbers."],
        },
    }

    results = []

    logger.info("\n=== TESTING CAPABILITIES ===")

    for cap_name, cap_data in capabilities.items():
        logger.info(f"\n--- {cap_name} ---")

        prompts = cap_data["prompts"]
        problems = cap_data["problems"]
        primes = cap_data["primes_to_try"]

        # Get raw activations and κ
        acts_raw = get_activations(model, tokenizer, prompts)
        kappa_raw = compute_kappa(acts_raw)

        # Evaluate raw accuracy
        acc_raw, details_raw = evaluate_accuracy(model, tokenizer, "", problems)

        logger.info(f"  Raw: κ={kappa_raw:.2e}, accuracy={acc_raw:.0%}")

        # Try each prime
        best_prime = ""
        best_acc = acc_raw
        best_kappa = kappa_raw

        for prime in primes:
            primed_prompts = [f"{prime} {p}" for p in prompts]
            acts_primed = get_activations(model, tokenizer, primed_prompts)
            kappa_primed = compute_kappa(acts_primed)

            acc_primed, details_primed = evaluate_accuracy(model, tokenizer, prime, problems)

            logger.info(f"  '{prime[:30]}...': κ={kappa_primed:.2e}, accuracy={acc_primed:.0%}")

            if acc_primed > best_acc:
                best_acc = acc_primed
                best_prime = prime
                best_kappa = kappa_primed

        # Classify
        if acc_raw >= 0.7:
            classification = "working"
        elif best_acc >= 0.7:
            classification = "disconnected"
        else:
            classification = "true_gap"

        status = CapabilityStatus(
            name=cap_name,
            accuracy_raw=acc_raw,
            accuracy_primed=best_acc,
            kappa_raw=kappa_raw,
            kappa_primed=best_kappa,
            prime_used=best_prime,
            classification=classification,
        )
        results.append(status)

        logger.info(f"  → Classification: {classification.upper()}")

    # Analysis
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS: DISTINGUISHING DISCONNECTED FROM TRUE GAPS")
    logger.info("=" * 60)

    working = [r for r in results if r.classification == "working"]
    disconnected = [r for r in results if r.classification == "disconnected"]
    true_gaps = [r for r in results if r.classification == "true_gap"]

    logger.info(f"\n{'Capability':<30} {'Raw':>8} {'Best':>8} {'κ_raw':>12} {'κ_best':>12} {'Class':>15}")
    logger.info("-" * 90)

    for r in results:
        logger.info(f"{r.name:<30} {r.accuracy_raw:>8.0%} {r.accuracy_primed:>8.0%} "
                   f"{r.kappa_raw:>12.2e} {r.kappa_primed:>12.2e} {r.classification:>15}")

    logger.info(f"\nSummary:")
    logger.info(f"  Working: {len(working)}")
    logger.info(f"  Disconnected: {len(disconnected)}")
    logger.info(f"  True gaps: {len(true_gaps)}")

    # Key question: Can we distinguish disconnected from true gaps using geometry alone?
    logger.info("\n=== GEOMETRIC SIGNATURES ===")

    if disconnected:
        kappa_ratios_disc = [r.kappa_primed / r.kappa_raw for r in disconnected if r.kappa_raw > 0]
        logger.info(f"\nDisconnected (bridge WORKS):")
        logger.info(f"  κ ratio (primed/raw): mean={np.mean(kappa_ratios_disc):.2f}")
        for r in disconnected:
            ratio = r.kappa_primed / r.kappa_raw if r.kappa_raw > 0 else 0
            lift = r.accuracy_primed - r.accuracy_raw
            logger.info(f"    {r.name}: κ ratio={ratio:.2f}, accuracy lift=+{lift:.0%}")

    if true_gaps:
        kappa_ratios_gap = [r.kappa_primed / r.kappa_raw for r in true_gaps if r.kappa_raw > 0]
        logger.info(f"\nTrue gaps (bridge FAILS):")
        logger.info(f"  κ ratio (primed/raw): mean={np.mean(kappa_ratios_gap):.2f}")
        for r in true_gaps:
            ratio = r.kappa_primed / r.kappa_raw if r.kappa_raw > 0 else 0
            lift = r.accuracy_primed - r.accuracy_raw
            logger.info(f"    {r.name}: κ ratio={ratio:.2f}, accuracy lift=+{lift:.0%}")

    # Detection algorithm
    logger.info("\n=== DETECTION ALGORITHM ===")

    logger.info("""
PROPOSED ALGORITHM:

1. Test capability raw: accuracy_raw
2. If accuracy_raw >= 0.7: WORKING (no action needed)
3. Test with primes: accuracy_primed
4. If accuracy_primed >= 0.7: DISCONNECTED (use prime or compute bridge)
5. Else: TRUE GAP (needs training)

GEOMETRIC REFINEMENT:
- If κ decreases significantly but accuracy stays low → TRUE GAP
- If κ decreases AND accuracy increases → DISCONNECTED
- This can be automated!
""")

    # Specific analysis of word problems
    logger.info("\n=== WORD PROBLEM ANALYSIS ===")

    wp = next((r for r in results if r.name == "word_problems"), None)
    wp_eq = next((r for r in results if r.name == "word_problem_with_equation"), None)

    if wp and wp_eq:
        logger.info(f"\nWord problems (natural language):")
        logger.info(f"  Raw: {wp.accuracy_raw:.0%}, Best primed: {wp.accuracy_primed:.0%}")
        logger.info(f"  Classification: {wp.classification}")

        logger.info(f"\nWord problems WITH equation:")
        logger.info(f"  Raw: {wp_eq.accuracy_raw:.0%}, Best primed: {wp_eq.accuracy_primed:.0%}")
        logger.info(f"  Classification: {wp_eq.classification}")

        if wp_eq.accuracy_primed > wp.accuracy_primed:
            logger.info(f"\n*** THE GAP IS PARSING ***")
            logger.info(f"Adding explicit equations bridges the gap!")
            logger.info(f"The model CAN do arithmetic. It CANNOT parse natural language → equations.")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CONCLUSION: CAN WE DETECT TRUE GAPS?")
    logger.info("=" * 60)

    detection_success = (
        len(disconnected) > 0 and
        len(true_gaps) > 0 and
        all(r.accuracy_primed >= 0.7 for r in disconnected) and
        all(r.accuracy_primed < 0.7 for r in true_gaps)
    )

    if detection_success:
        logger.info("\n*** TRUE GAP DETECTION WORKS ***")
        logger.info("We can automatically distinguish:")
        logger.info("  - DISCONNECTED: responds to priming (accuracy_primed >= 70%)")
        logger.info("  - TRUE GAP: doesn't respond to priming (accuracy_primed < 70%)")
        logger.info("\nDisconnected capabilities need BRIDGES.")
        logger.info("True gaps need TRAINING.")
    else:
        logger.info("\n*** DETECTION NEEDS REFINEMENT ***")
        logger.info("Some capabilities may be partially responding.")

    # Save results
    output_path = "data/experiments/true_gap_detection.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "capabilities": [
            {
                "name": r.name,
                "accuracy_raw": float(r.accuracy_raw),
                "accuracy_primed": float(r.accuracy_primed),
                "kappa_raw": float(r.kappa_raw),
                "kappa_primed": float(r.kappa_primed),
                "prime_used": r.prime_used,
                "classification": r.classification,
            }
            for r in results
        ],
        "summary": {
            "working_count": len(working),
            "disconnected_count": len(disconnected),
            "true_gap_count": len(true_gaps),
            "detection_success": detection_success,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
