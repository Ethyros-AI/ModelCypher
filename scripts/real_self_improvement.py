#!/usr/bin/env python3
"""Experiment 90: Real Self-Improvement on LFM2.5-1.2B.

This is the real test. Can the model:
1. Identify what it knows (oracle capabilities)
2. Identify what it doesn't know (gaps)
3. Generate verified training data for gaps
4. Specify training that would fill the gaps

Key insight: Pre-training is kindergarten. This is how the model
continues learning - using what it knows to verify what it learns.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add src to path for library imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.core.use_cases.self_improve import (
    AutonomousSelfImprover,
    Capability,
    CapabilityScanner,
    CapabilityStatus,
    VerificationOracle,
    SafeSelfPlayGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_model_self_awareness(model, tokenizer):
    """Can the model answer questions about what it knows?"""
    import mlx.core as mx

    logger.info("\n" + "=" * 60)
    logger.info("TEST: MODEL SELF-AWARENESS")
    logger.info("=" * 60)

    # Questions the model should be able to answer about itself
    self_questions = [
        "What is 2+2?",
        "What is 5-3?",
        "I have 3 apples and get 2 more. How many do I have?",
        "Complete: The capital of France is",
        "Complete: 1, 2, 3, 4,",
    ]

    logger.info("\nAsking model about its own capabilities:")
    for question in self_questions:
        tokens = tokenizer.encode(question)
        input_ids = mx.array([tokens])

        # Generate a few tokens
        generated = []
        for _ in range(10):
            logits = model(mx.array([tokens + generated]))
            mx.eval(logits)

            import numpy as np
            logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
            probs = np.exp(logits_np - logits_np.max())
            probs = probs / probs.sum()

            next_token = int(np.argmax(probs))
            generated.append(next_token)

            # Stop on newline or period
            text = tokenizer.decode([next_token])
            if '\n' in text or text.strip() in ['.', '?', '!']:
                break

        response = tokenizer.decode(generated).strip()
        logger.info(f"  Q: {question}")
        logger.info(f"  A: {response}")


def main():
    from mlx_lm import load

    # Use LFM2.5-1.2B-Instruct - the most capable model available
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"

    logger.info("Loading model...")
    logger.info(f"  Path: {model_path}")
    model, tokenizer = load(model_path)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 90: REAL SELF-IMPROVEMENT")
    logger.info("=" * 60)
    logger.info(f"Model: LFM2.5-1.2B-Instruct")
    logger.info(f"Goal: Test if the model can improve itself")

    # First, test basic self-awareness
    test_model_self_awareness(model, tokenizer)

    # Define capabilities to test
    capabilities = [
        # Arithmetic - the oracle capability
        Capability.from_lists(
            name="arithmetic",
            prompts=["1+1=", "2+2=", "3+3=", "4+4=", "5+5="],
            problems=[
                ("1+1=", "2"),
                ("2+2=", "4"),
                ("3+1=", "4"),
                ("5+2=", "7"),
                ("4+3=", "7"),
                ("6+1=", "7"),
                ("3+3=", "6"),
                ("2+5=", "7"),
            ],
        ),
        # Subtraction
        Capability.from_lists(
            name="subtraction",
            prompts=["5-2=", "4-1=", "7-3=", "6-2=", "9-4="],
            problems=[
                ("5-2=", "3"),
                ("4-1=", "3"),
                ("7-3=", "4"),
                ("6-2=", "4"),
                ("9-4=", "5"),
                ("8-3=", "5"),
                ("5-1=", "4"),
                ("7-2=", "5"),
            ],
        ),
        # Word problems - the gap we want to fill
        Capability.from_lists(
            name="word_problems",
            prompts=[
                "I have 3 apples. I get 2 more. Total:",
                "5 birds. 2 fly away. Remaining:",
                "Start with 4. Add 3. Result:",
            ],
            problems=[
                ("I have 3 apples. I get 2 more. Total:", "5"),
                ("5 birds. 2 fly away. Remaining:", "3"),
                ("Start with 4. Add 3. Result:", "7"),
                ("There are 6 cats. 2 leave. Left:", "4"),
                ("Mary has 2 toys. She gets 5 more. Total:", "7"),
                ("Begin with 8. Take away 3. Result:", "5"),
            ],
        ),
        # Counting - baseline working capability
        Capability.from_lists(
            name="counting",
            prompts=["1, 2, 3, 4,", "2, 4, 6,", "10, 9, 8,"],
            problems=[
                ("1, 2, 3, 4,", "5"),
                ("2, 4, 6,", "8"),
                ("10, 9, 8,", "7"),
                ("5, 6, 7,", "8"),
            ],
        ),
    ]

    # Phase 1: Scan capabilities
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: SCANNING CAPABILITIES")
    logger.info("=" * 60)

    scanner = CapabilityScanner(model, tokenizer)

    analyses = []
    for cap in capabilities:
        logger.info(f"\nScanning: {cap.name}")
        analysis = scanner.scan(cap)
        analyses.append(analysis)

        status_icon = (
            "✓" if analysis.status == CapabilityStatus.WORKING
            else "⚡" if analysis.status == CapabilityStatus.DISCONNECTED
            else "✗"
        )
        logger.info(f"  Status: {status_icon} {analysis.status.value.upper()}")
        logger.info(f"  Raw accuracy: {analysis.accuracy_raw:.0%}")
        logger.info(f"  Primed accuracy: {analysis.accuracy_primed:.0%}")
        logger.info(f"  Best prime: '{analysis.best_prime}'")
        logger.info(f"  κ(raw): {analysis.kappa_raw:.1f}")
        logger.info(f"  κ(primed): {analysis.kappa_primed:.1f}")

    # Phase 2: Identify oracle and gaps
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: IDENTIFYING ORACLE AND GAPS")
    logger.info("=" * 60)

    working = [a for a in analyses if a.status == CapabilityStatus.WORKING]
    disconnected = [a for a in analyses if a.status == CapabilityStatus.DISCONNECTED]
    true_gaps = [a for a in analyses if a.status == CapabilityStatus.TRUE_GAP]

    logger.info(f"\nWorking (potential oracles): {[a.capability.name for a in working]}")
    logger.info(f"Disconnected (need priming): {[a.capability.name for a in disconnected]}")
    logger.info(f"True gaps (need training): {[a.capability.name for a in true_gaps]}")

    # Find best oracle
    oracle_capability = None
    oracle_prime = ""
    best_oracle_acc = 0

    for a in working + disconnected:
        if a.accuracy_primed > best_oracle_acc:
            best_oracle_acc = a.accuracy_primed
            oracle_capability = a.capability.name
            oracle_prime = a.best_prime

    logger.info(f"\nBest oracle: {oracle_capability} ({best_oracle_acc:.0%} with prime '{oracle_prime}')")

    # Phase 3: Calibrate verification oracle
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: CALIBRATING VERIFICATION ORACLE")
    logger.info("=" * 60)

    oracle = VerificationOracle(model, tokenizer, prime=oracle_prime if oracle_prime else None)
    calibration_tests = oracle.default_calibration_tests()
    accuracy, details = oracle.calibrate(calibration_tests)

    logger.info(f"\nOracle calibration results:")
    for eq, expected, computed, correct in details:
        status = "✓" if correct else "✗"
        logger.info(f"  {status} {eq} → '{computed}' (expected '{expected}')")

    logger.info(f"\nOracle calibration accuracy: {accuracy:.0%}")

    if accuracy < 0.9:
        logger.warning("*** ORACLE NOT RELIABLE ENOUGH ***")
        logger.warning("Cannot safely generate training data.")
    else:
        logger.info("*** ORACLE IS CALIBRATED AND RELIABLE ***")

        # Phase 4: Generate verified training data
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: GENERATING VERIFIED TRAINING DATA")
        logger.info("=" * 60)

        generator = SafeSelfPlayGenerator(oracle)
        samples = generator.generate_verified(n_samples=100, seed=42)
        stats = generator.get_statistics(samples)

        logger.info(f"\nGenerated {stats['total']} verified samples")
        logger.info(f"  Addition: {stats['addition']}")
        logger.info(f"  Subtraction: {stats['subtraction']}")

        logger.info("\nSample training data:")
        for sample in samples[:5]:
            logger.info(f"  Input:  '{sample.input_text}'")
            logger.info(f"  Output: '{sample.output_text}{sample.answer}'")

        # Save training data
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "lfm25_self_play.jsonl"
        generator.save_jsonl(samples, train_path)
        logger.info(f"\nTraining data saved to: {train_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: REAL SELF-IMPROVEMENT ANALYSIS")
    logger.info("=" * 60)

    logger.info(f"""
MODEL: LFM2.5-1.2B-Instruct

CAPABILITY ANALYSIS:
  Working: {len(working)} capabilities
  Disconnected: {len(disconnected)} capabilities
  True Gaps: {len(true_gaps)} capabilities

ORACLE:
  Best oracle: {oracle_capability}
  Calibration: {accuracy:.0%}
  Prime: '{oracle_prime}'

SELF-IMPROVEMENT PATH:
  1. Use {oracle_capability} as verification oracle
  2. Generate verified training data for gaps
  3. Train LoRA on early layers
  4. Verify no regression on oracle capabilities

THIS IS THE KEY INSIGHT:
  The model KNOWS arithmetic (with priming).
  It can use this knowledge to VERIFY training data.
  This means it can learn safely - no nonsense.

  Pre-training gave it kindergarten.
  Self-improvement lets it continue learning.
""")

    # Save full results
    results_path = "data/experiments/real_self_improvement.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "LFM2.5-1.2B-Instruct",
        "capabilities": [a.to_dict() for a in analyses],
        "oracle": {
            "capability": oracle_capability,
            "prime": oracle_prime,
            "calibration_accuracy": accuracy,
        },
        "training_data": {
            "samples": stats['total'] if accuracy >= 0.9 else 0,
            "path": str(train_path) if accuracy >= 0.9 else None,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
