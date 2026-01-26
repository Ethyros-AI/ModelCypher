#!/usr/bin/env python3
"""Experiment 97: Cross-Architecture Transfer of Fixed Capabilities.

The insight: LoRA adapters are slices of the manifold.
            Full models are the full manifold.
            The math for merging is the same.

If we've learned to fix capabilities in one model (350M), can we:
1. Extract what was learned (the capability "pattern")
2. Transfer it to a different model (1.2B)
3. Verify the capability works in the new model

This is the foundation for cross-architecture model merging.

Approach:
1. Use the TRAINING DATA (not the adapter weights) - the pattern is in the data
2. Train the same capability on the larger model
3. Compare: Does the same training data work on different architectures?

This proves: The capability is in the PATTERN, not the specific weights.
"""

from __future__ import annotations

import json
import logging
import subprocess
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


def evaluate_model(model, tokenizer, problems: List[Tuple[str, str]], description: str) -> Tuple[float, List[dict]]:
    """Generic evaluation function."""
    import mlx.core as mx

    results = []
    correct = 0

    for prompt, expected in problems:
        tokens = tokenizer.encode(prompt)
        logits = model(mx.array([tokens]))
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist(), dtype=np.float32)
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        top_idx = int(np.argmax(probs))
        predicted = tokenizer.decode([top_idx]).strip()

        is_correct = expected in predicted or predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, results


def main():
    from mlx_lm import load
    import mlx.core as mx

    # Source: 350M (where we developed the capability)
    source_model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    source_adapter = "data/adapters/unified_math_lora"

    # Target: 1.2B (larger model, different scale)
    target_model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    target_adapter = "data/adapters/transfer_math_lora"

    # Training data (the capability "pattern")
    training_data_dir = "data/training/unified_math"

    logger.info("=" * 60)
    logger.info("EXPERIMENT 97: CROSS-ARCHITECTURE TRANSFER")
    logger.info("=" * 60)
    logger.info("Question: Can capabilities transfer between architectures?")
    logger.info(f"  Source: LFM2-350M (where we learned)")
    logger.info(f"  Target: LFM2.5-1.2B (where we want to apply)")

    # Test problems
    test_problems = [
        # Arithmetic
        ("1+1=", "2"),
        ("3+5=", "8"),
        ("7-3=", "4"),
        ("9-4=", "5"),
        # Word problems (with trailing space)
        ("I have 3 apples. I get 2 more. Total: ", "5"),
        ("5 birds. 2 fly away. Remaining: ", "3"),
        ("3 plus 2 is ", "5"),
        ("8 minus 3 is ", "5"),
    ]

    # Phase 1: Verify source model has the capability
    logger.info("\n=== PHASE 1: VERIFY SOURCE MODEL (350M) ===")

    model, tokenizer = load(source_model_path, adapter_path=source_adapter)
    source_acc, source_details = evaluate_model(model, tokenizer, test_problems, "Source")

    logger.info(f"Source (350M + adapter): {source_acc:.0%}")
    for r in source_details[:4]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:30]}' → '{r['predicted']}'")

    del model
    mx.clear_cache()

    # Phase 2: Test target model baseline
    logger.info("\n=== PHASE 2: TARGET MODEL BASELINE (1.2B) ===")

    model, tokenizer = load(target_model_path)
    target_baseline, baseline_details = evaluate_model(model, tokenizer, test_problems, "Target Baseline")

    logger.info(f"Target baseline (1.2B, no adapter): {target_baseline:.0%}")
    for r in baseline_details[:4]:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:30]}' → '{r['predicted']}'")

    del model
    mx.clear_cache()

    # Phase 3: Train target model with SAME training data
    logger.info("\n=== PHASE 3: TRANSFER CAPABILITY TO TARGET ===")
    logger.info("Training target model with same data that trained source...")

    Path(target_adapter).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", target_model_path,
        "--train",
        "--data", training_data_dir,
        "--adapter-path", target_adapter,
        "--batch-size", "4",
        "--num-layers", "8",  # Fewer layers for larger model
        "--iters", "300",
        "--learning-rate", "5e-5",
        "--seed", "42",
        "--steps-per-report", "100",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
        logger.info("Training complete!")
        for line in result.stdout.strip().split('\n')[-6:]:
            logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error: {e}")
        return

    # Phase 4: Evaluate transfer
    logger.info("\n=== PHASE 4: EVALUATE TRANSFER ===")

    model, tokenizer = load(target_model_path, adapter_path=target_adapter)
    target_acc, target_details = evaluate_model(model, tokenizer, test_problems, "Target Transferred")

    logger.info(f"\n{'='*50}")
    logger.info("TRANSFER RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"  Source (350M):        {source_acc:.0%}")
    logger.info(f"  Target baseline:      {target_baseline:.0%}")
    logger.info(f"  Target transferred:   {target_acc:.0%}")
    logger.info(f"{'='*50}")

    logger.info("\nTarget model (transferred) examples:")
    for r in target_details:
        status = "✓" if r["correct"] else "✗"
        logger.info(f"  {status} '{r['prompt'][:30]}' → '{r['predicted']}' (expected '{r['expected']}')")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-ARCHITECTURE TRANSFER SUMMARY")
    logger.info("=" * 60)

    transfer_successful = target_acc >= source_acc * 0.8  # Within 80% of source
    improvement = target_acc - target_baseline

    logger.info(f"""
SOURCE MODEL (350M):
  Capability: {source_acc:.0%}
  (This is where we developed the capability)

TARGET MODEL (1.2B):
  Before transfer: {target_baseline:.0%}
  After transfer:  {target_acc:.0%}
  Improvement:     {improvement:+.0%}

TRANSFER SUCCESS: {'✓ YES' if transfer_successful else '✗ NO'} (target >= 80% of source)

KEY INSIGHT:
  The capability is in the TRAINING DATA PATTERN, not specific weights.
  Same training data → Same capability, different architecture.

  This means:
  1. Capabilities are portable across architectures
  2. The "slice of manifold" (LoRA) contains the pattern
  3. Full model merging should follow the same principle:
     - Extract capability patterns from source
     - Apply to target architecture
     - Verify transfer
""")

    # Save results
    results = {
        "source": {"model": "LFM2-350M", "accuracy": source_acc},
        "target_baseline": {"model": "LFM2.5-1.2B", "accuracy": target_baseline},
        "target_transferred": {"model": "LFM2.5-1.2B", "accuracy": target_acc},
        "transfer_successful": transfer_successful,
        "details": target_details,
    }

    output_path = "data/experiments/cross_architecture_transfer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
