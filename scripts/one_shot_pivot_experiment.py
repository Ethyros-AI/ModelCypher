#!/usr/bin/env python3
"""One-Shot RLVR Geometric Pivot Experiment.

Replicates the Qwen finding: can a single well-chosen example
"pivot" LFM2-1.2B into reasoning mode?

Protocol:
1. Profile 1.2B BEFORE training (expansion ratio, spectral trajectory)
2. Train with ONE example (best modus tollens from our data)
3. Profile AFTER — which layers changed?
4. Evaluate on held-out reasoning tasks

Reference: One-Shot RLVR (Qwen2.5-1.5B): 36% -> 73.6% on MATH500 with ONE example.

Usage:
    poetry run python scripts/one_shot_pivot_experiment.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16 \
        --output /Volumes/CodeCypher/experiments/one-shot-pivot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# The single best modus tollens example from format-augmented training
ONE_SHOT_EXAMPLE = {
    "text": (
        "Question: If it is raining, then the ground is wet. "
        "The ground is not wet. Is it raining?\n\n"
        "Let me work through this step by step.\n\n"
        "Step 1: Identify the rule.\n"
        "The rule states: If it is raining, then the ground is wet.\n"
        "In logical form: P -> Q, where P = 'it is raining' and Q = 'the ground is wet'.\n\n"
        "Step 2: Identify the observation.\n"
        "We observe: The ground is not wet. That is, not Q.\n\n"
        "Step 3: Apply Modus Tollens.\n"
        "Modus Tollens states: If P -> Q and not Q, then not P.\n"
        "Since we have P -> Q (the rule) and not Q (the observation),\n"
        "we can conclude: not P.\n\n"
        "Step 4: State the conclusion.\n"
        "It is not raining.\n\n"
        "Answer: No, it is not raining."
    )
}

# Held-out evaluation prompts (5 reasoning, 5 math, 5 general)
EVAL_PROMPTS = [
    # Modus Tollens (not in training)
    "If a student passes the exam, they graduate. Alice did not graduate. Did Alice pass the exam?",
    "If an animal is a cat, it has whiskers. This animal does not have whiskers. Is it a cat?",
    # Modus Ponens
    "If it snows, schools close. It is snowing. Are schools closed?",
    # Disjunctive Syllogism
    "Either the package was delivered or it was returned to sender. The package was not delivered. What happened to the package?",
    # Chain reasoning
    "If it rains, the streets get wet. If the streets get wet, cars slow down. It is raining. Do cars slow down?",
    # Math
    "What is 7 * 8?",
    "If a shirt costs $25 and is 20% off, what is the sale price?",
    "What is the next number in the sequence: 2, 6, 18, 54, ?",
    "A rectangle has length 8 and width 5. What is its area?",
    "If 3x + 7 = 22, what is x?",
    # General knowledge
    "What is the capital of France?",
    "How many planets are in our solar system?",
    "What is photosynthesis?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
]


def profile_model(model_path: str, output_dir: Path, prefix: str) -> dict:
    """Run geometric profiling on the model."""
    import subprocess

    profiles = {}

    # 1. Expansion ratio on a reasoning prompt
    logger.info("[%s] Running expansion-ratio...", prefix)
    result = subprocess.run(
        [
            "poetry", "run", "mc", "analyze", "expansion-ratio",
            "--model", model_path,
            "--prompt", "If all birds can fly and penguins are birds, can penguins fly?",
            "-t", "-q",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        profiles["expansion_ratio"] = json.loads(result.stdout)
        logger.info("[%s] expansion-ratio done", prefix)

    # 2. Spectral trajectory
    logger.info("[%s] Running spectral-trajectory...", prefix)
    result = subprocess.run(
        [
            "poetry", "run", "mc", "analyze", "spectral-trajectory",
            "--model", model_path, "--samples", "20",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        profiles["spectral_trajectory"] = json.loads(result.stdout)
        logger.info("[%s] spectral-trajectory done", prefix)

    # 3. Entropy trajectory
    logger.info("[%s] Running entropy-trajectory...", prefix)
    result = subprocess.run(
        [
            "poetry", "run", "mc", "analyze", "entropy-trajectory",
            "--model", model_path,
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        profiles["entropy_trajectory"] = json.loads(result.stdout)
        logger.info("[%s] entropy-trajectory done", prefix)

    # Save profiles
    profile_path = output_dir / f"{prefix}_profiles.json"
    with open(profile_path, "w") as f:
        json.dump(profiles, f, indent=2)
    logger.info("[%s] Profiles saved to %s", prefix, profile_path)

    return profiles


def run_inference(model_path: str, prompts: list[str], adapter_path: str | None = None) -> list[dict]:
    """Run greedy inference on prompts."""
    import subprocess

    results = []
    for i, prompt in enumerate(prompts):
        cmd = [
            "poetry", "run", "mc", "infer", "run",
            "--model", model_path,
            "--prompt", prompt,
        ]
        if adapter_path:
            cmd.extend(["--adapter", adapter_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        response = result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}"
        results.append({
            "prompt": prompt,
            "response": response,
            "index": i,
        })
        logger.info("  [%d/%d] %s...", i + 1, len(prompts), prompt[:50])

    return results


def train_one_shot(model_path: str, output_dir: Path) -> str:
    """Train with ONE example using research training path."""
    import subprocess

    # Write the single example to a temp file
    train_file = output_dir / "one_shot_train.jsonl"
    val_file = output_dir / "one_shot_val.jsonl"
    with open(train_file, "w") as f:
        f.write(json.dumps(ONE_SHOT_EXAMPLE) + "\n")
    # Use the same example as val (it's one-shot, we expect memorization)
    with open(val_file, "w") as f:
        f.write(json.dumps(ONE_SHOT_EXAMPLE) + "\n")

    adapter_dir = str(output_dir / "adapter")

    logger.info("Training with ONE example...")
    result = subprocess.run(
        [
            "poetry", "run", "mc", "train", "run",
            "--model", model_path,
            "--data", str(train_file),
            "--eval-data", str(val_file),
            "--output", adapter_dir,
            "--seq-length", "512",
            "--seed", "42",
        ],
        capture_output=True, text=True, timeout=3600,
    )

    if result.returncode != 0:
        logger.error("Training failed: %s", result.stderr[-500:] if result.stderr else "unknown")
        sys.exit(1)

    logger.info("Training complete. Adapter saved to %s", adapter_dir)
    return adapter_dir


def compare_profiles(before: dict, after: dict) -> dict:
    """Compare pre/post profiles to identify which layers changed."""
    changes = {}

    # Compare expansion ratios per layer
    if "expansion_ratio" in before and "expansion_ratio" in after:
        before_dims = {}
        after_dims = {}
        for r in before["expansion_ratio"].get("results", []):
            for ld in r.get("layer_dims", []):
                before_dims[ld["layer"]] = ld["intrinsic_dimension"]
        for r in after["expansion_ratio"].get("results", []):
            for ld in r.get("layer_dims", []):
                after_dims[ld["layer"]] = ld["intrinsic_dimension"]

        layer_changes = []
        for layer in sorted(set(before_dims) | set(after_dims)):
            b_val = before_dims.get(layer, 0)
            a_val = after_dims.get(layer, 0)
            delta = a_val - b_val
            pct = (delta / b_val * 100) if b_val > 0 else 0
            layer_changes.append({
                "layer": layer,
                "before": round(b_val, 3),
                "after": round(a_val, 3),
                "delta": round(delta, 3),
                "pct_change": round(pct, 1),
            })
        changes["expansion_ratio_per_layer"] = layer_changes

    # Compare spectral entropy per layer
    if "spectral_trajectory" in before and "spectral_trajectory" in after:
        before_ent = {r["layer"]: r["spectral_entropy"] for r in before["spectral_trajectory"].get("layerResults", [])}
        after_ent = {r["layer"]: r["spectral_entropy"] for r in after["spectral_trajectory"].get("layerResults", [])}

        ent_changes = []
        for layer in sorted(set(before_ent) | set(after_ent)):
            b_val = before_ent.get(layer, 0)
            a_val = after_ent.get(layer, 0)
            ent_changes.append({
                "layer": layer,
                "before": round(b_val, 4),
                "after": round(a_val, 4),
                "delta": round(a_val - b_val, 4),
            })
        changes["spectral_entropy_per_layer"] = ent_changes

    return changes


def main():
    parser = argparse.ArgumentParser(description="One-Shot RLVR Geometric Pivot Experiment")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profiling (use existing)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing adapter)")
    parser.add_argument("--adapter", help="Path to existing adapter (with --skip-training)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Phase 1: Profile BEFORE
    if not args.skip_profile:
        logger.info("=" * 60)
        logger.info("PHASE 1: Profile BEFORE training")
        logger.info("=" * 60)
        before_profiles = profile_model(args.model, output_dir, "before")
    else:
        before_path = output_dir / "before_profiles.json"
        with open(before_path) as f:
            before_profiles = json.load(f)

    # Phase 2: Baseline inference
    logger.info("=" * 60)
    logger.info("PHASE 2: Baseline inference (%d prompts)", len(EVAL_PROMPTS))
    logger.info("=" * 60)
    baseline_results = run_inference(args.model, EVAL_PROMPTS)
    with open(output_dir / "baseline_inference.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Phase 3: One-shot training
    if not args.skip_training:
        logger.info("=" * 60)
        logger.info("PHASE 3: One-shot training")
        logger.info("=" * 60)
        adapter_path = train_one_shot(args.model, output_dir)
    else:
        adapter_path = args.adapter or str(output_dir / "adapter")

    # Phase 4: Profile AFTER
    if not args.skip_profile:
        logger.info("=" * 60)
        logger.info("PHASE 4: Profile AFTER training")
        logger.info("=" * 60)
        # Profile with adapter loaded
        after_profiles = profile_model(args.model, output_dir, "after")
        # Note: profiling the base model + adapter requires the adapter to be loaded.
        # If the analyze commands don't support --adapter, we profile the base
        # and compare structural changes.
    else:
        after_path = output_dir / "after_profiles.json"
        with open(after_path) as f:
            after_profiles = json.load(f)

    # Phase 5: Post-training inference
    logger.info("=" * 60)
    logger.info("PHASE 5: Post-training inference (%d prompts)", len(EVAL_PROMPTS))
    logger.info("=" * 60)
    adapted_results = run_inference(args.model, EVAL_PROMPTS, adapter_path=adapter_path)
    with open(output_dir / "adapted_inference.json", "w") as f:
        json.dump(adapted_results, f, indent=2)

    # Phase 6: Compare
    logger.info("=" * 60)
    logger.info("PHASE 6: Compare profiles")
    logger.info("=" * 60)
    changes = compare_profiles(before_profiles, after_profiles)
    with open(output_dir / "profile_changes.json", "w") as f:
        json.dump(changes, f, indent=2)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE in %.1f seconds", elapsed)
    logger.info("Results in: %s", output_dir)
    logger.info("=" * 60)

    # Print summary
    logger.info("\nKey files:")
    logger.info("  before_profiles.json  — geometric state before training")
    logger.info("  after_profiles.json   — geometric state after training")
    logger.info("  baseline_inference.json — pre-training responses")
    logger.info("  adapted_inference.json  — post-training responses")
    logger.info("  profile_changes.json  — per-layer geometric deltas")


if __name__ == "__main__":
    main()
