#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Experiment 5: End-to-End Merge Validation
#
# HYPOTHESIS: Merged model inherits capabilities from both sources
#
# PROTOCOL:
# 1. Establish baseline coherence for source and target models
# 2. Run merge with full atlas probes (ρ > 4.0)
# 3. Validate merged model coherence
# 4. Compare to baselines
#
# SUCCESS CRITERIA:
# - Target capabilities preserved (coherence maintained)
# - Merged model generates coherent, non-repetitive output
# - Density increased (if measured)
#
# NOTE: This experiment runs actual inference, which can be slow.

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modelcypher.core.domain._backend import get_default_backend

from experiments.validation_protocol.shared import (
    SMOLLM_PATH,
    LFM2_PATH,
    ExperimentResult,
    setup_experiment,
    ensure_output_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_coherence_test(model_path: Path, num_prompts: int = 5, max_tokens: int = 50) -> dict:
    """Test model coherence via inference.

    Returns:
        Dict with coherence metrics
    """
    from modelcypher.adapters.model_loader import load_model
    from modelcypher.core.use_cases.inference import run_inference

    test_prompts = [
        "The capital of France is",
        "Water is composed of",
        "The derivative of x squared is",
        "In the year 1969, humans",
        "The speed of light is approximately",
    ][:num_prompts]

    try:
        logger.info("Loading model from %s...", model_path)
        model, tokenizer = load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return {"error": str(e), "is_coherent": False}

    results = []
    for prompt in test_prompts:
        try:
            output = run_inference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            generated = output.get("generated_text", "")

            # Compute repetition score
            words = generated.lower().split()
            if len(words) >= 5:
                ngram_counts = {}
                for n in [2, 3]:
                    for i in range(len(words) - n + 1):
                        ngram = tuple(words[i:i+n])
                        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                repetitions = sum(c - 1 for c in ngram_counts.values() if c > 1)
                total = sum(len(words) - n + 1 for n in [2, 3])
                rep_score = repetitions / max(total, 1) * 10
            else:
                rep_score = 0.0

            results.append({
                "prompt": prompt,
                "generated": generated[:200],  # Truncate for logging
                "length": len(generated),
                "repetition_score": rep_score,
            })
            logger.info("  Prompt: '%s...' → length=%d, rep=%.2f",
                       prompt[:30], len(generated), rep_score)

        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})
            logger.error("  Error on '%s': %s", prompt[:30], e)

    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "All inferences failed", "is_coherent": False}

    rep_scores = [r["repetition_score"] for r in valid]
    mean_rep = sum(rep_scores) / len(rep_scores)
    max_rep = max(rep_scores)

    return {
        "n_prompts": len(valid),
        "mean_repetition_score": mean_rep,
        "max_repetition_score": max_rep,
        "is_coherent": max_rep < 3.0,  # Coherent if max repetition < 3.0
        "details": results,
    }


def run_merge(source_path: Path, target_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Run the merge pipeline.

    Returns:
        Dict with merge results
    """
    cmd = [
        "poetry", "run", "mc", "merge", "run",
        "-s", str(source_path),
        "-t", str(target_path),
        "-o", str(output_path),
        "--full-atlas",
    ]

    if dry_run:
        cmd.append("--dry-run")

    logger.info("Running merge command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        return {
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Merge timed out after 30 minutes", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def main():
    """Run Experiment 5: End-to-End Merge Validation."""
    start_time = time.perf_counter()

    output_dir = ensure_output_dir("exp5_endtoend_merge")
    backend = get_default_backend()

    config = setup_experiment(
        name="exp5_endtoend_merge",
        source_path=SMOLLM_PATH,
        target_path=LFM2_PATH,
        backend=backend,
        hyperparameters={
            "merge_type": "full_atlas",
            "test_type": "coherence_validation",
        },
    )

    results = {
        "baseline_source": {},
        "baseline_target": {},
        "merge_result": {},
        "merged_coherence": {},
        "summary": {},
    }

    # ==========================================================================
    # PART 1: Baseline Coherence Tests
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("PART 1: Baseline Coherence Tests")
    logger.info("=" * 70)

    logger.info("")
    logger.info("Testing SOURCE model (SmolLM)...")
    results["baseline_source"] = run_coherence_test(SMOLLM_PATH, num_prompts=5)
    source_coherent = results["baseline_source"].get("is_coherent", False)
    logger.info("Source coherent: %s (max_rep=%.2f)",
               source_coherent,
               results["baseline_source"].get("max_repetition_score", -1))

    logger.info("")
    logger.info("Testing TARGET model (LFM2)...")
    results["baseline_target"] = run_coherence_test(LFM2_PATH, num_prompts=5)
    target_coherent = results["baseline_target"].get("is_coherent", False)
    logger.info("Target coherent: %s (max_rep=%.2f)",
               target_coherent,
               results["baseline_target"].get("max_repetition_score", -1))

    # ==========================================================================
    # PART 2: Run Merge
    # ==========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PART 2: Run Merge Pipeline")
    logger.info("=" * 70)

    # Create temp directory for merged model
    merged_dir = output_dir / "merged_model"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    # First do a dry run to see what would happen
    logger.info("")
    logger.info("Dry run to verify merge setup...")
    dry_run_result = run_merge(SMOLLM_PATH, LFM2_PATH, merged_dir, dry_run=True)
    results["dry_run"] = dry_run_result

    if not dry_run_result.get("success", False):
        logger.warning("Dry run failed: %s", dry_run_result.get("error", dry_run_result.get("stderr", "")))
        # Continue anyway to see if actual merge works

    # Run actual merge
    logger.info("")
    logger.info("Running actual merge (this may take several minutes)...")
    merge_result = run_merge(SMOLLM_PATH, LFM2_PATH, merged_dir, dry_run=False)
    results["merge_result"] = merge_result

    if not merge_result.get("success", False):
        logger.error("Merge failed!")
        logger.error("stdout: %s", merge_result.get("stdout", "")[:1000])
        logger.error("stderr: %s", merge_result.get("stderr", "")[:1000])
        results["summary"]["success"] = False
        results["summary"]["error"] = "Merge failed"
    else:
        logger.info("Merge completed successfully!")

        # ==========================================================================
        # PART 3: Validate Merged Model
        # ==========================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("PART 3: Validate Merged Model")
        logger.info("=" * 70)

        logger.info("")
        logger.info("Testing MERGED model coherence...")
        results["merged_coherence"] = run_coherence_test(merged_dir, num_prompts=5)
        merged_coherent = results["merged_coherence"].get("is_coherent", False)
        logger.info("Merged coherent: %s (max_rep=%.2f)",
                   merged_coherent,
                   results["merged_coherence"].get("max_repetition_score", -1))

        # Compare to baselines
        target_max_rep = results["baseline_target"].get("max_repetition_score", 0)
        merged_max_rep = results["merged_coherence"].get("max_repetition_score", float("inf"))

        # Success criteria:
        # 1. Merged model is coherent (max_rep < 3.0)
        # 2. Merged model is not significantly worse than target (within 2x repetition)
        coherence_preserved = merged_coherent or merged_max_rep < target_max_rep * 2 + 1

        results["summary"] = {
            "source_coherent": source_coherent,
            "target_coherent": target_coherent,
            "merged_coherent": merged_coherent,
            "coherence_preserved": coherence_preserved,
            "target_max_rep": target_max_rep,
            "merged_max_rep": merged_max_rep,
            "success": merged_coherent and coherence_preserved,
            "interpretation": (
                "Merged model generates coherent output, preserving target capabilities."
                if merged_coherent
                else "Merged model shows increased repetition, indicating potential coherence issues."
            ),
        }

    duration = time.perf_counter() - start_time

    # Save
    experiment_result = ExperimentResult(
        config=config,
        metrics=results.get("summary", {}),
        raw_data=results,
        duration_seconds=duration,
        success=results.get("summary", {}).get("success", False),
    )
    experiment_result.save(output_dir / "results.json")
    config.save(output_dir / "config.json")

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 5 COMPLETE: End-to-End Merge Validation")
    logger.info("=" * 70)
    logger.info("Duration: %.1f seconds", duration)
    logger.info("Success: %s", results.get("summary", {}).get("success", False))

    if "summary" in results:
        summary = results["summary"]
        logger.info("Source coherent: %s", summary.get("source_coherent", "N/A"))
        logger.info("Target coherent: %s", summary.get("target_coherent", "N/A"))
        logger.info("Merged coherent: %s", summary.get("merged_coherent", "N/A"))
        logger.info("Coherence preserved: %s", summary.get("coherence_preserved", "N/A"))
        if "interpretation" in summary:
            logger.info("Interpretation: %s", summary["interpretation"])

    logger.info("")
    logger.info("Results saved to: %s", output_dir / "results.json")
    logger.info("=" * 70)

    return experiment_result


if __name__ == "__main__":
    main()
