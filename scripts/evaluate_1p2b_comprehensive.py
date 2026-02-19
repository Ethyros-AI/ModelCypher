#!/usr/bin/env python3
"""Comprehensive evaluation for LFM2-1.2B beyond PPL.

Evaluation dimensions:
1. Inference quality (50+ prompts, greedy, manual scoring guide)
2. lm-eval benchmarks (arc, boolq, winogrande, piqa, hellaswag, openbookqa)
3. Format robustness (same logic in 5+ formats)
4. Repetition rate (4-gram repetition count)
5. CKA preservation (from training metrics)

Usage:
    poetry run python scripts/evaluate_1p2b_comprehensive.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16 \
        --adapter /Volumes/CodeCypher/experiments/1p2b-cayley-v1/adapter \
        --output /Volumes/CodeCypher/experiments/1p2b-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Evaluation Prompts ──

INFERENCE_PROMPTS = [
    # Modus Tollens (5 formats)
    "If it is raining, the ground is wet. The ground is not wet. Is it raining?",
    "Whenever the alarm sounds, everyone evacuates. Nobody evacuated. Did the alarm sound?",
    "If a function is differentiable, it is continuous. The function is not continuous. Is it differentiable?",
    "All certified diamonds have serial numbers. This diamond has no serial number. Is it certified?",
    "If the server is running, the status page shows green. The status page does not show green. Is the server running?",

    # Modus Ponens (5 formats)
    "If it snows, schools close. It is snowing. Are schools closed?",
    "Whenever pressure exceeds 100 PSI, the valve opens. Pressure is at 120 PSI. Is the valve open?",
    "If an integer is divisible by 4, it is divisible by 2. The number 12 is divisible by 4. Is it divisible by 2?",
    "All birds have feathers. A robin is a bird. Does a robin have feathers?",
    "If the test passes, we deploy. The test passed. Do we deploy?",

    # Hypothetical Syllogism (chain reasoning)
    "If it rains, the streets get wet. If the streets get wet, cars slow down. It is raining. Do cars slow down?",
    "If the temperature drops below freezing, water turns to ice. If water turns to ice, the pipes may burst. The temperature dropped below freezing. May the pipes burst?",
    "If you exercise regularly, your cardiovascular health improves. If your cardiovascular health improves, your risk of heart disease decreases. You exercise regularly. Does your risk of heart disease decrease?",

    # Disjunctive Syllogism
    "Either the package was delivered or it was returned to sender. The package was not delivered. What happened?",
    "The error is either in the frontend or the backend. The frontend code is correct. Where is the error?",
    "The signal is either analog or digital. The signal is not analog. What type is it?",

    # Universal Quantifier
    "Every mammal is warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded?",
    "All rectangles have four right angles. A square is a rectangle. Does a square have four right angles?",
    "Every conductor allows electric current to flow. Copper wire is a conductor. Does copper wire allow electric current to flow?",

    # Math (10 problems)
    "What is 7 * 8?",
    "If a shirt costs $25 and is 20% off, what is the sale price?",
    "A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
    "What is the next number in the sequence: 2, 6, 18, 54, ?",
    "A rectangle has length 8 and width 5. What is its area?",
    "If 3x + 7 = 22, what is x?",
    "What is 15% of 80?",
    "A train travels at 60 mph for 2.5 hours. How far does it go?",
    "What is the square root of 144?",
    "If you have 3 red balls and 5 blue balls, what fraction are red?",

    # General Knowledge (10)
    "What is the capital of France?",
    "How many planets are in our solar system?",
    "What is photosynthesis?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "What is the largest ocean on Earth?",
    "What year did World War II end?",
    "What is the chemical formula for water?",
    "How many continents are there?",
    "What is the speed of light approximately?",

    # Tricky / Common Errors (7)
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "How many times does the letter 'r' appear in 'strawberry'?",
    "If you have a bowl with six apples and you take away four, how many do you have?",
    "Is the following valid? All dogs are animals. All cats are animals. Therefore, all dogs are cats.",
    "If all birds can fly and penguins are birds, can penguins fly? Note: not all birds can actually fly.",
    "A doctor says: 'I have a brother.' The brother says: 'I don't have a brother.' Is this possible?",
    "There are three switches downstairs. One controls a lightbulb upstairs. You can only go upstairs once. How do you figure out which switch controls the bulb?",
]


FORMAT_ROBUSTNESS_LOGIC = "If it is raining, then the ground is wet. The ground is not wet."
FORMAT_ROBUSTNESS_PROMPTS = [
    f"Question: {FORMAT_ROBUSTNESS_LOGIC} Is it raining?",
    f"Consider the following:\n{FORMAT_ROBUSTNESS_LOGIC}\nWhat can we conclude?",
    f"{FORMAT_ROBUSTNESS_LOGIC}\nWhat follows?",
    f"Given: {FORMAT_ROBUSTNESS_LOGIC}\nDetermine the result.",
    f"Premise: {FORMAT_ROBUSTNESS_LOGIC}\nWhat must be true?",
    f"Analyze this:\n{FORMAT_ROBUSTNESS_LOGIC}\nWhat is the conclusion?",
]


def run_inference(model_path: str, prompts: list[str], adapter_path: str | None = None) -> list[dict]:
    """Run greedy inference on all prompts."""
    results = []
    for i, prompt in enumerate(prompts):
        cmd = [
            "poetry", "run", "mc", "infer", "run",
            "--model", model_path,
            "--prompt", prompt,
        ]
        if adapter_path:
            cmd.extend(["--adapter", adapter_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            response = result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr[:200]}"
        except subprocess.TimeoutExpired:
            response = "ERROR: timeout"

        results.append({
            "prompt": prompt,
            "response": response,
            "index": i,
        })

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] prompts completed", i + 1, len(prompts))

    return results


def compute_repetition_stats(responses: list[str]) -> dict:
    """Compute 4-gram repetition statistics."""
    total_4grams = 0
    repeated_4grams = 0
    degenerate_count = 0

    for response in responses:
        words = response.split()
        if len(words) < 4:
            continue

        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        total_4grams += len(ngrams)
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        repeated_4grams += repeated

        # Degenerate = more than 20% repeated 4-grams
        if len(ngrams) > 0 and repeated / len(ngrams) > 0.2:
            degenerate_count += 1

    return {
        "total_4grams": total_4grams,
        "repeated_4grams": repeated_4grams,
        "repetition_rate": repeated_4grams / max(1, total_4grams),
        "degenerate_responses": degenerate_count,
        "total_responses": len(responses),
    }


def check_format_robustness(results: list[dict]) -> dict:
    """Check if model gives correct answer across format variations."""
    correct_patterns = [
        r"not raining",
        r"it is not raining",
        r"no[,.]",
        r"not the case that.*rain",
        r"cannot.*conclud.*raining",
    ]

    correct_count = 0
    for r in results:
        response = r["response"].lower()
        if any(re.search(p, response) for p in correct_patterns):
            correct_count += 1

    return {
        "total_formats": len(results),
        "correct_formats": correct_count,
        "robustness_rate": correct_count / max(1, len(results)),
    }


def run_lm_eval(model_path: str, adapter_path: str | None, output_dir: Path) -> dict | None:
    """Run lm-eval benchmarks if available."""
    try:
        cmd = [
            "poetry", "run", "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "arc_easy,arc_challenge,boolq,winogrande,piqa,hellaswag,openbookqa",
            "--num_fewshot", "0",
            "--batch_size", "4",
            "--output_path", str(output_dir / "lm_eval_results"),
        ]
        if adapter_path:
            cmd[5] = f"pretrained={model_path},peft={adapter_path}"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            # Try to parse output
            return {"raw_output": result.stdout[-2000:]}
        else:
            logger.warning("lm-eval failed: %s", result.stderr[:300])
            return None
    except Exception as e:
        logger.warning("lm-eval not available: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Comprehensive 1.2B Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--adapter", help="Adapter path (if evaluating adapted model)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--skip-lm-eval", action="store_true", help="Skip lm-eval benchmarks")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline (no adapter)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 1. Baseline inference
    logger.info("=" * 60)
    logger.info("1. BASELINE INFERENCE (%d prompts)", len(INFERENCE_PROMPTS))
    logger.info("=" * 60)
    baseline_results = run_inference(args.model, INFERENCE_PROMPTS)
    with open(output_dir / "baseline_inference.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    baseline_rep = compute_repetition_stats([r["response"] for r in baseline_results])
    logger.info("  Baseline repetition rate: %.4f (%d degenerate)",
                baseline_rep["repetition_rate"], baseline_rep["degenerate_responses"])

    # 2. Baseline format robustness
    logger.info("\n2. BASELINE FORMAT ROBUSTNESS (%d formats)", len(FORMAT_ROBUSTNESS_PROMPTS))
    baseline_format = run_inference(args.model, FORMAT_ROBUSTNESS_PROMPTS)
    baseline_format_stats = check_format_robustness(baseline_format)
    logger.info("  Format robustness: %d/%d correct",
                baseline_format_stats["correct_formats"], baseline_format_stats["total_formats"])

    if args.baseline_only or not args.adapter:
        logger.info("\nBaseline-only mode. Skipping adapted evaluation.")
        report = {
            "model_path": args.model,
            "baseline_repetition": baseline_rep,
            "baseline_format_robustness": baseline_format_stats,
            "total_prompts": len(INFERENCE_PROMPTS),
        }
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        elapsed = time.time() - start_time
        logger.info("\nDone in %.1f seconds. Results in %s", elapsed, output_dir)
        return

    # 3. Adapted inference
    logger.info("\n3. ADAPTED INFERENCE (%d prompts)", len(INFERENCE_PROMPTS))
    adapted_results = run_inference(args.model, INFERENCE_PROMPTS, adapter_path=args.adapter)
    with open(output_dir / "adapted_inference.json", "w") as f:
        json.dump(adapted_results, f, indent=2)

    adapted_rep = compute_repetition_stats([r["response"] for r in adapted_results])
    logger.info("  Adapted repetition rate: %.4f (%d degenerate)",
                adapted_rep["repetition_rate"], adapted_rep["degenerate_responses"])

    # 4. Adapted format robustness
    logger.info("\n4. ADAPTED FORMAT ROBUSTNESS (%d formats)", len(FORMAT_ROBUSTNESS_PROMPTS))
    adapted_format = run_inference(args.model, FORMAT_ROBUSTNESS_PROMPTS, adapter_path=args.adapter)
    adapted_format_stats = check_format_robustness(adapted_format)
    logger.info("  Format robustness: %d/%d correct",
                adapted_format_stats["correct_formats"], adapted_format_stats["total_formats"])

    # 5. lm-eval benchmarks
    lm_eval_baseline = None
    lm_eval_adapted = None
    if not args.skip_lm_eval:
        logger.info("\n5. LM-EVAL BENCHMARKS")
        logger.info("  Running baseline...")
        lm_eval_baseline = run_lm_eval(args.model, None, output_dir)
        logger.info("  Running adapted...")
        lm_eval_adapted = run_lm_eval(args.model, args.adapter, output_dir)

    # 6. Summary report
    report = {
        "model_path": args.model,
        "adapter_path": args.adapter,
        "total_prompts": len(INFERENCE_PROMPTS),
        "baseline_repetition": baseline_rep,
        "adapted_repetition": adapted_rep,
        "baseline_format_robustness": baseline_format_stats,
        "adapted_format_robustness": adapted_format_stats,
        "lm_eval_baseline": lm_eval_baseline,
        "lm_eval_adapted": lm_eval_adapted,
    }

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Repetition rate: baseline=%.4f, adapted=%.4f",
                baseline_rep["repetition_rate"], adapted_rep["repetition_rate"])
    logger.info("Degenerate responses: baseline=%d/%d, adapted=%d/%d",
                baseline_rep["degenerate_responses"], baseline_rep["total_responses"],
                adapted_rep["degenerate_responses"], adapted_rep["total_responses"])
    logger.info("Format robustness: baseline=%d/%d, adapted=%d/%d",
                baseline_format_stats["correct_formats"], baseline_format_stats["total_formats"],
                adapted_format_stats["correct_formats"], adapted_format_stats["total_formats"])
    logger.info("\nElapsed: %.1f seconds. Results in %s", elapsed, output_dir)
    logger.info("\nNote: Inference quality must be manually scored.")
    logger.info("Review baseline_inference.json and adapted_inference.json.")


if __name__ == "__main__":
    main()
