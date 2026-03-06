#!/usr/bin/env python3
"""Experiment 2: Scale Bound A/B Test.

Tests the hypothesis that QLoRA degradation is caused by spectral scale
violation, not quantization error.

Two arms on the same quantized base model, same data, same seed:
  Arm A (standard): scale_bound = 1.0 (effective delta up to 2.0 per SV direction,
                     mimicking standard LoRA alpha/rank ≈ 2.0)
  Arm B (geometric): scale_bound derived from sigma_k / 2 (spectral safety)

Measures: baseline perplexity, post-training perplexity, CKA vs base,
          spectral bounds, training stability.

Usage:
    poetry run python scripts/quantization_scale_ab_test.py

    # Custom model
    poetry run python scripts/quantization_scale_ab_test.py \\
        --quantized-model /path/to/8bit \\
        --fp-model /path/to/bf16

    # Custom scale for standard arm
    poetry run python scripts/quantization_scale_ab_test.py --standard-scale 1.0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("scale_ab_test")

# Default paths
DEFAULT_QUANTIZED = (
    "results/feasibility_map/20260225T160732Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-8bit-g64-affine"
)
DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_TRAIN = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"


def _evaluate_perplexity(
    model_path: str,
    dataset_path: str,
    adapter_path: str | None = None,
) -> dict[str, float]:
    """Compute perplexity of a model on a dataset.

    Lightweight standalone implementation that avoids the EvaluationService
    dependency graph (store, model_loader ports).
    """
    import json as json_mod

    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()

    logger.info(
        "Evaluating perplexity: model=%s adapter=%s dataset=%s",
        Path(model_path).name,
        Path(adapter_path).name if adapter_path else None,
        Path(dataset_path).name,
    )

    # Load model (with optional adapter)
    model, tokenizer = backend.load_model(str(model_path), adapter_path)

    # Load dataset
    samples = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json_mod.loads(line)
                    text = data.get("text", "")
                    if text:
                        samples.append(text)
                except json_mod.JSONDecodeError:
                    continue

    if not samples:
        return {"average_loss": 0.0, "perplexity": 0.0, "n_samples": 0}

    total_loss = 0.0
    total_tokens = 0

    for text in samples:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        tokens_arr = backend.array(tokens)
        input_arr = backend.reshape(tokens_arr, (1, -1))
        logits = model(input_arr)
        logits = logits[0, :-1, :]
        targets = tokens_arr[1:]

        log_scores = backend.log_softmax(logits, axis=-1)
        targets_expanded = backend.reshape(targets, (-1, 1))
        target_log_scores = backend.take_along_axis(log_scores, targets_expanded, axis=-1)
        target_log_scores = backend.squeeze(target_log_scores, axis=-1)
        backend.eval(target_log_scores)

        mean_arr = backend.mean(target_log_scores)
        backend.eval(mean_arr)
        sample_loss = -float(backend.to_scalar(mean_arr))
        n_targets = int(targets.shape[0])
        total_loss += sample_loss * n_targets
        total_tokens += n_targets

    average_loss = total_loss / max(total_tokens, 1)
    perplexity_arr = backend.exp(backend.array([average_loss]))
    backend.eval(perplexity_arr)
    perplexity = float(backend.to_scalar(perplexity_arr))

    logger.info(
        "Perplexity: %.4f (loss=%.4f, %d samples, %d tokens)",
        perplexity, average_loss, len(samples), total_tokens,
    )

    # Release model memory
    del model, tokenizer
    gc.collect()

    return {
        "average_loss": average_loss,
        "perplexity": perplexity,
        "n_samples": len(samples),
        "n_tokens": total_tokens,
    }


def _run_training_arm(
    arm_name: str,
    quantized_model: str,
    train_dataset: str,
    eval_dataset: str,
    output_dir: Path,
    scale_bound_override: float | None,
    seed: int,
    max_iters_cap: int | None,
) -> dict[str, Any]:
    """Run a single training arm and return results."""
    from modelcypher.cli.composition import get_dataset_training_service

    logger.info("=" * 60)
    logger.info("ARM: %s (scale_bound_override=%s)", arm_name, scale_bound_override)
    logger.info("=" * 60)

    service = get_dataset_training_service()

    adapter_output = output_dir / f"adapter_{arm_name}"

    start = time.monotonic()
    result = service.train_from_dataset(
        model_path=quantized_model,
        dataset_path=train_dataset,
        eval_dataset_path=eval_dataset,
        output_path=str(adapter_output),
        seed=seed,
        scale_bound_override=scale_bound_override,
        max_iters_cap=max_iters_cap,
    )
    elapsed = time.monotonic() - start

    arm_result = {
        "arm": arm_name,
        "scale_bound_override": scale_bound_override,
        "train_iters": result.train_iters,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "stop_reason": result.stop_reason,
        "baseline_loss": result.baseline_loss,
        "baseline_perplexity": result.baseline_perplexity,
        "post_loss": result.post_loss,
        "post_perplexity": result.post_perplexity,
        "n_lora_layers": result.n_lora_layers,
        "n_trainable_params": result.n_trainable_params,
        "adapter_path": result.adapter_path,
        "spectral_bounds_ok": result.spectral_bounds_ok,
        "max_spectral_ratio": result.max_spectral_ratio,
        "min_cka": result.min_cka,
        "mean_cka": result.mean_cka,
        "training_time_seconds": elapsed,
    }

    logger.info(
        "ARM %s complete: iters=%d, baseline_ppl=%.2f, post_ppl=%.2f, "
        "mean_cka=%s, spectral_ok=%s, stop=%s",
        arm_name,
        result.train_iters,
        result.baseline_perplexity,
        result.post_perplexity,
        f"{result.mean_cka:.4f}" if result.mean_cka is not None else "N/A",
        result.spectral_bounds_ok,
        result.stop_reason,
    )

    # Force cleanup between arms
    gc.collect()

    return arm_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Scale Bound A/B Test",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED,
        help="Path to 8-bit quantized model",
    )
    parser.add_argument(
        "--fp-model",
        default=DEFAULT_FP,
        help="Path to full-precision (bf16) model for baseline perplexity",
    )
    parser.add_argument(
        "--train-dataset",
        default=DEFAULT_TRAIN,
        help="Path to training dataset (JSONL)",
    )
    parser.add_argument(
        "--eval-dataset",
        default=DEFAULT_EVAL,
        help="Path to evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/quantization_scale_ab_test",
        help="Base output directory",
    )
    parser.add_argument(
        "--standard-scale",
        type=float,
        default=1.0,
        help="Scale bound for standard arm (default: 1.0, giving effective delta up to 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed (same for both arms)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Cap training iterations (default: derived from data)",
    )
    parser.add_argument(
        "--skip-fp-baseline",
        action="store_true",
        help="Skip bf16 baseline perplexity (saves time if already known)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Initialize backend
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scale Bound A/B Test — run_id=%s", run_id)
    logger.info("Quantized model: %s", args.quantized_model)
    logger.info("FP model: %s", args.fp_model)
    logger.info("Standard scale: %.4f", args.standard_scale)
    logger.info("Output: %s", output_dir)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "quantization_scale_ab_test",
        "config": {
            "quantized_model": args.quantized_model,
            "fp_model": args.fp_model,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "standard_scale_bound": args.standard_scale,
            "seed": args.seed,
            "max_iters_cap": args.max_iters,
        },
    }

    # Phase 1: Baseline perplexity measurements
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: BASELINE PERPLEXITY")
    logger.info("=" * 60)

    if not args.skip_fp_baseline:
        fp_ppl = _evaluate_perplexity(args.fp_model, args.eval_dataset)
        results["fp_baseline"] = fp_ppl
        logger.info("FP baseline perplexity: %.4f", fp_ppl["perplexity"])
    else:
        results["fp_baseline"] = {"skipped": True}
        logger.info("FP baseline: SKIPPED")

    # Phase 2: Train Arm A (standard scale)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: ARM A — STANDARD SCALE (bound=%.4f)", args.standard_scale)
    logger.info("=" * 60)

    arm_a = _run_training_arm(
        arm_name="standard_scale",
        quantized_model=args.quantized_model,
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        output_dir=output_dir,
        scale_bound_override=args.standard_scale,
        seed=args.seed,
        max_iters_cap=args.max_iters,
    )
    results["arm_a_standard"] = arm_a

    # Phase 3: Train Arm B (geometric scale)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: ARM B — GEOMETRIC SCALE (derived)")
    logger.info("=" * 60)

    arm_b = _run_training_arm(
        arm_name="geometric_scale",
        quantized_model=args.quantized_model,
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        output_dir=output_dir,
        scale_bound_override=None,  # Use default geometric derivation
        seed=args.seed,
        max_iters_cap=args.max_iters,
    )
    results["arm_b_geometric"] = arm_b

    # Phase 4: Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)

    comparison = {
        "baseline_perplexity_quantized": arm_a["baseline_perplexity"],
        "post_perplexity_standard": arm_a["post_perplexity"],
        "post_perplexity_geometric": arm_b["post_perplexity"],
        "ppl_delta_standard": arm_a["post_perplexity"] - arm_a["baseline_perplexity"],
        "ppl_delta_geometric": arm_b["post_perplexity"] - arm_b["baseline_perplexity"],
        "spectral_ok_standard": arm_a["spectral_bounds_ok"],
        "spectral_ok_geometric": arm_b["spectral_bounds_ok"],
        "cka_standard": arm_a["mean_cka"],
        "cka_geometric": arm_b["mean_cka"],
        "iters_standard": arm_a["train_iters"],
        "iters_geometric": arm_b["train_iters"],
        "stop_standard": arm_a["stop_reason"],
        "stop_geometric": arm_b["stop_reason"],
    }

    if not args.skip_fp_baseline:
        comparison["fp_baseline_perplexity"] = results["fp_baseline"]["perplexity"]

    results["comparison"] = comparison

    # Measured deltas — no heuristic thresholds.
    # The data determines the verdict, not a magic number.
    std_ppl = arm_a["post_perplexity"]
    geo_ppl = arm_b["post_perplexity"]
    std_spectral = arm_a["spectral_bounds_ok"]
    geo_spectral = arm_b["spectral_bounds_ok"]
    ppl_delta = geo_ppl - std_ppl  # negative = geometric wins
    ppl_rel = ppl_delta / max(std_ppl, 1e-8)

    comparison["ppl_delta_geo_minus_std"] = ppl_delta
    comparison["ppl_relative_delta"] = ppl_rel
    comparison["geometric_ppl_lower"] = geo_ppl < std_ppl
    comparison["geometric_spectral_ok"] = geo_spectral
    comparison["standard_spectral_ok"] = std_spectral

    verdict = (
        f"MEASURED: ppl(geometric)={geo_ppl:.4f}, ppl(standard)={std_ppl:.4f}, "
        f"delta={ppl_delta:+.4f} ({ppl_rel:+.2%}). "
        f"Spectral bounds: geometric={'OK' if geo_spectral else 'VIOLATED'}, "
        f"standard={'OK' if std_spectral else 'VIOLATED'}."
    )

    results["verdict"] = verdict

    # Write results
    output_path = output_dir / "scale_ab_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results written to %s", output_path)

    # Pretty print summary
    print("\n" + "=" * 72)
    print("SCALE BOUND A/B TEST — SUMMARY")
    print("=" * 72)
    if not args.skip_fp_baseline:
        print(f"  FP (bf16) baseline perplexity:     {results['fp_baseline']['perplexity']:.4f}")
    print(f"  Quantized baseline perplexity:     {comparison['baseline_perplexity_quantized']:.4f}")
    print(f"  Standard scale post-training:      {comparison['post_perplexity_standard']:.4f} "
          f"(delta: {comparison['ppl_delta_standard']:+.4f})")
    print(f"  Geometric scale post-training:     {comparison['post_perplexity_geometric']:.4f} "
          f"(delta: {comparison['ppl_delta_geometric']:+.4f})")
    print()
    print(f"  Standard spectral bounds OK:       {comparison['spectral_ok_standard']}")
    print(f"  Geometric spectral bounds OK:      {comparison['spectral_ok_geometric']}")
    print(f"  Standard CKA:                      {comparison['cka_standard']}")
    print(f"  Geometric CKA:                     {comparison['cka_geometric']}")
    print(f"  Standard iters (stop):             {comparison['iters_standard']} ({comparison['stop_standard']})")
    print(f"  Geometric iters (stop):            {comparison['iters_geometric']} ({comparison['stop_geometric']})")
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
