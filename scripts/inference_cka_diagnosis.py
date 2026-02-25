#!/usr/bin/env python3
"""Diagnose inference-manifold CKA collapse.

The 350M 5-trial validation revealed eval-manifold CKA is excellent (min 0.950-0.959)
while inference-manifold CKA collapses (min 0.000-0.161) across ALL trials.

Two hypotheses:
  A — Measurement artifact: 10 inference probes in 1024-dim → 10x10 Gram with
      effective rank ≤ 9 after centering. No geometric slack; tiny perturbations
      collapse CKA.
  B — Real manifold split: Inference probes are 3-shot reasoning prompts (500-800
      tokens). Eval probes are short Q&A (~80 tokens). Mean-pooling over long
      sequences washes out layer-specific structure.

This script runs controlled experiments to discriminate A from B.

Usage:
    poetry run python scripts/inference_cka_diagnosis.py \
      --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
      --adapter results/pipeline_validation_cert_350m_5t/350M/phase5_artifacts/trial_000_seed_4231027559 \
      --eval-data data/training/benchmark_val.jsonl \
      --output results/inference_cka_diagnosis/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference_cka_diagnosis")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """Measurements at a single probe-count / manifold / pooling combination."""

    n_probes: int
    manifold: str  # "eval" or "inference"
    pooling: str   # "mean" or "last_token"
    per_layer: dict[str, dict[str, float]]  # layer_idx_str → metrics
    min_cka: float
    mean_cka: float
    n_layers: int


# ---------------------------------------------------------------------------
# Activation collection (matches training service pattern exactly)
# ---------------------------------------------------------------------------

def collect_activations_single_probe(
    backend: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    pooling: str = "mean",
) -> dict[int, list]:
    """Collect per-layer activations, one text at a time.

    Matches the pattern in dataset_training_service._collect_probe_activations:
    collect_hidden_activations() returns [1, seq, hidden] per prompt, then
    pool over the seq dimension.

    Args:
        backend: MLX backend instance.
        model: Loaded model.
        tokenizer: Tokenizer.
        texts: Probe texts.
        pooling: "mean" (mean over seq dim) or "last_token" (last token only).

    Returns:
        dict[layer_idx, list[pooled_vector]] where each vector is [hidden].
    """
    activations: dict[int, list] = {}
    for text in texts:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx, act in acts.items():
            # act: [1, seq, hidden]
            if pooling == "last_token":
                # Take the last token activation
                pooled = act[0, -1, :]  # [hidden]
            else:
                # Mean over seq dimension (default — matches training service)
                pooled = backend.mean(act, axis=1)   # [1, hidden]
                pooled = backend.reshape(pooled, (-1,))  # [hidden]
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)
    return activations


def stack_activations(
    backend: Any,
    layer_acts: list,
) -> Any:
    """Stack list of [hidden] vectors into [n_probes, hidden] matrix."""
    return backend.stack(layer_acts)


# ---------------------------------------------------------------------------
# CKA + Gram spectrum at one sweep point
# ---------------------------------------------------------------------------

def measure_sweep_point(
    backend: Any,
    base_acts: dict[int, list],
    adapted_acts: dict[int, list],
    n_probes: int,
    manifold: str,
    pooling: str,
    rng: random.Random,
) -> SweepPoint:
    """Compute CKA and Gram spectrum for a subsampled probe set."""
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )
    from modelcypher.core.domain.geometry.gram_spectrum import (
        compute_gram_spectrum,
    )

    # Determine subsample indices (same for all layers)
    layer_indices = sorted(base_acts.keys())
    n_available = len(base_acts[layer_indices[0]])
    if n_probes >= n_available:
        indices = list(range(n_available))
        n_probes = n_available
    else:
        indices = sorted(rng.sample(range(n_available), n_probes))

    per_layer: dict[str, dict[str, float]] = {}
    cka_values: list[float] = []

    for layer_idx in layer_indices:
        base_subset = [base_acts[layer_idx][i] for i in indices]
        adapted_subset = [adapted_acts[layer_idx][i] for i in indices]

        base_matrix = stack_activations(backend, base_subset)
        adapted_matrix = stack_activations(backend, adapted_subset)
        backend.eval(base_matrix, adapted_matrix)

        # CKA
        cka = compute_linear_cka_from_activations(
            base_matrix, adapted_matrix, backend=backend,
        )

        # Gram spectrum for both
        gram_base = compute_gram_spectrum(base_matrix, backend=backend)
        gram_adapted = compute_gram_spectrum(adapted_matrix, backend=backend)

        layer_key = str(layer_idx)
        per_layer[layer_key] = {
            "cka": cka,
            "numeric_rank_base": gram_base.numeric_rank,
            "numeric_rank_adapted": gram_adapted.numeric_rank,
            "condition_number_base": gram_base.condition_number,
            "condition_number_adapted": gram_adapted.condition_number,
            "intrinsic_dimension_base": gram_base.intrinsic_dimension,
            "intrinsic_dimension_adapted": gram_adapted.intrinsic_dimension,
            "max_eigenvalue_base": gram_base.max_eigenvalue,
            "max_eigenvalue_adapted": gram_adapted.max_eigenvalue,
            "min_eigenvalue_base": gram_base.min_eigenvalue,
            "min_eigenvalue_adapted": gram_adapted.min_eigenvalue,
        }
        cka_values.append(cka)

    return SweepPoint(
        n_probes=n_probes,
        manifold=manifold,
        pooling=pooling,
        per_layer=per_layer,
        min_cka=min(cka_values) if cka_values else 0.0,
        mean_cka=sum(cka_values) / len(cka_values) if cka_values else 0.0,
        n_layers=len(layer_indices),
    )


# ---------------------------------------------------------------------------
# Inference probe generation
# ---------------------------------------------------------------------------

def generate_inference_probes(n_problems: int, seed: int) -> list[str]:
    """Generate STaR inference probe prompts (few-shot format)."""
    from modelcypher.core.domain.star.problem_generator import StarProblemGenerator
    from modelcypher.core.domain.star.prompting import (
        build_forward_prompt,
        default_few_shot_examples,
    )

    gen = StarProblemGenerator(seed=seed)
    problems = gen.generate(n_problems)
    n_demos = len(default_few_shot_examples())
    return [build_forward_prompt(p, demonstrations=n_demos) for p in problems]


# ---------------------------------------------------------------------------
# Main diagnostic flow
# ---------------------------------------------------------------------------

def run_diagnosis(
    model_path: str,
    adapter_path: str,
    eval_data_path: str,
    output_dir: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full CKA collapse diagnostic.

    Returns the results dict (also written to JSON).
    """
    import mlx.core as mx

    from modelcypher.core.domain._backend import get_default_backend

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    backend = get_default_backend()
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Load eval data
    # ------------------------------------------------------------------
    logger.info("Loading eval data from %s", eval_data_path)
    eval_samples: list[dict] = []
    with open(eval_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_samples.append(json.loads(line))

    eval_texts = [
        s["text"] for s in eval_samples
        if isinstance(s.get("text"), str) and s["text"]
    ]
    logger.info("Loaded %d eval probe texts", len(eval_texts))

    # ------------------------------------------------------------------
    # 2. Generate inference probes (100 — enough for full sweep)
    # ------------------------------------------------------------------
    n_inference_max = 100
    logger.info("Generating %d inference probes (StarProblem few-shot)", n_inference_max)
    inference_texts = generate_inference_probes(n_inference_max, seed=seed)
    logger.info(
        "Inference probe lengths: min=%d, max=%d, mean=%.0f chars",
        min(len(t) for t in inference_texts),
        max(len(t) for t in inference_texts),
        sum(len(t) for t in inference_texts) / len(inference_texts),
    )
    logger.info(
        "Eval probe lengths: min=%d, max=%d, mean=%.0f chars",
        min(len(t) for t in eval_texts),
        max(len(t) for t in eval_texts),
        sum(len(t) for t in eval_texts) / len(eval_texts),
    )

    # ------------------------------------------------------------------
    # 3. Load base model, collect activations
    # ------------------------------------------------------------------
    logger.info("Loading base model from %s", model_path)
    base_model, tokenizer = backend.load_model(model_path)

    logger.info("Collecting base activations on eval probes (%d texts)...", len(eval_texts))
    t0 = time.time()
    base_eval_mean = collect_activations_single_probe(
        backend, base_model, tokenizer, eval_texts, pooling="mean",
    )
    logger.info("  eval mean-pooled: %.1fs, %d layers", time.time() - t0, len(base_eval_mean))

    logger.info("Collecting base activations on eval probes (last_token)...")
    t0 = time.time()
    base_eval_last = collect_activations_single_probe(
        backend, base_model, tokenizer, eval_texts, pooling="last_token",
    )
    logger.info("  eval last-token: %.1fs", time.time() - t0)

    logger.info("Collecting base activations on inference probes (%d texts)...", len(inference_texts))
    t0 = time.time()
    base_inf_mean = collect_activations_single_probe(
        backend, base_model, tokenizer, inference_texts, pooling="mean",
    )
    logger.info("  inference mean-pooled: %.1fs, %d layers", time.time() - t0, len(base_inf_mean))

    logger.info("Collecting base activations on inference probes (last_token)...")
    t0 = time.time()
    base_inf_last = collect_activations_single_probe(
        backend, base_model, tokenizer, inference_texts, pooling="last_token",
    )
    logger.info("  inference last-token: %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # 4. Load adapted model, collect activations
    # ------------------------------------------------------------------
    logger.info("Loading adapted model with adapter from %s", adapter_path)
    adapted_model, _ = backend.load_model(model_path, adapter_path=adapter_path)

    logger.info("Collecting adapted activations on eval probes (mean)...")
    t0 = time.time()
    adapted_eval_mean = collect_activations_single_probe(
        backend, adapted_model, tokenizer, eval_texts, pooling="mean",
    )
    logger.info("  eval mean-pooled: %.1fs", time.time() - t0)

    logger.info("Collecting adapted activations on eval probes (last_token)...")
    t0 = time.time()
    adapted_eval_last = collect_activations_single_probe(
        backend, adapted_model, tokenizer, eval_texts, pooling="last_token",
    )
    logger.info("  eval last-token: %.1fs", time.time() - t0)

    logger.info("Collecting adapted activations on inference probes (mean)...")
    t0 = time.time()
    adapted_inf_mean = collect_activations_single_probe(
        backend, adapted_model, tokenizer, inference_texts, pooling="mean",
    )
    logger.info("  inference mean-pooled: %.1fs", time.time() - t0)

    logger.info("Collecting adapted activations on inference probes (last_token)...")
    t0 = time.time()
    adapted_inf_last = collect_activations_single_probe(
        backend, adapted_model, tokenizer, inference_texts, pooling="last_token",
    )
    logger.info("  inference last-token: %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # 5. Probe-count sweep: eval manifold (mean pooling)
    # ------------------------------------------------------------------
    eval_sweep_counts = [10, 20, 50, 100, len(eval_texts)]
    eval_sweep_counts = sorted(set(c for c in eval_sweep_counts if c <= len(eval_texts)))

    logger.info("=== Eval manifold sweep (mean pooling) ===")
    eval_mean_sweeps: list[dict] = []
    for n in eval_sweep_counts:
        sp = measure_sweep_point(
            backend, base_eval_mean, adapted_eval_mean,
            n_probes=n, manifold="eval", pooling="mean",
            rng=random.Random(seed),
        )
        eval_mean_sweeps.append(_sweep_to_dict(sp))
        logger.info("  n=%3d: min_cka=%.4f  mean_cka=%.4f", sp.n_probes, sp.min_cka, sp.mean_cka)

    # ------------------------------------------------------------------
    # 6. Probe-count sweep: eval manifold (last_token pooling)
    # ------------------------------------------------------------------
    logger.info("=== Eval manifold sweep (last_token pooling) ===")
    eval_last_sweeps: list[dict] = []
    for n in eval_sweep_counts:
        sp = measure_sweep_point(
            backend, base_eval_last, adapted_eval_last,
            n_probes=n, manifold="eval", pooling="last_token",
            rng=random.Random(seed),
        )
        eval_last_sweeps.append(_sweep_to_dict(sp))
        logger.info("  n=%3d: min_cka=%.4f  mean_cka=%.4f", sp.n_probes, sp.min_cka, sp.mean_cka)

    # ------------------------------------------------------------------
    # 7. Probe-count sweep: inference manifold (mean pooling)
    # ------------------------------------------------------------------
    inf_sweep_counts = [10, 30, 50, 100]

    logger.info("=== Inference manifold sweep (mean pooling) ===")
    inf_mean_sweeps: list[dict] = []
    for n in inf_sweep_counts:
        sp = measure_sweep_point(
            backend, base_inf_mean, adapted_inf_mean,
            n_probes=n, manifold="inference", pooling="mean",
            rng=random.Random(seed),
        )
        inf_mean_sweeps.append(_sweep_to_dict(sp))
        logger.info("  n=%3d: min_cka=%.4f  mean_cka=%.4f", sp.n_probes, sp.min_cka, sp.mean_cka)

    # ------------------------------------------------------------------
    # 8. Probe-count sweep: inference manifold (last_token pooling)
    # ------------------------------------------------------------------
    logger.info("=== Inference manifold sweep (last_token pooling) ===")
    inf_last_sweeps: list[dict] = []
    for n in inf_sweep_counts:
        sp = measure_sweep_point(
            backend, base_inf_last, adapted_inf_last,
            n_probes=n, manifold="inference", pooling="last_token",
            rng=random.Random(seed),
        )
        inf_last_sweeps.append(_sweep_to_dict(sp))
        logger.info("  n=%3d: min_cka=%.4f  mean_cka=%.4f", sp.n_probes, sp.min_cka, sp.mean_cka)

    # ------------------------------------------------------------------
    # 9. Assemble results and verdicts
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "metadata": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "eval_data_path": eval_data_path,
            "seed": seed,
            "n_eval_probes": len(eval_texts),
            "n_inference_probes": len(inference_texts),
            "eval_char_lengths": {
                "min": min(len(t) for t in eval_texts),
                "max": max(len(t) for t in eval_texts),
                "mean": sum(len(t) for t in eval_texts) / len(eval_texts),
            },
            "inference_char_lengths": {
                "min": min(len(t) for t in inference_texts),
                "max": max(len(t) for t in inference_texts),
                "mean": sum(len(t) for t in inference_texts) / len(inference_texts),
            },
        },
        "sweeps": {
            "eval_mean": eval_mean_sweeps,
            "eval_last_token": eval_last_sweeps,
            "inference_mean": inf_mean_sweeps,
            "inference_last_token": inf_last_sweeps,
        },
        "verdicts": _compute_verdicts(
            eval_mean_sweeps, eval_last_sweeps,
            inf_mean_sweeps, inf_last_sweeps,
        ),
    }

    # Write results
    results_file = output_path / "diagnosis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", results_file)

    # Print summary
    _print_summary(results)

    return results


def _sweep_to_dict(sp: SweepPoint) -> dict[str, Any]:
    """Convert SweepPoint to serializable dict."""
    return {
        "n_probes": sp.n_probes,
        "manifold": sp.manifold,
        "pooling": sp.pooling,
        "min_cka": sp.min_cka,
        "mean_cka": sp.mean_cka,
        "n_layers": sp.n_layers,
        "per_layer": sp.per_layer,
    }


def _compute_verdicts(
    eval_mean: list[dict],
    eval_last: list[dict],
    inf_mean: list[dict],
    inf_last: list[dict],
) -> dict[str, Any]:
    """Compute diagnostic verdicts from sweep data."""
    verdicts: dict[str, Any] = {}

    # Test 1a: Does eval CKA collapse at n=10?
    # If eval min_cka at n=10 << eval min_cka at full count → artifact
    eval_at_10 = next((s for s in eval_mean if s["n_probes"] == 10), None)
    eval_at_full = eval_mean[-1] if eval_mean else None

    if eval_at_10 and eval_at_full:
        eval_10_min = eval_at_10["min_cka"]
        eval_full_min = eval_at_full["min_cka"]
        eval_collapse_at_10 = eval_full_min - eval_10_min

        verdicts["eval_cka_at_n10"] = eval_10_min
        verdicts["eval_cka_at_full"] = eval_full_min
        verdicts["eval_collapse_magnitude"] = eval_collapse_at_10
        verdicts["hypothesis_A_eval_evidence"] = (
            "STRONG" if eval_collapse_at_10 > 0.3 else
            "MODERATE" if eval_collapse_at_10 > 0.1 else
            "WEAK"
        )

    # Test 1b: Does inference CKA recover as n increases?
    if inf_mean:
        inf_10 = next((s for s in inf_mean if s["n_probes"] == 10), None)
        inf_100 = next((s for s in inf_mean if s["n_probes"] == 100), None)

        if inf_10 and inf_100:
            inf_recovery = inf_100["min_cka"] - inf_10["min_cka"]
            verdicts["inference_cka_at_n10"] = inf_10["min_cka"]
            verdicts["inference_cka_at_n100"] = inf_100["min_cka"]
            verdicts["inference_recovery"] = inf_recovery

            # If inference CKA recovers toward eval level → Hypothesis A
            # If inference CKA stays collapsed → Hypothesis B
            if inf_100["min_cka"] > 0.8:
                verdicts["hypothesis_A_inference_evidence"] = "STRONG"
                verdicts["hypothesis_B_inference_evidence"] = "WEAK"
            elif inf_recovery > 0.3:
                verdicts["hypothesis_A_inference_evidence"] = "MODERATE"
                verdicts["hypothesis_B_inference_evidence"] = "MODERATE"
            else:
                verdicts["hypothesis_A_inference_evidence"] = "WEAK"
                verdicts["hypothesis_B_inference_evidence"] = "STRONG"

    # Test 1d: Does last-token pooling help inference CKA?
    if inf_last:
        inf_last_100 = next((s for s in inf_last if s["n_probes"] == 100), None)
        inf_mean_100 = next((s for s in inf_mean if s["n_probes"] == 100), None)

        if inf_last_100 and inf_mean_100:
            pooling_improvement = inf_last_100["min_cka"] - inf_mean_100["min_cka"]
            verdicts["inference_last_token_cka_n100"] = inf_last_100["min_cka"]
            verdicts["inference_mean_cka_n100"] = inf_mean_100["min_cka"]
            verdicts["pooling_improvement"] = pooling_improvement
            verdicts["mean_pooling_artifact"] = (
                "SIGNIFICANT" if pooling_improvement > 0.2 else
                "MINOR" if pooling_improvement > 0.05 else
                "NONE"
            )

    # Final verdict
    a_evidence = []
    b_evidence = []
    for key, val in verdicts.items():
        if "hypothesis_A" in key:
            a_evidence.append(val)
        if "hypothesis_B" in key:
            b_evidence.append(val)

    a_strong = sum(1 for e in a_evidence if e == "STRONG")
    b_strong = sum(1 for e in b_evidence if e == "STRONG")

    if a_strong > b_strong:
        verdicts["overall"] = "HYPOTHESIS_A_DOMINANT"
        verdicts["summary"] = (
            "Measurement artifact dominates. CKA is unreliable at low probe counts "
            "in high-dimensional space. Increase minimum probe count."
        )
    elif b_strong > a_strong:
        verdicts["overall"] = "HYPOTHESIS_B_DOMINANT"
        verdicts["summary"] = (
            "Real manifold split. Training preserves eval geometry but disrupts "
            "inference geometry. CKA verification needs inference-diverse probes."
        )
    else:
        verdicts["overall"] = "MIXED"
        verdicts["summary"] = (
            "Both artifact and real split contribute. Need more probes AND "
            "inference-diverse probes for reliable CKA."
        )

    return verdicts


def _print_summary(results: dict[str, Any]) -> None:
    """Print human-readable diagnostic summary."""
    verdicts = results["verdicts"]
    meta = results["metadata"]

    print("\n" + "=" * 72)
    print("INFERENCE CKA COLLAPSE DIAGNOSIS")
    print("=" * 72)

    print(f"\nModel: {meta['model_path']}")
    print(f"Adapter: {meta['adapter_path']}")
    print(f"Eval probes: {meta['n_eval_probes']} (mean {meta['eval_char_lengths']['mean']:.0f} chars)")
    print(f"Inference probes: {meta['n_inference_probes']} (mean {meta['inference_char_lengths']['mean']:.0f} chars)")

    print("\n--- Eval Manifold (mean pooling) ---")
    for s in results["sweeps"]["eval_mean"]:
        print(f"  n={s['n_probes']:>3d}: min_cka={s['min_cka']:.4f}  mean_cka={s['mean_cka']:.4f}")

    print("\n--- Eval Manifold (last_token pooling) ---")
    for s in results["sweeps"]["eval_last_token"]:
        print(f"  n={s['n_probes']:>3d}: min_cka={s['min_cka']:.4f}  mean_cka={s['mean_cka']:.4f}")

    print("\n--- Inference Manifold (mean pooling) ---")
    for s in results["sweeps"]["inference_mean"]:
        print(f"  n={s['n_probes']:>3d}: min_cka={s['min_cka']:.4f}  mean_cka={s['mean_cka']:.4f}")

    print("\n--- Inference Manifold (last_token pooling) ---")
    for s in results["sweeps"]["inference_last_token"]:
        print(f"  n={s['n_probes']:>3d}: min_cka={s['min_cka']:.4f}  mean_cka={s['mean_cka']:.4f}")

    print("\n--- Verdicts ---")
    for key, val in verdicts.items():
        if key in ("overall", "summary"):
            continue
        print(f"  {key}: {val}")

    print(f"\n  OVERALL: {verdicts.get('overall', 'UNKNOWN')}")
    print(f"  {verdicts.get('summary', '')}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose inference-manifold CKA collapse",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to base model directory",
    )
    parser.add_argument(
        "--adapter", required=True,
        help="Path to adapter directory (trial artifacts)",
    )
    parser.add_argument(
        "--eval-data", required=True,
        help="Path to eval JSONL file (benchmark_val.jsonl)",
    )
    parser.add_argument(
        "--output", default="results/inference_cka_diagnosis/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).is_dir():
        logger.error("Model path not found: %s", args.model)
        sys.exit(1)
    if not Path(args.adapter).is_dir():
        logger.error("Adapter path not found: %s", args.adapter)
        sys.exit(1)
    if not Path(args.eval_data).is_file():
        logger.error("Eval data not found: %s", args.eval_data)
        sys.exit(1)

    run_diagnosis(
        model_path=args.model,
        adapter_path=args.adapter,
        eval_data_path=args.eval_data,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
