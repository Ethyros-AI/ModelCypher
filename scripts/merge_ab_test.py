#!/usr/bin/env python3
"""Experiment: Merge Projector A/B Test.

Compares binary eigenvalue mask (pre-Tikhonov) vs MP-weighted Tikhonov
null-space projector on an identical model pair. Both arms use the same
probes, density weights, and alignment transforms — only the projector
mode differs.

Arms:
    A (binary):   Hard eigenvalue mask — svd_rank_threshold() → boolean cutoff
    B (tikhonov): MP-weighted continuous weights — w_i = λ_i / (λ_i + α)

Metrics (per arm):
    - mean_preserved_fraction (from transplant stage)
    - CKA on held-out validation probes (model vs target)
    - Perplexity on benchmark_val.jsonl
    - Max 4-gram repetition rate (degeneration)

Usage:
    poetry run python scripts/merge_ab_test.py \
        -s /Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16 \
        -t /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        -o /Volumes/CodeCypher/models/merged/ab-test

    # Custom eval dataset
    poetry run python scripts/merge_ab_test.py \
        -s SOURCE -t TARGET -o OUTPUT \
        --eval-dataset data/training/benchmark_val.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("merge_ab_test")

DEFAULT_SOURCE = "/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16"
DEFAULT_TARGET = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_OUTPUT = "/Volumes/CodeCypher/models/merged/ab-test"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"

TEST_PROMPTS = [
    "Explain what a prime number is.",
    "What causes the seasons on Earth?",
    "Describe how a binary search works.",
    "What is the difference between a stack and a queue?",
    "How does photosynthesis work?",
]


# ── Utilities ────────────────────────────────────────────────────────────


def _evaluate_perplexity(
    model_path: str,
    dataset_path: str,
) -> dict[str, float]:
    """Compute perplexity of a model on a dataset.

    Lightweight standalone implementation — avoids EvaluationService dependency.
    """
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    logger.info(
        "Evaluating perplexity: model=%s dataset=%s",
        Path(model_path).name,
        Path(dataset_path).name,
    )

    model, tokenizer = backend.load_model(str(model_path))

    samples = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        samples.append(text)
                except json.JSONDecodeError:
                    continue

    if not samples:
        del model, tokenizer
        gc.collect()
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
        target_log_scores = backend.take_along_axis(
            log_scores, targets_expanded, axis=-1
        )
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
        perplexity,
        average_loss,
        len(samples),
        total_tokens,
    )

    del model, tokenizer
    gc.collect()

    return {
        "average_loss": average_loss,
        "perplexity": perplexity,
        "n_samples": len(samples),
        "n_tokens": total_tokens,
    }


def _evaluate_degeneration(
    model_path: str,
    prompts: list[str],
    max_tokens: int = 256,
) -> dict[str, float]:
    """Measure 4-gram repetition rate on generated completions."""
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.training.degeneration import fourgram_repetition_rate

    backend = get_default_backend()
    logger.info("Evaluating degeneration: model=%s", Path(model_path).name)

    model, tokenizer = backend.load_model(str(model_path))

    rates = []
    completions = []
    for prompt in prompts:
        try:
            response = backend.generate(
                model, tokenizer, prompt, max_tokens=max_tokens
            )
        except Exception as e:
            logger.warning("Generation failed for prompt '%s...': %s", prompt[:30], e)
            response = ""
        rate = fourgram_repetition_rate(response)
        rates.append(rate)
        completions.append({"prompt": prompt, "response": response, "repetition": rate})
        logger.info("  prompt='%s...' repetition=%.4f", prompt[:30], rate)

    del model, tokenizer
    gc.collect()

    mean_rate = sum(rates) / len(rates) if rates else 0.0
    max_rate = max(rates) if rates else 0.0

    return {
        "mean_repetition": mean_rate,
        "max_repetition": max_rate,
        "n_prompts": len(prompts),
        "completions": completions,
    }


def _evaluate_cka_vs_target(
    merged_model_path: str,
    target_model_path: str,
    eval_dataset_path: str,
    n_probes: int = 30,
    max_seq_len: int = 128,
) -> dict[str, float]:
    """Measure CKA between merged model and target on held-out samples.

    CKA close to 1.0 → merged model preserves target behavior.
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import compute_cka

    backend = get_default_backend()
    logger.info(
        "Evaluating CKA: merged=%s vs target=%s",
        Path(merged_model_path).name,
        Path(target_model_path).name,
    )

    # Load eval samples
    samples = []
    with open(eval_dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        samples.append(text)
                except json.JSONDecodeError:
                    continue

    if len(samples) > n_probes:
        samples = samples[:n_probes]

    # Collect final-layer activations from both models
    def _collect_activations(model_path: str) -> list:
        model, tokenizer = backend.load_model(str(model_path))
        all_acts = []
        for text in samples:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:max_seq_len]
            input_arr = backend.reshape(backend.array(tokens), (1, -1))
            # Forward pass — get logits, collect hidden state via model internals
            logits = model(input_arr)
            backend.eval(logits)
            # Use mean of logits across sequence as a representation
            # (logits space is the readout-level embedding)
            mean_logit = backend.mean(logits[0], axis=0)
            backend.eval(mean_logit)
            all_acts.append(mean_logit)
        del model, tokenizer
        gc.collect()
        return all_acts

    merged_acts = _collect_activations(merged_model_path)
    target_acts = _collect_activations(target_model_path)

    if not merged_acts or not target_acts:
        return {"cka": 0.0, "n_probes": 0}

    # Stack into matrices
    merged_matrix = backend.stack(merged_acts)
    target_matrix = backend.stack(target_acts)
    backend.eval(merged_matrix, target_matrix)

    cka_result = compute_cka(merged_matrix, target_matrix, backend=backend)

    logger.info("CKA (merged vs target): %.6f", cka_result.cka)

    return {
        "cka": cka_result.cka,
        "n_probes": len(merged_acts),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def _run_merge_arm(
    arm_name: str,
    projector_mode: str,
    source_path: str,
    target_path: str,
    output_dir: Path,
) -> dict:
    """Run one arm of the A/B test (merge only, no eval)."""
    from modelcypher.cli.composition import get_merge_service

    logger.info("=" * 60)
    logger.info("ARM %s: projector_mode=%s", arm_name, projector_mode)
    logger.info("=" * 60)

    arm_output = str(output_dir / arm_name)

    merger = get_merge_service()
    t0 = time.time()
    result = merger.merge(
        source_path=source_path,
        target_path=target_path,
        output_dir=arm_output,
        projector_mode=projector_mode,
    )
    elapsed = time.time() - t0

    metrics = {
        "arm": arm_name,
        "projector_mode": projector_mode,
        "mean_preserved_fraction": result.mean_preserved_fraction,
        "mean_procrustes_error": result.mean_procrustes_error,
        "layer_count": result.layer_count,
        "weight_count": result.weight_count,
        "output_path": arm_output,
        "merge_seconds": round(elapsed, 2),
        "transplant_metrics": result.transplant_metrics,
    }

    logger.info(
        "ARM %s complete: preserved=%.4f procrustes=%.6f (%.1fs)",
        arm_name,
        result.mean_preserved_fraction,
        result.mean_procrustes_error,
        elapsed,
    )

    # Free merge service memory
    del merger, result
    gc.collect()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="A/B test: binary vs Tikhonov null-space projector"
    )
    parser.add_argument(
        "-s", "--source", default=DEFAULT_SOURCE, help="Source model path"
    )
    parser.add_argument(
        "-t", "--target", default=DEFAULT_TARGET, help="Target model path"
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT, help="Output directory"
    )
    parser.add_argument(
        "--eval-dataset", default=DEFAULT_EVAL, help="Evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--skip-ppl", action="store_true", help="Skip perplexity evaluation"
    )
    parser.add_argument(
        "--skip-cka", action="store_true", help="Skip CKA evaluation"
    )
    parser.add_argument(
        "--skip-degen", action="store_true", help="Skip degeneration evaluation"
    )
    args = parser.parse_args()

    # Validate paths
    source_path = Path(args.source)
    target_path = Path(args.target)
    if not source_path.exists():
        logger.error("Source model not found: %s", source_path)
        return
    if not target_path.exists():
        logger.error("Target model not found: %s", target_path)
        return

    output_dir = Path(args.output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Merge A/B Test")
    logger.info("  Source: %s", source_path)
    logger.info("  Target: %s", target_path)
    logger.info("  Output: %s", run_dir)
    logger.info("  Eval:   %s", args.eval_dataset)

    # Initialize backend once
    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()

    # ── Run both merge arms ──────────────────────────────────────────────
    arm_a = _run_merge_arm(
        arm_name="binary",
        projector_mode="binary",
        source_path=str(source_path),
        target_path=str(target_path),
        output_dir=run_dir,
    )

    arm_b = _run_merge_arm(
        arm_name="tikhonov",
        projector_mode="tikhonov",
        source_path=str(source_path),
        target_path=str(target_path),
        output_dir=run_dir,
    )

    # ── Evaluate both arms ───────────────────────────────────────────────
    for arm in [arm_a, arm_b]:
        arm_path = arm["output_path"]

        if not args.skip_ppl:
            arm["ppl"] = _evaluate_perplexity(arm_path, args.eval_dataset)
        if not args.skip_degen:
            arm["degeneration"] = _evaluate_degeneration(arm_path, TEST_PROMPTS)
        if not args.skip_cka:
            arm["cka_vs_target"] = _evaluate_cka_vs_target(
                arm_path, str(target_path), args.eval_dataset
            )

    # ── Comparison ───────────────────────────────────────────────────────
    comparison = {
        "preserved_fraction": {
            "binary": arm_a["mean_preserved_fraction"],
            "tikhonov": arm_b["mean_preserved_fraction"],
            "delta": arm_b["mean_preserved_fraction"] - arm_a["mean_preserved_fraction"],
        },
        "procrustes_error": {
            "binary": arm_a["mean_procrustes_error"],
            "tikhonov": arm_b["mean_procrustes_error"],
            "delta": arm_b["mean_procrustes_error"] - arm_a["mean_procrustes_error"],
        },
    }

    if not args.skip_ppl:
        comparison["perplexity"] = {
            "binary": arm_a["ppl"]["perplexity"],
            "tikhonov": arm_b["ppl"]["perplexity"],
            "delta": arm_b["ppl"]["perplexity"] - arm_a["ppl"]["perplexity"],
        }

    if not args.skip_degen:
        comparison["max_repetition"] = {
            "binary": arm_a["degeneration"]["max_repetition"],
            "tikhonov": arm_b["degeneration"]["max_repetition"],
            "delta": (
                arm_b["degeneration"]["max_repetition"]
                - arm_a["degeneration"]["max_repetition"]
            ),
        }

    if not args.skip_cka:
        comparison["cka_vs_target"] = {
            "binary": arm_a["cka_vs_target"]["cka"],
            "tikhonov": arm_b["cka_vs_target"]["cka"],
            "delta": arm_b["cka_vs_target"]["cka"] - arm_a["cka_vs_target"]["cka"],
        }

    # ── Output ───────────────────────────────────────────────────────────
    full_result = {
        "experiment": "merge_projector_ab_test",
        "timestamp": timestamp,
        "source": str(source_path),
        "target": str(target_path),
        "eval_dataset": args.eval_dataset,
        "arms": {"binary": arm_a, "tikhonov": arm_b},
        "comparison": comparison,
    }

    result_path = run_dir / "merge_ab_test.json"
    with open(result_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for metric, values in comparison.items():
        logger.info(
            "  %-20s binary=%.6f  tikhonov=%.6f  delta=%+.6f",
            metric,
            values["binary"],
            values["tikhonov"],
            values["delta"],
        )
    logger.info("Full results: %s", result_path)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
