#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Exp 5: REINFORCE-only (no CE) experiment.
#
# Hypothesis: CE and REINFORCE on the same parameters create conflicting
# gradients. REINFORCE alone does not degrade.
#
# This is a standalone script because train_loop always runs CE in the
# inner loop. We reuse existing functions for outcome collection,
# batch preparation, loss, and evaluation.
#
# Usage:
#   poetry run python scripts/reinforce_only_experiment.py
"""Exp 5: REINFORCE-only training (no CE) on LFM2-350M."""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("reinforce_only_exp5")

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
OUTPUT_DIR = Path("/Volumes/CodeCypher/models/experiments/reinforce-ablation/exp5")
SEED = 42
MAX_ITERS = 65  # Full epoch
SEQ_LENGTH = 256
N_EVAL_PROBLEMS = 25


def main() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten as mlx_flatten

    start_ts = datetime.now(tz=timezone.utc).isoformat()
    log.info("=" * 60)
    log.info("Exp 5: REINFORCE-Only (No CE)")
    log.info("Model: %s", MODEL_PATH)
    log.info("Seed: %d", SEED)
    log.info("LR: derived from spectral ceiling (sigma_k_min / sigma_max)")
    log.info("Max iters: %d", MAX_ITERS)
    log.info("Start: %s", start_ts)
    log.info("=" * 60)

    if not Path(MODEL_PATH).exists():
        log.error("Model path does not exist: %s", MODEL_PATH)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics.jsonl"
    run_log_path = OUTPUT_DIR / "run_log.json"

    # --- Load model + tokenizer ---
    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()
    backend = service._backend
    adapter = service._adapter

    model, tokenizer = backend.load_model(MODEL_PATH)

    # --- Analyze geometry + inject NB-LoRA ---
    from modelcypher.core.domain.training.geometric_lora import (
        analyze_weight_geometries,
        select_target_modules,
        apply_data_rank_ceiling,
        compute_coupled_ranks,
    )

    weights = adapter.extract_weight_matrices(model)
    geometries = analyze_weight_geometries(weights, backend)
    target_modules = select_target_modules(geometries)

    # Data-rank ceiling
    # rank_coupling imports already handled above
    from modelcypher.core.domain.dataset_loading import load_jsonl_dataset

    train_samples = load_jsonl_dataset(Path(DATASET_PATH))
    coupled_ranks = compute_coupled_ranks(geometries, target_modules)
    data_ceiling_ranks = apply_data_rank_ceiling(
        coupled_ranks, n_samples=len(train_samples),
    )

    n_lora_layers = adapter.inject_nb_lora(
        model, geometries, target_modules,
        safety_margin=0.9,
        rank_overrides=data_ceiling_ranks,
    )
    adapter.freeze_and_apply_lora(model)
    log.info("NB-LoRA injected: %d layers", n_lora_layers)

    sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)
    sigma_k_min = min(g.sigma_k for g in geometries.values() if g.sigma_k > 0)

    # --- Derive LR from spectral ceiling (Weyl 1912) ---
    tokenized_dataset = adapter.prepare_dataset(train_samples, tokenizer)
    batch_size = adapter.derive_critical_batch_size(
        model, tokenized_dataset[:100], SEQ_LENGTH,
    )
    log.info("Batch size (B_crit): %d", batch_size)

    eta = adapter._derive_spectral_ceiling(
        sigma_k_min=sigma_k_min,
        sigma_max_global=sigma_max,
        lr_override=None,
    )
    log.info("Spectral ceiling: sigma_k_min=%.4e, sigma_max=%.4e, eta=%.4e",
             sigma_k_min, sigma_max, eta)

    optimizer = optim.SGD(learning_rate=eta)
    mx.eval(model.parameters(), optimizer.state)

    # --- Set up online eval ---
    from modelcypher.core.domain.training.online_eval import (
        create_eval_problem_set,
        evaluate_correctness,
    )

    eval_problems = create_eval_problem_set(
        n_problems=N_EVAL_PROBLEMS, seed=SEED + 1,
    )

    def _gen_fn(prompt: str, max_toks: int) -> str:
        return backend.generate(model, tokenizer, prompt, max_toks)

    # Baseline measurement
    baseline_result = evaluate_correctness(
        problems=eval_problems,
        generate_fn=_gen_fn,
        epoch=0,
        baseline_correct_ids=None,
        max_tokens=SEQ_LENGTH,
    )
    baseline_ids = baseline_result.correct_ids
    log.info(
        "Baseline: %d/%d (%.1f%%)",
        baseline_result.n_correct, baseline_result.n_total,
        baseline_result.accuracy * 100,
    )

    # --- Set up REINFORCE outcome problems ---
    outcome_problems = create_eval_problem_set(
        n_problems=N_EVAL_PROBLEMS, seed=SEED + 2,
    )

    from modelcypher.core.domain.star.prompting import default_few_shot_examples
    from modelcypher.core.domain.training.outcome_objective import collect_outcomes
    from modelcypher.backends.mlx_training_adapter import (
        make_outcome_loss,
        prepare_outcome_batches,
    )

    n_variants = len(default_few_shot_examples())

    # --- Cayley preconditioner setup ---
    use_cayley = True  # Match main training path

    # --- Training loop: REINFORCE only, no CE ---
    metrics_records = []
    eval_history = []
    all_evals_pass = True

    t_start = time.monotonic()

    for iteration in range(1, MAX_ITERS + 1):
        iter_start = time.monotonic()

        # Phase A: collect outcomes
        def _outcome_gen_fn(prompt: str, max_toks: int) -> str:
            return backend.generate(model, tokenizer, prompt, max_toks)

        def _outcome_tok_fn(text: str) -> list[int]:
            return tokenizer.encode(text)

        outcome_result = collect_outcomes(
            problems=outcome_problems,
            generate_fn=_outcome_gen_fn,
            tokenize_fn=_outcome_tok_fn,
            n_variants=n_variants,
            max_tokens=SEQ_LENGTH,
        )

        active_completions = [
            (c.tokens, c.advantage)
            for c in outcome_result.completions
            if c.advantage != 0.0
        ]

        n_steps = 0
        last_o_eta = None
        last_o_grad_norm = None

        if active_completions:
            outcome_batches = prepare_outcome_batches(
                active_completions, batch_size, SEQ_LENGTH,
            )
            outcome_loss_fn = make_outcome_loss()
            outcome_vg = nn.value_and_grad(model, outcome_loss_fn)

            # REINFORCE step size: sqrt(eps) * ||θ|| (no CE to calibrate from)
            sqrt_eps = math.sqrt(float(mx.finfo(mx.float32).eps))
            trainable_params = dict(mlx_flatten(model.trainable_parameters()))
            theta_sq = mx.array(0.0, dtype=mx.float32)
            for p in trainable_params.values():
                if p.size > 0:
                    theta_sq = theta_sq + mx.sum(p * p)
            mx.eval(theta_sq)
            theta_norm = float(mx.sqrt(theta_sq).item())
            target_step_norm = (sqrt_eps * theta_norm) if theta_norm > 0 else sqrt_eps

            for ob_batch, ob_lengths, ob_advantages in outcome_batches:
                (o_loss, o_ntoks), o_grad = outcome_vg(
                    model, ob_batch, ob_lengths, ob_advantages,
                )

                # Cayley preconditioner
                if use_cayley:
                    o_grad, _ = adapter._apply_cayley_preconditioner(
                        model, o_grad,
                    )

                # Measure preconditioned gradient norm
                o_flat = [
                    p.reshape(-1)
                    for _, p in mlx_flatten(o_grad)
                    if p.size > 0
                ]
                if o_flat:
                    o_grad_norm_val = mx.sqrt(
                        sum(mx.sum(p * p) for p in o_flat)
                    ).item()
                else:
                    o_grad_norm_val = 1.0

                # Scale LR
                if o_grad_norm_val > 0:
                    o_eta = min(eta, target_step_norm / o_grad_norm_val)
                else:
                    o_eta = eta

                optimizer.learning_rate = mx.array(o_eta)
                optimizer.update(model, o_grad)
                mx.eval(model.parameters(), optimizer.state)
                adapter._clamp_all_scales(model)
                n_steps += 1
                last_o_eta = o_eta
                last_o_grad_norm = o_grad_norm_val

        iter_elapsed = time.monotonic() - iter_start

        # Online eval every 10 iterations
        eval_acc = None
        eval_n_correct = None
        eval_degraded = None
        if iteration % 10 == 0 or iteration == 1:
            eval_result = evaluate_correctness(
                problems=eval_problems,
                generate_fn=_gen_fn,
                epoch=iteration,
                baseline_correct_ids=baseline_ids,
                max_tokens=SEQ_LENGTH,
            )
            eval_acc = eval_result.accuracy
            eval_n_correct = eval_result.n_correct
            eval_degraded = eval_result.degraded
            eval_history.append({
                "iteration": iteration,
                "n_correct": eval_n_correct,
                "n_total": eval_result.n_total,
                "accuracy": eval_acc,
                "degraded": eval_degraded,
            })
            log.info(
                "Iter %d: eval %d/%d (%.1f%%), signal=%.1f%%, steps=%d, "
                "o_eta=%.2e, o_grad_norm=%.2e, elapsed=%.1fs",
                iteration, eval_n_correct, eval_result.n_total,
                eval_acc * 100,
                outcome_result.signal_density * 100,
                n_steps,
                last_o_eta or 0,
                last_o_grad_norm or 0,
                iter_elapsed,
            )
            if eval_n_correct < 18:
                all_evals_pass = False
                # Don't stop early — we want the full trajectory
        else:
            log.info(
                "Iter %d: signal=%.1f%%, steps=%d, elapsed=%.1fs",
                iteration,
                outcome_result.signal_density * 100,
                n_steps,
                iter_elapsed,
            )

        # Record metrics
        record = {
            "experiment": "exp5",
            "iteration": iteration,
            "outcome_n_problems": outcome_result.n_problems,
            "outcome_n_active": len(active_completions),
            "outcome_signal_density": outcome_result.signal_density,
            "outcome_n_steps": n_steps,
            "outcome_o_eta": last_o_eta,
            "outcome_o_grad_norm": last_o_grad_norm,
            "online_eval_accuracy": eval_acc,
            "online_eval_n_correct": eval_n_correct,
            "online_eval_degraded": eval_degraded,
            "elapsed_seconds": iter_elapsed,
        }
        metrics_records.append(record)

    total_elapsed = time.monotonic() - t_start

    # Write metrics
    with open(metrics_path, "w") as f:
        for r in metrics_records:
            f.write(json.dumps(r, default=str) + "\n")
    log.info("Metrics written to: %s", metrics_path)

    # Evaluate success
    final_eval = eval_history[-1] if eval_history else None
    no_degradation = all(e["n_correct"] >= 18 for e in eval_history) if eval_history else False
    final_improved = final_eval["n_correct"] > 18 if final_eval else False

    run_log = {
        "experiment": "exp5",
        "description": "REINFORCE-only (no CE)",
        "timestamp": start_ts,
        "model_path": MODEL_PATH,
        "seed": SEED,
        "derived_lr": eta,
        "sigma_k_min": sigma_k_min,
        "sigma_max": sigma_max,
        "max_iters": MAX_ITERS,
        "elapsed_seconds": round(total_elapsed, 1),
        "baseline_n_correct": baseline_result.n_correct,
        "baseline_n_total": baseline_result.n_total,
        "eval_history": eval_history,
        "success_criteria": {
            "no_degradation_all_checkpoints": no_degradation,
            "final_improved_above_18": final_improved,
        },
    }

    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    log.info("Run log saved to: %s", run_log_path)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY: exp5 (REINFORCE-only)")
    log.info("  Elapsed: %.1f seconds", total_elapsed)
    log.info("  Baseline: %d/%d", baseline_result.n_correct, baseline_result.n_total)
    if final_eval:
        log.info("  Final eval: %d/%d", final_eval["n_correct"], final_eval["n_total"])
        log.info("  No degradation (all >= 18): %s", no_degradation)
        log.info("  Final improved (> 18): %s", final_improved)
        if no_degradation and final_improved:
            log.info("  >>> PASS <<<")
        elif no_degradation:
            log.info("  >>> PARTIAL PASS (no degradation, no improvement) <<<")
        else:
            log.info("  >>> FAIL (degradation detected) <<<")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
