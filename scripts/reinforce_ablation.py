#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# REINFORCE degradation ablation study.
#
# Usage:
#   poetry run python scripts/reinforce_ablation.py exp0
#   poetry run python scripts/reinforce_ablation.py exp1
#   ...
#   poetry run python scripts/reinforce_ablation.py exp8
"""Controlled ablation study for REINFORCE degradation on LFM2-350M."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("reinforce_ablation")

# Constants across all experiments
MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
EVAL_DATASET_PATH = "data/training/benchmark_val.jsonl"
OUTPUT_ROOT = Path("/Volumes/CodeCypher/models/experiments/reinforce-ablation")
SEED = 42

# Experiment configurations
# Each maps to kwargs for service.train_from_dataset()
EXPERIMENTS: dict[str, dict] = {
    "exp0": {
        "description": "Negative control: reproduce failure (auto_regime, default LR)",
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 1000,
        },
    },
    "exp1": {
        "description": "CE-only control: no REINFORCE, no entropy",
        "kwargs": {
            "auto_regime": False,
            "online_eval": True,
            "online_eval_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 65,
            "outcome_training": False,
            "entropy_regularization": False,
        },
    },
    "exp2": {
        "description": "LR/10: derived 1/(10*L) from Lipschitz measurement",
        "lr_divisor": 10,
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
        },
    },
    "exp3": {
        "description": "LR/100: derived 1/(100*L) from Lipschitz measurement",
        "lr_divisor": 100,
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
        },
    },
    "exp4": {
        "description": "Strong entropy floor (95% of baseline)",
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
            "entropy_floor_fraction": 0.95,
        },
    },
    # exp5 is a standalone script (reinforce_only_experiment.py)
    "exp6": {
        "description": "KL penalty from frozen reference",
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
            "kl_reference_penalty": True,
        },
    },
    "exp7": {
        "description": "Signal density gate (>50% required)",
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
            "outcome_signal_density_gate": 0.5,
        },
    },
    "exp8": {
        "description": "Lipschitz stabilization (10 batches)",
        "kwargs": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 10,
            "max_iters": 200,
            # lipschitz_batches removed — MASS uses spectral ceiling now
        },
    },
}


def run_experiment(exp_name: str) -> None:
    if exp_name not in EXPERIMENTS:
        log.error(
            "Unknown experiment: %s. Available: %s",
            exp_name, ", ".join(sorted(EXPERIMENTS.keys())),
        )
        sys.exit(1)

    exp = EXPERIMENTS[exp_name]
    start_ts = datetime.now(tz=timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("REINFORCE Ablation: %s", exp_name)
    log.info("Description: %s", exp["description"])
    log.info("Model: %s", MODEL_PATH)
    log.info("Seed: %d", SEED)
    log.info("Start: %s", start_ts)
    log.info("=" * 60)

    # Verify volume
    if not Path(MODEL_PATH).exists():
        log.error("Model path does not exist: %s", MODEL_PATH)
        sys.exit(1)

    # Create output directory
    output_dir = OUTPUT_ROOT / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter"
    metrics_path = output_dir / "metrics.jsonl"
    run_log_path = output_dir / "run_log.json"

    log.info("Output: %s", output_dir)

    # Import and create service
    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()

    kwargs = dict(exp["kwargs"])

    # If lr_divisor is set, measure Lipschitz first, then derive lr_override = 1/(divisor*L)
    lr_divisor = exp.get("lr_divisor")
    spectral_ceiling = None
    if lr_divisor is not None:
        log.info("Deriving spectral ceiling for LR with divisor=%d...", lr_divisor)
        from modelcypher.core.domain.dataset_loading import load_jsonl_dataset

        backend = service._backend
        adapter = service._adapter
        model_pre, tok_pre = backend.load_model(MODEL_PATH)
        train_samples = load_jsonl_dataset(Path(DATASET_PATH))

        # Inject NB-LoRA for measurement (Lipschitz is on the trainable model)
        from modelcypher.core.domain.training.geometric_lora import (
            analyze_weight_geometries,
            select_target_modules,
            apply_data_rank_ceiling,
            compute_coupled_ranks,
        )

        weights_pre = adapter.extract_weight_matrices(model_pre)
        geometries_pre = analyze_weight_geometries(weights_pre, backend)
        target_modules_pre = select_target_modules(geometries_pre)
        coupled_pre = compute_coupled_ranks(geometries_pre, target_modules_pre)
        ceiling_pre = apply_data_rank_ceiling(coupled_pre, n_samples=len(train_samples))
        adapter.inject_nb_lora(
            model_pre, geometries_pre, target_modules_pre,
            safety_margin=None, rank_overrides=ceiling_pre,
        )
        adapter.freeze_and_apply_lora(model_pre)
        sigma_max_pre = max(
            g.sigma_max for g in geometries_pre.values() if g.sigma_max > 0
        )

        # Derive spectral ceiling and apply lr_divisor
        sigma_k_min_pre = min(
            g.sigma_k for g in geometries_pre.values() if g.sigma_k > 0
        )
        spectral_ceiling = sigma_k_min_pre / sigma_max_pre
        derived_lr = spectral_ceiling / lr_divisor
        kwargs["lr_override"] = derived_lr
        log.info(
            "Spectral ceiling=%.4e (sigma_k_min=%.4e / sigma_max=%.4e), "
            "divisor=%d, derived lr_override=%.4e",
            spectral_ceiling, sigma_k_min_pre, sigma_max_pre, lr_divisor, derived_lr,
        )
        del model_pre, tok_pre  # Free memory before training run

    # Run
    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        eval_dataset_path=EVAL_DATASET_PATH,
        output_path=str(adapter_path),
        seed=SEED,
        **kwargs,
    )
    elapsed = time.monotonic() - t0

    # Extract online eval history
    online_eval_history = []
    if result.epoch_metrics:
        for i, m in enumerate(result.epoch_metrics):
            acc = m.get("online_eval_accuracy")
            if acc is not None:
                entry = {
                    "epoch": m.get("epoch", i),
                    "accuracy": acc,
                    "n_correct": m.get("online_eval_n_correct"),
                    "n_total": m.get("online_eval_n_total"),
                    "degraded": m.get("online_eval_degraded"),
                }
                online_eval_history.append(entry)
                log.info(
                    "  Epoch %s: %s/%s (%.1f%%)",
                    entry["epoch"], entry["n_correct"], entry["n_total"],
                    acc * 100,
                )

    # Write per-checkpoint metrics.jsonl
    with open(metrics_path, "w") as f:
        for m in (result.epoch_metrics or []):
            record = {"experiment": exp_name}
            record.update(m)
            f.write(json.dumps(record, default=str) + "\n")
    log.info("Metrics written to: %s", metrics_path)

    # Evaluate success criteria
    all_evals = [e for e in online_eval_history if e["n_correct"] is not None]
    no_degradation = all(e["n_correct"] >= 18 for e in all_evals) if all_evals else False
    final_improved = all_evals[-1]["n_correct"] > 18 if all_evals else False

    # Save run log
    run_log = {
        "experiment": exp_name,
        "description": exp["description"],
        "timestamp": start_ts,
        "model_path": MODEL_PATH,
        "seed": SEED,
        "kwargs_sent": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                        for k, v in kwargs.items()},
        "lr_divisor": lr_divisor,
        "spectral_ceiling": spectral_ceiling if lr_divisor > 1 else None,
        "elapsed_seconds": round(elapsed, 1),
        "train_iters": result.train_iters,
        "stop_reason": result.stop_reason,
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "baseline_loss": round(result.baseline_loss, 6),
        "post_loss": round(result.post_loss, 6),
        "online_eval_history": online_eval_history,
        "success_criteria": {
            "no_degradation_all_checkpoints": no_degradation,
            "final_improved_above_18": final_improved,
        },
        "n_epoch_metrics": len(result.epoch_metrics) if result.epoch_metrics else 0,
    }

    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    log.info("Run log saved to: %s", run_log_path)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY: %s", exp_name)
    log.info("  Elapsed: %.1f seconds", elapsed)
    log.info("  Stop reason: %s", result.stop_reason)
    log.info("  Train iters: %d", result.train_iters)
    log.info("  Loss: %.4f -> %.4f", result.initial_loss, result.final_loss)
    if all_evals:
        log.info("  Final eval: %d/%d", all_evals[-1]["n_correct"], all_evals[-1]["n_total"])
        log.info("  No degradation (all >= 18): %s", no_degradation)
        log.info("  Final improved (> 18): %s", final_improved)
        if no_degradation and final_improved:
            log.info("  >>> PASS <<<")
        elif no_degradation:
            log.info("  >>> PARTIAL PASS (no degradation, no improvement) <<<")
        else:
            log.info("  >>> FAIL (degradation detected) <<<")
    else:
        log.info("  No online eval results captured.")
    log.info("=" * 60)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/reinforce_ablation.py <exp_name>")
        print(f"Available: {', '.join(sorted(EXPERIMENTS.keys()))}")
        print("Note: exp5 uses scripts/reinforce_only_experiment.py")
        sys.exit(1)

    exp_name = sys.argv[1]
    run_experiment(exp_name)


if __name__ == "__main__":
    main()
