#!/usr/bin/env python3
"""MASS step size validation.

Run 10 training steps and report per-step eta_step, eta_sps, eta_weyl values
to verify MASS produces step sizes in the empirical sweet spot (~0.003-0.004).

Usage:
    poetry run python scripts/mass_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mass_validation")

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
EVAL_DATASET_PATH = "data/training/benchmark_val.jsonl"


def main():
    if not Path(MODEL_PATH).exists():
        log.error("Model path does not exist: %s", MODEL_PATH)
        sys.exit(1)

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()

    log.info("=" * 60)
    log.info("MASS STEP SIZE VALIDATION")
    log.info("=" * 60)

    # Run with enough iterations to complete one epoch + see MASS metrics
    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        eval_dataset_path=EVAL_DATASET_PATH,
        output_path="/tmp/mass-validation",
        max_iters=200,
        seed=42,
        auto_regime=False,
    )
    elapsed = time.monotonic() - t0

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info("Stop reason: %s", result.stop_reason)
    log.info("Train iters: %d", result.train_iters)
    log.info("Elapsed: %.1f sec", elapsed)
    log.info("")

    # Extract MASS metrics from epoch_metrics
    d = result.to_dict()
    epoch_metrics = d.get("epoch_metrics", [])
    if not epoch_metrics:
        log.warning("No epoch metrics found — training stopped before epoch boundary")
        log.info("Final JSON output:")
        # Print key fields
        for key in ["initial_loss", "final_loss", "post_loss", "stop_reason",
                     "train_iters", "n_trainable_params"]:
            if key in d:
                log.info("  %s: %s", key, d[key])
        return

    log.info("MASS per-epoch step sizes:")
    log.info("%-8s %-12s %-12s %-12s %-12s %-12s %-12s",
             "Epoch", "eta_step", "eta_sps", "eta_weyl", "eta_ceiling", "d_norm", "disp")
    log.info("-" * 80)

    for em in epoch_metrics:
        log.info("%-8s %-12.4e %-12.4e %-12.4e %-12.4e %-12.4f %-12.4e",
                 em.get("epoch", "?"),
                 em.get("eta_step", 0),
                 em.get("eta_sps", 0),
                 em.get("eta_weyl", 0),
                 em.get("eta_ceiling", 0),
                 em.get("d_norm", 0),
                 em.get("displacement", 0))

    log.info("")
    log.info("VALIDATION:")
    eta_steps = [em.get("eta_step", 0) for em in epoch_metrics if em.get("eta_step")]
    if eta_steps:
        mean_eta = sum(eta_steps) / len(eta_steps)
        log.info("  Mean eta_step: %.4e", mean_eta)
        log.info("  Range: [%.4e, %.4e]", min(eta_steps), max(eta_steps))
        log.info("  Target range: [3e-3, 4e-3] (from ablation exp3)")
        if 1e-4 <= mean_eta <= 1e-1:
            log.info("  STATUS: eta_step in plausible range")
        elif mean_eta < 1e-4:
            log.warning("  STATUS: eta_step too small — may undertrain")
        else:
            log.warning("  STATUS: eta_step too large — may overtrain")

    # Also report which layer binds most often
    bindings = {"sps": 0, "weyl": 0, "ceiling": 0}
    for em in epoch_metrics:
        eta_step = em.get("eta_step", float("inf"))
        eta_sps = em.get("eta_sps", float("inf"))
        eta_weyl = em.get("eta_weyl", float("inf"))
        eta_ceiling = em.get("eta_ceiling", float("inf"))
        min_val = min(eta_sps, eta_weyl, eta_ceiling)
        if abs(min_val - eta_sps) < 1e-12:
            bindings["sps"] += 1
        elif abs(min_val - eta_weyl) < 1e-12:
            bindings["weyl"] += 1
        else:
            bindings["ceiling"] += 1
    log.info("  Binding constraint: sps=%d, weyl=%d, ceiling=%d",
             bindings["sps"], bindings["weyl"], bindings["ceiling"])


if __name__ == "__main__":
    main()
