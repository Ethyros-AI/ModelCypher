#!/usr/bin/env python3
"""REINFORCE re-run with MASS step size (√N-corrected).

This is the critical validation: does MASS + REINFORCE maintain or improve
the 350M baseline of 18/25 (72%)?

The ablation (2026-02-22) showed the Lipschitz LR was the root cause of
REINFORCE degradation. MASS replaces Lipschitz with Weyl ceiling + SPS +
Weyl displacement, corrected by √N for epoch budget distribution.

Usage:
    poetry run python scripts/mass_reinforce_run.py
"""

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
log = logging.getLogger("mass_reinforce")

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
EVAL_DATASET_PATH = "data/training/benchmark_val.jsonl"
OUTPUT_ROOT = Path("/Volumes/CodeCypher/models/experiments/mass-reinforce")


def main():
    if not Path(MODEL_PATH).exists():
        log.error("Model path does not exist: %s", MODEL_PATH)
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_ROOT / "run1"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter"
    metrics_path = output_dir / "metrics.jsonl"
    run_log_path = output_dir / "run_log.json"

    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()

    start_ts = datetime.now(timezone.utc).isoformat()
    log.info("=" * 60)
    log.info("MASS + REINFORCE VALIDATION RUN")
    log.info("Start: %s", start_ts)
    log.info("=" * 60)
    log.info("No lr_override — MASS derives step size from geometry")
    log.info("auto_regime=True — will select CE/REINFORCE/hybrid from baseline eval")

    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        eval_dataset_path=EVAL_DATASET_PATH,
        output_path=str(adapter_path),
        max_iters=1000,
        seed=42,
        auto_regime=True,
        regime_n_problems=25,
        eval_interval=46,  # ~1 epoch
    )
    elapsed = time.monotonic() - t0

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info("Stop reason: %s", result.stop_reason)
    log.info("Train iters: %d", result.train_iters)
    log.info("Elapsed: %.1f sec", elapsed)

    d = result.to_dict()

    # Write metrics
    epoch_metrics = d.get("epoch_metrics", [])
    with metrics_path.open("w") as f:
        for em in epoch_metrics:
            f.write(json.dumps(em) + "\n")

    # Extract key results
    online_eval_history = []
    for em in epoch_metrics:
        if em.get("online_eval_accuracy") is not None:
            online_eval_history.append({
                "epoch": em["epoch"],
                "accuracy": em["online_eval_accuracy"],
                "n_correct": em.get("online_eval_n_correct"),
                "n_total": em.get("online_eval_n_total"),
                "degraded": em.get("online_eval_degraded"),
            })

    # Print MASS metrics
    log.info("")
    log.info("MASS step sizes:")
    for em in epoch_metrics:
        eta_step = em.get("precond_eta_step", 0)
        eta_sps = em.get("eta_sps", 0)
        eta_weyl = em.get("eta_weyl", 0)
        eta_ceil = em.get("eta_ceiling", 0)
        log.info(
            "  Epoch %s: η_eff=%.4e η_sps=%.4e η_weyl=%.4e η_ceil=%.4e",
            em.get("epoch", "?"), eta_step, eta_sps, eta_weyl, eta_ceil,
        )

    # Print online eval
    log.info("")
    log.info("Online eval history:")
    for ev in online_eval_history:
        log.info(
            "  Epoch %s: %d/%d = %.0f%% %s",
            ev["epoch"], ev["n_correct"], ev["n_total"],
            ev["accuracy"] * 100,
            "(DEGRADED)" if ev.get("degraded") else "",
        )

    # Write run log
    run_log = {
        "experiment": "mass-reinforce-run1",
        "description": "MASS √N-corrected ceiling + auto_regime REINFORCE",
        "timestamp": start_ts,
        "model_path": MODEL_PATH,
        "seed": 42,
        "kwargs_sent": {
            "auto_regime": True,
            "regime_n_problems": 25,
            "eval_interval": 46,
            "max_iters": 1000,
            "lr_override": None,
        },
        "mass_used": True,
        "elapsed_seconds": elapsed,
        "train_iters": result.train_iters,
        "stop_reason": result.stop_reason,
        "initial_loss": d.get("initial_loss"),
        "final_loss": d.get("final_loss"),
        "baseline_loss": d.get("baseline_loss"),
        "post_loss": d.get("post_loss"),
        "online_eval_history": online_eval_history,
        "n_epoch_metrics": len(epoch_metrics),
        "success_criteria": {
            "no_degradation_all_checkpoints": all(
                not ev.get("degraded", False) for ev in online_eval_history
            ),
            "final_improved_above_18": (
                online_eval_history[-1]["n_correct"] > 18
                if online_eval_history else False
            ),
        },
    }
    with run_log_path.open("w") as f:
        json.dump(run_log, f, indent=2)
    log.info("")
    log.info("Run log: %s", run_log_path)
    log.info("Metrics: %s", metrics_path)

    # Verdict
    log.info("")
    log.info("=" * 60)
    baseline = 18
    if online_eval_history:
        final = online_eval_history[-1]
        if final["n_correct"] >= baseline:
            log.info("PASS: %d/%d (>= baseline %d/25)",
                     final["n_correct"], final["n_total"], baseline)
        else:
            log.warning("FAIL: %d/%d (< baseline %d/25, delta=%d)",
                        final["n_correct"], final["n_total"], baseline,
                        final["n_correct"] - baseline)
    else:
        log.warning("NO ONLINE EVAL DATA — cannot determine pass/fail")


if __name__ == "__main__":
    main()
