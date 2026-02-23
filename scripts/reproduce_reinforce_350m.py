#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Reproduction script: REINFORCE 14/20 claim on LFM2-350M.
#
# Uses auto_regime=True to exercise the new regime selector,
# which should detect baseline capability and select REINFORCE+entropy.
#
# Usage:
#   poetry run python scripts/reproduce_reinforce_350m.py
"""Reproduce REINFORCE 14/20 outcome-training result on LFM2-350M."""

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
log = logging.getLogger("reproduce_reinforce_350m")

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DATASET_PATH = "data/training/benchmark_train.jsonl"
EVAL_DATASET_PATH = "data/training/benchmark_val.jsonl"
OUTPUT_DIR = Path("/Volumes/CodeCypher/models/experiments/reinforce-350m-reproduce")
SEED = 42


def main() -> None:
    start_ts = datetime.now(tz=timezone.utc).isoformat()
    log.info("=== REINFORCE 350M Reproduction ===")
    log.info("Model: %s", MODEL_PATH)
    log.info("Training data: %s", DATASET_PATH)
    log.info("Eval data: %s", EVAL_DATASET_PATH)
    log.info("Seed: %d", SEED)
    log.info("Start: %s", start_ts)

    # Verify volume is mounted
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        log.error("Model path does not exist: %s", MODEL_PATH)
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_path = OUTPUT_DIR / "adapter"
    log_path = OUTPUT_DIR / "run_log.json"

    log.info("Output adapter: %s", adapter_path)
    log.info("Run log: %s", log_path)

    # Import and create service
    from modelcypher.cli.composition import get_dataset_training_service

    service = get_dataset_training_service()

    # Run with auto_regime=True (validates regime selector + runs REINFORCE)
    log.info("")
    log.info("--- Run: auto_regime=True ---")
    log.info("Regime selector will measure baseline and choose training mode.")
    log.info("")

    t0 = time.monotonic()
    result = service.train_from_dataset(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        eval_dataset_path=EVAL_DATASET_PATH,
        output_path=str(adapter_path),
        auto_regime=True,
        regime_n_problems=None,
        eval_interval=10,
        max_iters=1000,
        seed=SEED,
    )
    elapsed = time.monotonic() - t0

    # Extract results
    log.info("")
    log.info("=== RESULTS ===")
    log.info("Elapsed: %.1f seconds", elapsed)
    log.info("Train iters: %d", result.train_iters)
    log.info("Stop reason: %s", result.stop_reason)
    log.info("Initial loss: %.6f", result.initial_loss)
    log.info("Final loss: %.6f", result.final_loss)
    log.info("Baseline loss: %.6f", result.baseline_loss)
    log.info("Post loss: %.6f", result.post_loss)

    # Online eval results from epoch_metrics (dicts with to_dict() output)
    online_eval_final = None
    online_eval_history = []
    if result.epoch_metrics:
        for i, m in enumerate(result.epoch_metrics):
            acc = m.get("online_eval_accuracy")
            n_c = m.get("online_eval_n_correct")
            n_t = m.get("online_eval_n_total")
            if acc is not None:
                entry = {"epoch": m.get("epoch", i), "accuracy": acc, "n_correct": n_c, "n_total": n_t}
                online_eval_history.append(entry)
                log.info(
                    "  Epoch %s: online_eval %s/%s (%.1f%%)",
                    m.get("epoch", i), n_c, n_t, acc * 100,
                )

        # Last epoch with online eval
        if online_eval_history:
            online_eval_final = online_eval_history[-1]

    # Save run log
    run_log = {
        "timestamp": start_ts,
        "model_path": MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "eval_dataset_path": EVAL_DATASET_PATH,
        "seed": SEED,
        "auto_regime": True,
        "regime_n_problems": 25,
        "eval_interval": 10,
        "max_iters": 1000,
        "elapsed_seconds": round(elapsed, 1),
        "train_iters": result.train_iters,
        "stop_reason": result.stop_reason,
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "baseline_loss": round(result.baseline_loss, 6),
        "post_loss": round(result.post_loss, 6),
        "baseline_perplexity": round(result.baseline_perplexity, 4),
        "post_perplexity": round(result.post_perplexity, 4),
        "n_lora_layers": result.n_lora_layers,
        "n_trainable_params": result.n_trainable_params,
        "spectral_bounds_ok": result.spectral_bounds_ok,
        "max_spectral_ratio": round(result.max_spectral_ratio, 6),
        "adapter_saturation_median_ratio": result.adapter_saturation_median_ratio,
        "online_eval_final": online_eval_final,
        "online_eval_history": online_eval_history,
        "epoch_metrics": result.epoch_metrics,
        "adapter_path": str(adapter_path),
        "result_dict": result.to_dict(),
    }

    with open(log_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    log.info("")
    log.info("Run log saved to: %s", log_path)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    if online_eval_final:
        n_correct = online_eval_final["n_correct"]
        n_total = online_eval_final["n_total"]
        log.info(
            "Final online eval: %d/%d (%.1f%%) — target was 14/20 (70%%)",
            n_correct, n_total,
            (n_correct / n_total * 100) if n_total else 0,
        )
    else:
        log.info("No online eval results captured — check epoch_metrics structure.")
        log.info("Epoch metrics keys: %s", list(result.epoch_metrics[-1].keys()) if result.epoch_metrics else "none")

    log.info("Done.")


if __name__ == "__main__":
    main()
