# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""B0 training harness for LKM validation protocol.

Baseline arm: standard LoRA training via mlx_lm with paper-matched config.
No geometric interventions from the ModelCypher pipeline.

Usage:
    poetry run python scripts/lkm/run_b0.py \\
        --model /path/to/model \\
        --data data/lkm/phonebook_4000tok.jsonl \\
        --r-cap 16 \\
        --tokens 4000 \\
        --output-base results/lora_memory_capacity_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_b0_config(
    r_cap: int,
    batch_size: int = 8,
    iters: int = 1500,
    learning_rate: float = 5e-4,
) -> dict:
    """Build B0 training config dict with paper-matched LoRA parameters.

    The paper uses alpha=r, and mlx_lm scale = alpha/rank = r/r = 1.0.
    This is an invariant: scale=1.0 regardless of rank.

    Args:
        r_cap: LoRA rank.
        batch_size: Training batch size (default: 8).
        iters: Training iterations (default: 1500).
        learning_rate: AdamW learning rate (default: 5e-4).

    Returns:
        Config dict with keys: batch_size, iters, learning_rate, lora_parameters.
    """
    return {
        "batch_size": batch_size,
        "iters": iters,
        "learning_rate": learning_rate,
        "lora_parameters": {
            "rank": r_cap,
            "scale": 1.0,  # alpha/rank = r/r = 1.0, always
            "dropout": 0.0,
        },
    }


def make_run_id(arm: str, r_cap: int, tokens: int) -> str:
    """Build a run identifier string.

    Args:
        arm: Experiment arm name (e.g. "B0").
        r_cap: LoRA rank.
        tokens: Token size label.

    Returns:
        Run ID string in format "{arm}_r{r_cap}_{tokens}tok".
    """
    return f"{arm}_r{r_cap}_{tokens}tok"


def train_b0(
    model_path: str,
    data_path: str,
    r_cap: int,
    output_dir: str,
    config_overrides: dict | None = None,
) -> Path:
    """Run B0 baseline LoRA training via mlx_lm.

    Loads model, applies LoRA, trains on JSONL data, and saves adapter
    weights + config files.

    Args:
        model_path: Path to the base model directory.
        data_path: Path to training JSONL file (each line: {"text": "..."}).
        r_cap: LoRA rank.
        output_dir: Directory to save outputs.
        config_overrides: Optional dict to override default config values
            (batch_size, iters, learning_rate).

    Returns:
        Path to the output directory.
    """
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.datasets import CacheDataset, TextDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    # Build config
    overrides = config_overrides or {}
    config = make_b0_config(
        r_cap=r_cap,
        batch_size=overrides.get("batch_size", 8),
        iters=overrides.get("iters", 1500),
        learning_rate=overrides.get("learning_rate", 5e-4),
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    adapter_file = str(out_path / "adapters.safetensors")

    # Load model and tokenizer
    model, tokenizer = mlx_load(model_path)

    # Freeze all parameters, then apply LoRA
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=-1,
        config=config["lora_parameters"],
    )

    # Load training data from JSONL
    with open(data_path) as f:
        all_data = [json.loads(line) for line in f]

    # Split: last 10% as validation, minimum batch_size items.
    # If dataset too small for a proper split, reuse training data as val
    # (val loss is informational; exact match is the real metric).
    batch_size = config["batch_size"]
    n_val = max(batch_size, len(all_data) // 10)
    if len(all_data) > n_val + batch_size:
        train_data = all_data[: len(all_data) - n_val]
        val_data = all_data[len(all_data) - n_val :]
    else:
        train_data = all_data
        val_data = all_data  # reuse for small datasets

    train_set = TextDataset(train_data, tokenizer)
    val_set = TextDataset(val_data, tokenizer)

    # Build optimizer and training args
    optimizer = optim.AdamW(learning_rate=config["learning_rate"])
    args = TrainingArgs(
        batch_size=config["batch_size"],
        iters=config["iters"],
        val_batches=5,
        steps_per_report=100,
        steps_per_eval=500,
        steps_per_save=500,
        max_seq_length=2048,
        adapter_file=adapter_file,
    )

    # Train
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(val_set),
        args=args,
    )

    # Save config.json (full run config)
    run_config = {
        "arm": "B0",
        "model": model_path,
        "data": data_path,
        "r_cap": r_cap,
        **config,
    }
    config_path = out_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # Save adapter_config.json (mlx_lm load_adapters compatibility)
    adapter_config = {
        "num_layers": -1,
        "lora_parameters": config["lora_parameters"],
    }
    adapter_config_path = out_path / "adapter_config.json"
    with open(adapter_config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)

    return out_path


def main() -> None:
    """CLI entry point for B0 baseline training."""
    parser = argparse.ArgumentParser(
        description="Run B0 baseline LoRA training for LKM validation."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to base model directory.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training JSONL file.",
    )
    parser.add_argument(
        "--r-cap",
        type=int,
        required=True,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="Token size label for run ID.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/lora_memory_capacity_validation",
        help="Base results directory (default: results/lora_memory_capacity_validation).",
    )

    args = parser.parse_args()

    run_id = make_run_id("B0", args.r_cap, args.tokens)
    output_dir = str(Path(args.output_base) / run_id)

    print(f"Run ID: {run_id}")
    print(f"Output: {output_dir}")
    print(f"Model:  {args.model}")
    print(f"Data:   {args.data}")
    print(f"Rank:   {args.r_cap}")

    train_b0(
        model_path=args.model,
        data_path=args.data,
        r_cap=args.r_cap,
        output_dir=output_dir,
    )

    print(f"B0 training complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
