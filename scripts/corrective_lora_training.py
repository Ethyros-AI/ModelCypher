#!/usr/bin/env python3
"""Experiment 3: Corrective LoRA Training.

Trains a LoRA adapter on a quantized base model with a distillation loss
that minimizes the divergence between quantized+LoRA outputs and full-precision
(bf16) outputs.

Both models are loaded simultaneously. For each batch:
1. Forward through bf16 model (no gradient)
2. Forward through quantized+LoRA model (with gradient)
3. Compute MSE on logits: ||q_logits - fp_logits||^2
4. Backpropagate through LoRA parameters only

This directly tests whether a low-rank adapter can correct the systematic
component of quantization error identified in Experiment 1.

Success criterion: CKA(quantized+adapter, bf16) > CKA(quantized, bf16)

Usage:
    poetry run python scripts/corrective_lora_training.py

    # Custom iteration count
    poetry run python scripts/corrective_lora_training.py --max-iters 200
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("corrective_lora")

# Default paths
DEFAULT_QUANTIZED = (
    "results/feasibility_map/20260225T160732Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-8bit-g64-affine"
)
DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_TRAIN = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"


def _collect_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    backend: Any,
    n_samples: int | None = None,
) -> dict[int, list]:
    """Collect per-layer mean-pooled activations for CKA measurement.

    Returns dict[layer_idx, list[Array[hidden_dim]]].
    """
    activations: dict[int, list] = {}
    samples = texts[:n_samples] if n_samples else texts

    for text in samples:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx, act in acts.items():
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(pooled)

    return activations


def _compute_cka(
    acts_a: dict[int, list],
    acts_b: dict[int, list],
    backend: Any,
) -> dict[str, Any]:
    """Compute linear CKA between two sets of per-layer activations."""
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )

    per_layer: dict[int, float] = {}
    common_layers = sorted(set(acts_a.keys()) & set(acts_b.keys()))

    for layer_idx in common_layers:
        # Stack list of [hidden_dim] arrays into [n_samples, hidden_dim]
        mat_a = mx.stack(acts_a[layer_idx])
        mat_b = mx.stack(acts_b[layer_idx])
        mx.eval(mat_a, mat_b)

        cka = compute_linear_cka_from_activations(mat_a, mat_b, backend)
        per_layer[layer_idx] = float(cka)

    values = list(per_layer.values())
    return {
        "min_cka": min(values) if values else 0.0,
        "mean_cka": sum(values) / len(values) if values else 0.0,
        "per_layer_cka": per_layer,
        "n_layers": len(per_layer),
    }


def _prepare_batches(
    dataset_path: str,
    tokenizer: Any,
    batch_size: int,
    seq_length: int,
) -> tuple[list[dict], list[str]]:
    """Load dataset and prepare for training."""
    samples = []
    texts = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        samples.append(data)
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

    # Tokenize all samples
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 2:
            tokenized.append(tokens)

    return tokenized, texts


def main():
    args = _parse_args()

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.training.geometric_lora import (
        select_target_modules,
    )
    from modelcypher.core.domain.training.mass_step_size import (
        compute_per_step_rates,
        derive_spectral_ceiling,
    )

    backend = get_default_backend()
    adapter = MLXTrainingAdapter(backend)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Corrective LoRA Training — run_id=%s", run_id)
    logger.info("Quantized model: %s", args.quantized_model)
    logger.info("FP model: %s", args.fp_model)
    logger.info("Output: %s", output_dir)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "corrective_lora_training",
        "config": {
            "quantized_model": args.quantized_model,
            "fp_model": args.fp_model,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_cka_probes": args.n_cka_probes,
        },
    }

    # ── Phase 1: Load models ──
    logger.info("Loading bf16 reference model...")
    fp_model, tokenizer = backend.load_model(str(args.fp_model))

    logger.info("Loading quantized model...")
    q_model, _ = backend.load_model(str(args.quantized_model))

    # ── Phase 2: Collect pre-training CKA baseline ──
    logger.info("Collecting CKA baseline activations (%d probes)...", args.n_cka_probes)

    eval_texts = []
    with open(args.eval_dataset, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        eval_texts.append(text)
                except json.JSONDecodeError:
                    continue

    fp_activations = _collect_activations(
        fp_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
    )
    q_activations_before = _collect_activations(
        q_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
    )
    cka_before = _compute_cka(fp_activations, q_activations_before, backend)
    logger.info(
        "CKA before correction: min=%.4f, mean=%.4f (%d layers)",
        cka_before["min_cka"], cka_before["mean_cka"], cka_before["n_layers"],
    )
    results["cka_before"] = cka_before
    del q_activations_before
    gc.collect()

    # ── Phase 3: Inject LoRA on quantized model ──
    logger.info("Analyzing quantized model geometry...")
    geometries = adapter.analyze_model_geometry_streaming(q_model, use_randomized=True)
    target_modules = select_target_modules(geometries)
    logger.info("Targeting %d modules for corrective LoRA", len(target_modules))

    n_injected = adapter.inject_nb_lora(q_model, geometries, target_modules)
    adapter.freeze_and_apply_lora(q_model)
    logger.info("Injected %d NB-LoRA layers (geometric scale bounds)", n_injected)

    n_trainable = sum(
        p.size for _, p in mlx.utils.tree_flatten(q_model.trainable_parameters())
    )
    logger.info("Trainable parameters: %d", n_trainable)

    # ── Phase 4: Set up corrective training ──
    # MASS spectral ceiling
    sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)
    sigma_k_vals = [g.sigma_k for g in geometries.values() if g.sigma_k > 0]
    sigma_k_min = min(sigma_k_vals)
    eta_ceiling = derive_spectral_ceiling(
        sigma_k_min=sigma_k_min, sigma_max_global=sigma_max,
    )
    logger.info(
        "MASS: sigma_max=%.4e, sigma_k_min=%.4e, eta_ceiling=%.4e",
        sigma_max, sigma_k_min, eta_ceiling,
    )

    # Apply sqrt(N) correction
    tokenized, train_texts = _prepare_batches(
        args.train_dataset, tokenizer, args.batch_size, 512,
    )
    n_batches_per_epoch = max(1, len(tokenized) // args.batch_size)
    if n_batches_per_epoch > 1:
        eta_ceiling = eta_ceiling / math.sqrt(n_batches_per_epoch)
        logger.info("MASS sqrt(N) correction: eta=%.4e (N=%d)", eta_ceiling, n_batches_per_epoch)

    optimizer = opt.SGD(learning_rate=eta_ceiling, momentum=0.0)

    # Define corrective loss: MSE on logits between quantized+LoRA and bf16
    def corrective_loss_fn(q_model, batch):
        """MSE distillation loss: match bf16 logits."""
        q_logits = q_model(batch)[:, :-1, :]  # [batch, seq-1, vocab]

        # bf16 reference (no gradient flow)
        fp_logits = mx.stop_gradient(fp_model(batch)[:, :-1, :])

        # MSE on logits
        diff = q_logits - fp_logits
        mse = mx.mean(diff * diff)

        ntoks = batch.shape[0] * (batch.shape[1] - 1)
        return mse, mx.array(float(ntoks))

    loss_and_grad = nn.value_and_grad(q_model, corrective_loss_fn)

    # ── Phase 5: Training loop ──
    logger.info("Starting corrective training: %d iterations, batch_size=%d", args.max_iters, args.batch_size)

    backend.random_seed(args.seed)
    seq_length = 256  # Short sequences for efficiency

    losses: list[tuple[int, float]] = []
    start_time = time.monotonic()

    for it in range(args.max_iters):
        # Create batch: randomly sample and pad/truncate
        indices = [i % len(tokenized) for i in range(it * args.batch_size, (it + 1) * args.batch_size)]
        batch_tokens = [tokenized[idx] for idx in indices]

        # Pad/truncate to seq_length
        padded = []
        for tokens in batch_tokens:
            if len(tokens) > seq_length:
                padded.append(tokens[:seq_length])
            else:
                padded.append(tokens + [0] * (seq_length - len(tokens)))
        batch = mx.array(padded)

        # Forward + backward
        (loss, ntoks), grad = loss_and_grad(q_model, batch)

        # MASS per-step rate
        flat_grads = [p.reshape(-1) for _, p in mlx.utils.tree_flatten(grad) if p.size > 0]
        d_norm_sq = sum(mx.sum(p * p) for p in flat_grads)
        mx.eval(d_norm_sq, loss)
        d_norm = float(mx.sqrt(d_norm_sq).item())
        loss_val = float(loss.item())

        eta_step, _, _, _, _ = compute_per_step_rates(
            loss_val, d_norm, sigma_k_min, eta_ceiling,
        )
        optimizer.learning_rate = mx.array(eta_step)

        # Apply update
        optimizer.update(q_model, grad)
        mx.eval(q_model.parameters(), optimizer.state)

        # Clamp scales (THE constraint)
        for _, module in adapter._iter_nb_lora_modules(q_model):
            module.clamp_scale()
            mx.eval(module.S_raw)

        losses.append((it, loss_val))

        if it % 10 == 0 or it == args.max_iters - 1:
            elapsed = time.monotonic() - start_time
            logger.info(
                "iter %d/%d: loss=%.6f, eta=%.4e, d_norm=%.4e, elapsed=%.1fs",
                it, args.max_iters, loss_val, eta_step, d_norm, elapsed,
            )

    training_time = time.monotonic() - start_time
    logger.info("Training complete: %.1fs, final_loss=%.6f", training_time, losses[-1][1] if losses else 0.0)

    results["training"] = {
        "n_iters": len(losses),
        "initial_loss": losses[0][1] if losses else 0.0,
        "final_loss": losses[-1][1] if losses else 0.0,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "n_lora_layers": n_injected,
        "eta_ceiling": eta_ceiling,
        "sigma_max": sigma_max,
        "sigma_k_min": sigma_k_min,
    }

    # ── Phase 6: Post-training CKA measurement ──
    logger.info("Collecting post-training CKA...")
    q_activations_after = _collect_activations(
        q_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
    )
    cka_after = _compute_cka(fp_activations, q_activations_after, backend)
    logger.info(
        "CKA after correction: min=%.4f, mean=%.4f (%d layers)",
        cka_after["min_cka"], cka_after["mean_cka"], cka_after["n_layers"],
    )
    results["cka_after"] = cka_after

    # ── Phase 7: Save adapter ──
    adapter_path = output_dir / "corrective_adapter"
    adapter.save_adapter(q_model, adapter_path)
    logger.info("Adapter saved to %s", adapter_path)
    results["adapter_path"] = str(adapter_path)

    # ── Phase 8: Summary ──
    cka_improvement = cka_after["mean_cka"] - cka_before["mean_cka"]
    results["summary"] = {
        "cka_mean_before": cka_before["mean_cka"],
        "cka_mean_after": cka_after["mean_cka"],
        "cka_improvement": cka_improvement,
        "cka_min_before": cka_before["min_cka"],
        "cka_min_after": cka_after["min_cka"],
    }

    if cka_improvement > 0:
        verdict = (
            f"SUCCESS: CKA improved by {cka_improvement:.4f} "
            f"({cka_before['mean_cka']:.4f} → {cka_after['mean_cka']:.4f}). "
            "Corrective LoRA training reduces quantization gap."
        )
    elif abs(cka_improvement) < 0.001:
        verdict = (
            f"INCONCLUSIVE: CKA change {cka_improvement:+.4f} is within noise. "
            "Corrective training had negligible effect."
        )
    else:
        verdict = (
            f"DEGRADATION: CKA decreased by {abs(cka_improvement):.4f}. "
            "Corrective training hurt alignment."
        )

    results["verdict"] = verdict

    # Write results
    output_path = output_dir / "corrective_lora.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", output_path)

    # Pretty print
    print("\n" + "=" * 72)
    print("CORRECTIVE LORA TRAINING — SUMMARY")
    print("=" * 72)
    print(f"  Training iterations:          {len(losses)}")
    print(f"  Initial loss:                 {losses[0][1]:.6f}" if losses else "  No training")
    print(f"  Final loss:                   {losses[-1][1]:.6f}" if losses else "")
    print(f"  Trainable params:             {n_trainable:,}")
    print(f"  LoRA layers:                  {n_injected}")
    print()
    print(f"  CKA before (mean):            {cka_before['mean_cka']:.4f}")
    print(f"  CKA after (mean):             {cka_after['mean_cka']:.4f}")
    print(f"  CKA improvement:              {cka_improvement:+.4f}")
    print()
    print(f"  CKA before (min):             {cka_before['min_cka']:.4f}")
    print(f"  CKA after (min):              {cka_after['min_cka']:.4f}")
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Corrective LoRA Training",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED,
        help="Path to 8-bit quantized model",
    )
    parser.add_argument(
        "--fp-model",
        default=DEFAULT_FP,
        help="Path to full-precision (bf16) model",
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
        default="results/corrective_lora_training",
        help="Base output directory",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (small: 2 models in memory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--n-cka-probes",
        type=int,
        default=30,
        help="Number of probe samples for CKA measurement",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
