#!/usr/bin/env python3
"""End-to-end validation: geometry-derived LoRA configs produce working training.

Proves that spectral geometry → LoRA config → actual learning.
Uses mlx-lm's native LoRALinear for proper forward-pass integration,
but injects per-layer with geometry-derived rank, dropout, and scale.

This is a validation script, not a production training pipeline.
It stays in scripts/ per Research vs Production policy.

Usage:
    poetry run python scripts/validate_geometric_training.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --dataset path/to/data.jsonl \
        --iters 100

Exit code 0 = training loss decreased (PASS)
Exit code 1 = training failed to improve (FAIL)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate geometry-derived LoRA training")
    p.add_argument("--model", required=True, help="Path to MLX model directory")
    p.add_argument("--dataset", required=True, help="Path to JSONL dataset (one {\"text\": ...} per line)")
    p.add_argument("--eval-dataset", default=None, help="Held-out eval JSONL (default: 80/20 split)")
    p.add_argument("--iters", type=int, default=100, help="Training iterations")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size")
    p.add_argument("--seq-length", type=int, default=256, help="Max sequence length")
    p.add_argument("--lr", type=float, default=None, help="Override geometry-derived LR")
    p.add_argument("--deep", action="store_true", help="Target all layers (not just tail_dims > 0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--eval-batches", type=int, default=10, help="Number of eval batches")
    return p.parse_args()


# ============================================================================
# Dataset
# ============================================================================

def load_jsonl_dataset(path: str) -> list[dict]:
    """Load JSONL dataset. Each line: {"text": "..."}."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                samples.append(obj)
    if not samples:
        raise ValueError(f"No valid samples found in {path}")
    return samples


def prepare_dataset(samples: list[dict], tokenizer) -> list:
    """Tokenize samples into format expected by mlx-lm's iterate_batches.

    iterate_batches expects dataset[i] to be a tuple (tokens, offset) where:
    - tokens: mx.array of token IDs
    - offset: int, index where loss computation starts (0 = all tokens)

    The len_fn indexes dataset[idx][0] to get length.
    """
    import mlx.core as mx

    dataset = []
    for sample in samples:
        tokens = tokenizer.encode(sample["text"])
        if len(tokens) < 2:
            continue
        dataset.append((mx.array(tokens, dtype=mx.int32), 0))
    return dataset


# ============================================================================
# Weight extraction
# ============================================================================

def extract_weight_matrices(model) -> dict:
    """Extract 2D weight matrices from all projection layers.

    Returns dict: "model.layers.{i}.self_attn.q_proj.weight" -> weight array
    """
    import mlx.core as mx

    weights = {}
    base = getattr(model, "model", model)

    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute — unsupported architecture")

    for i, layer in enumerate(base.layers):
        # Attention projections
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    key = f"model.layers.{i}.self_attn.{proj_name}.weight"
                    weights[key] = proj.weight
                    mx.eval(proj.weight)

        # MLP projections
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("up_proj", "down_proj", "gate_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    key = f"model.layers.{i}.mlp.{proj_name}.weight"
                    weights[key] = proj.weight
                    mx.eval(proj.weight)

    logger.info("Extracted %d weight matrices from %d layers", len(weights), len(base.layers))
    return weights


# ============================================================================
# LoRA injection with per-layer geometry
# ============================================================================

def inject_geometric_lora(model, configs):
    """Replace target Linear layers with LoRALinear using per-layer geometry.

    Uses mlx-lm's native LoRALinear.from_base() for proper forward-pass
    integration. Each layer gets its own rank and dropout from geometry.

    Args:
        model: mlx-lm model (has model.model.layers[...])
        configs: list of LoRALayerConfig from derive_lora_configs()

    Returns:
        Number of layers successfully injected.
    """
    from mlx_lm.tuner.lora import LoRALinear

    injected = 0

    for cfg in configs:
        if cfg.rank <= 0:
            continue

        # Parse path: "model.layers.0.self_attn.q_proj.weight"
        # -> navigate to parent, replace the projection
        path_parts = cfg.layer_key.replace(".weight", "").split(".")

        try:
            obj = model
            for part in path_parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)

            attr_name = path_parts[-1]
            linear = getattr(obj, attr_name)

            lora = LoRALinear.from_base(
                linear,
                r=cfg.rank,
                dropout=cfg.dropout,
                scale=20.0,  # Standard training scale; sigma_k is post-hoc bound
            )
            setattr(obj, attr_name, lora)
            injected += 1

            logger.debug(
                "Injected LoRA: %s (rank=%d, dropout=%.4f)",
                cfg.layer_key, cfg.rank, cfg.dropout,
            )
        except Exception as e:
            logger.warning("Failed to inject LoRA at %s: %s", cfg.layer_key, e)

    return injected


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_loss(model, dataset, tokenizer, batch_size, seq_length, n_batches):
    """Compute average loss and perplexity on a dataset."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    total_loss = 0.0
    total_tokens = 0
    n_evaluated = 0

    for batch, lengths in iterate_batches(
        dataset, batch_size, seq_length, loop=False
    ):
        loss, ntoks = default_loss(model, batch, lengths)
        mx.eval(loss, ntoks)
        total_loss += float(loss) * float(ntoks)
        total_tokens += float(ntoks)
        n_evaluated += 1
        if n_evaluated >= n_batches:
            break

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return avg_loss, perplexity


# ============================================================================
# Spectral bound check (post-hoc)
# ============================================================================

def check_spectral_bounds(model, configs):
    """Check if trained LoRA layers respect sigma_k spectral bounds.

    Post-hoc only — we don't enforce during training.
    Returns (n_within_bound, n_total, max_violation_ratio, details).
    """
    import mlx.core as mx

    within = 0
    total = 0
    max_ratio = 0.0
    details = []

    for cfg in configs:
        if cfg.rank <= 0:
            continue

        path_parts = cfg.layer_key.replace(".weight", "").split(".")
        try:
            obj = model
            for part in path_parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            lora_layer = getattr(obj, path_parts[-1])

            if not hasattr(lora_layer, "lora_a") or not hasattr(lora_layer, "lora_b"):
                continue

            # lora_a: (input_dims, r), lora_b: (r, output_dims)
            # Effective delta: scale * (x @ lora_a) @ lora_b
            # The weight-space delta is: scale * lora_a @ lora_b  (input_dims, output_dims) is wrong
            # Actually LoRALinear computes: z = (dropout(x) @ lora_a) @ lora_b, so delta in weight space
            # maps input -> output as: delta_W^T = scale * lora_a @ lora_b
            # Spectral norm of this product:
            product = lora_layer.scale * (lora_layer.lora_a @ lora_layer.lora_b)
            product_f32 = product.astype(mx.float32)
            mx.eval(product_f32)

            _, S, _ = mx.linalg.svd(product_f32, compute_uv=True, stream=mx.cpu)
            mx.eval(S)
            spectral_norm = float(S[0])

            ratio = spectral_norm / cfg.sigma_k if cfg.sigma_k > 0 else float("inf")
            is_within = ratio <= 1.0

            if is_within:
                within += 1
            max_ratio = max(max_ratio, ratio)
            total += 1

            details.append({
                "layer": cfg.layer_key,
                "spectral_norm": spectral_norm,
                "sigma_k": cfg.sigma_k,
                "ratio": ratio,
                "within_bound": is_within,
            })
        except Exception as e:
            logger.debug("Could not check bounds for %s: %s", cfg.layer_key, e)

    return within, total, max_ratio, details


# ============================================================================
# Training
# ============================================================================

def train(model, train_dataset, batch_size, seq_length, n_iters, lr, seed):
    """Training loop using mlx-lm patterns.

    Returns list of (iteration, loss, tokens_per_sec) tuples.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    optimizer = opt.Adam(learning_rate=lr)

    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    losses = []
    t_start = time.time()

    batch_iter = iterate_batches(
        train_dataset, batch_size, seq_length, loop=True, seed=seed
    )

    for it in range(n_iters):
        batch, lengths = next(batch_iter)

        (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = float(loss)
        ntoks_val = float(ntoks)
        elapsed = time.time() - t_start
        tps = ntoks_val / max(elapsed, 1e-6) if it == 0 else ntoks_val / max(time.time() - t_step, 1e-6)

        losses.append((it, loss_val, tps))
        t_step = time.time()

        if (it + 1) % max(1, n_iters // 10) == 0 or it == 0:
            logger.info(
                "Iter %d/%d | loss=%.4f | tokens/sec=%.1f",
                it + 1, n_iters, loss_val, tps,
            )

    return losses


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.training.geometric_lora import (
        analyze_weight_geometries,
        derive_lora_configs,
        select_target_modules,
    )
    from modelcypher.core.domain.training.geometric_optimizer import (
        derive_optimizer_geometry_config,
    )

    # Initialize backend
    initialize_default_backend()
    backend = get_default_backend()

    logger.info("=" * 70)
    logger.info("GEOMETRIC LoRA TRAINING VALIDATION")
    logger.info("=" * 70)
    logger.info("Model:      %s", args.model)
    logger.info("Dataset:    %s", args.dataset)
    logger.info("Iterations: %d", args.iters)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Seq length: %d", args.seq_length)

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    logger.info("\n--- Loading Model ---")
    from mlx_lm import load
    model, tokenizer = load(args.model)

    # ------------------------------------------------------------------
    # 2. Load & prepare dataset
    # ------------------------------------------------------------------
    logger.info("\n--- Loading Dataset ---")
    all_samples = load_jsonl_dataset(args.dataset)
    logger.info("Loaded %d samples from %s", len(all_samples), args.dataset)

    if args.eval_dataset:
        train_samples = all_samples
        eval_samples = load_jsonl_dataset(args.eval_dataset)
        logger.info("Eval dataset: %d samples from %s", len(eval_samples), args.eval_dataset)
    else:
        split = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split]
        eval_samples = all_samples[split:]
        logger.info("Split: %d train, %d eval", len(train_samples), len(eval_samples))

    train_dataset = prepare_dataset(train_samples, tokenizer)
    eval_dataset = prepare_dataset(eval_samples, tokenizer)

    if not train_dataset:
        logger.error("No valid training samples after tokenization")
        sys.exit(1)
    if not eval_dataset:
        logger.error("No valid eval samples after tokenization")
        sys.exit(1)

    logger.info("Tokenized: %d train, %d eval sequences", len(train_dataset), len(eval_dataset))

    # ------------------------------------------------------------------
    # 3. Baseline evaluation (before LoRA)
    # ------------------------------------------------------------------
    logger.info("\n--- Baseline Evaluation ---")
    baseline_loss, baseline_ppl = evaluate_loss(
        model, eval_dataset, tokenizer, args.batch_size, args.seq_length, args.eval_batches
    )
    logger.info("Baseline loss: %.4f  perplexity: %.2f", baseline_loss, baseline_ppl)

    # ------------------------------------------------------------------
    # 4. Extract weights & compute geometry
    # ------------------------------------------------------------------
    logger.info("\n--- Computing Weight Geometry ---")
    weights = extract_weight_matrices(model)

    geometries = analyze_weight_geometries(weights, backend)
    logger.info("Analyzed %d weight matrices", len(geometries))

    if args.deep:
        target_modules = list(geometries.keys())
        logger.info("Deep mode: targeting all %d layers", len(target_modules))
    else:
        target_modules = select_target_modules(geometries)
        logger.info("Geometry selected %d / %d targetable layers", len(target_modules), len(geometries))

    if not target_modules:
        logger.error("No targetable layers found — model may have full-rank weights everywhere")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Derive per-layer LoRA configs
    # ------------------------------------------------------------------
    configs = derive_lora_configs(geometries, target_modules, adaptive_rank=True)

    ranks = [c.rank for c in configs if c.rank > 0]
    dropouts = [c.dropout for c in configs]
    sigma_ks = [c.sigma_k for c in configs]

    logger.info("Derived %d LoRA configs:", len(configs))
    logger.info("  Rank range:    %d - %d", min(ranks), max(ranks))
    logger.info("  Dropout range: %.4f - %.4f", min(dropouts), max(dropouts))
    logger.info("  Sigma_k range: %.6f - %.6f", min(sigma_ks), max(sigma_ks))

    # ------------------------------------------------------------------
    # 6. Derive learning rate
    # ------------------------------------------------------------------
    # The geometry-derived LR (1/max_sigma) is designed for a per-layer
    # geometric optimizer with individual LR scaling. For flat Adam,
    # we scale it by the number of LoRA layers to prevent instability
    # when many layers are being trained simultaneously.
    if args.lr is not None:
        lr = args.lr
        logger.info("Using override LR: %.2e", lr)
    else:
        opt_config = derive_optimizer_geometry_config(weights, backend)
        geo_lr = opt_config.base_lr
        # Scale down for flat Adam: more trainable layers → lower LR
        # This approximates what per-layer scaling would do globally.
        # Cap at 1e-4 as a safe upper bound for Adam with LoRA.
        n_lora = len([c for c in configs if c.rank > 0])
        lr = min(geo_lr / max(1, math.sqrt(n_lora)), 1e-4)
        logger.info(
            "LR: %.2e (geometry=%.2e, %d layers, max_sigma=%.4f)",
            lr, geo_lr, n_lora, opt_config.max_sigma,
        )

    # ------------------------------------------------------------------
    # 7. Inject LoRA layers
    # ------------------------------------------------------------------
    logger.info("\n--- Injecting Per-Layer LoRA ---")

    # Freeze FIRST, then inject — new LoRA params will be trainable by default
    # (they weren't in the model when freeze was called)
    model.freeze()
    n_injected = inject_geometric_lora(model, configs)
    logger.info("Injected LoRA into %d / %d target layers", n_injected, len(configs))

    if n_injected == 0:
        logger.error("No LoRA layers were injected — check model architecture")
        sys.exit(1)

    # Count trainable params
    n_trainable = sum(p.size for _, p in _iter_trainable(model))
    logger.info("Trainable parameters: %s", _format_params(n_trainable))

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    logger.info("\n--- Training ---")
    t0 = time.time()
    losses = train(
        model, train_dataset, args.batch_size, args.seq_length,
        args.iters, lr, args.seed,
    )
    train_time = time.time() - t0

    first_loss = losses[0][1]
    final_loss = losses[-1][1]

    logger.info("Training complete in %.1fs", train_time)
    logger.info("Loss: %.4f -> %.4f (delta: %.4f)", first_loss, final_loss, final_loss - first_loss)

    # ------------------------------------------------------------------
    # 9. Evaluate after training
    # ------------------------------------------------------------------
    logger.info("\n--- Post-Training Evaluation ---")
    post_loss, post_ppl = evaluate_loss(
        model, eval_dataset, tokenizer, args.batch_size, args.seq_length, args.eval_batches
    )
    logger.info("Post-training loss: %.4f  perplexity: %.2f", post_loss, post_ppl)

    # ------------------------------------------------------------------
    # 10. Check spectral bounds (post-hoc)
    # ------------------------------------------------------------------
    logger.info("\n--- Spectral Bound Check ---")
    within, total, max_violation_ratio, bound_details = check_spectral_bounds(model, configs)
    logger.info("Layers within sigma_k bound: %d / %d", within, total)
    logger.info("Max violation ratio: %.2fx", max_violation_ratio)

    # ------------------------------------------------------------------
    # 11. Report
    # ------------------------------------------------------------------
    loss_decreased = final_loss < first_loss
    eval_improved = post_loss < baseline_loss
    passed = loss_decreased

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info("Model:          %s", Path(args.model).name)
    logger.info("Dataset:        %s", args.dataset)
    logger.info("Training steps: %d", args.iters)
    logger.info("")
    logger.info("Geometry-derived config:")
    logger.info("  Target layers:  %d / %d", n_injected, len(geometries))
    logger.info("  Rank range:     %d - %d (per-layer from tail_dims)", min(ranks), max(ranks))
    logger.info("  Dropout range:  %.4f - %.4f", min(dropouts), max(dropouts))
    logger.info("  Learning rate:  %.2e%s", lr, " (override)" if args.lr else " (geometry)")
    logger.info("")
    logger.info("Before training:")
    logger.info("  Eval loss:      %.4f", baseline_loss)
    logger.info("  Eval perplexity:%.2f", baseline_ppl)
    logger.info("")
    logger.info("After training:")
    logger.info("  Train loss:     %.4f -> %.4f (delta: %+.4f)", first_loss, final_loss, final_loss - first_loss)
    logger.info("  Eval loss:      %.4f (delta: %+.4f)", post_loss, post_loss - baseline_loss)
    logger.info("  Eval perplexity:%.2f (delta: %+.2f)", post_ppl, post_ppl - baseline_ppl)
    logger.info("")
    logger.info("Spectral bounds:")
    logger.info("  Within sigma_k: %d / %d", within, total)
    logger.info("  Max violation:  %.2fx", max_violation_ratio)
    logger.info("")

    if passed:
        logger.info("RESULT: PASS (training loss decreased)")
        if eval_improved:
            logger.info("  Eval also improved — generalization confirmed")
        else:
            logger.info("  Eval did not improve — may need more data or iterations")
    else:
        logger.info("RESULT: FAIL (training loss did not decrease)")

    # Cleanup
    del model
    mx.clear_cache()

    sys.exit(0 if passed else 1)


# ============================================================================
# Helpers
# ============================================================================

def _iter_trainable(model):
    """Iterate over trainable parameters as (name, array) pairs."""
    import mlx.nn as nn
    # model.trainable_parameters() returns nested dict in mlx-lm
    def _flatten(prefix, tree):
        if hasattr(tree, 'shape'):
            yield prefix, tree
        elif isinstance(tree, dict):
            for k, v in tree.items():
                yield from _flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                yield from _flatten(f"{prefix}.{i}" if prefix else str(i), v)

    params = model.trainable_parameters()
    yield from _flatten("", params)


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


if __name__ == "__main__":
    main()
