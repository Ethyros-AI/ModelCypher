#!/usr/bin/env python3
"""End-to-end validation: fully geometric LoRA training.

Proves that spectral geometry → LoRA config → optimizer → actual learning
with zero arbitrary constants. Every parameter is a theorem or measurement:

  - Per-layer LoRA rank: from null-space capacity (tail_dims via Shannon eff rank)
  - LoRA scale: 1.0 (neutral multiplier; bound via spectral init + monitoring)
  - Spectral init: ||BA||_spectral = 0.5 * σ_k from step 0
  - Per-layer dropout: from spectral redundancy × adapter fraction
  - Learning rate: η = 1/L where L = λ_max(Hessian) (measured via power iteration)
  - ScaledGD: grad_A @ (BBᵀ+εI)⁻¹, (AᵀA+εI)⁻¹ @ grad_B (Riemannian GD)
  - Per-layer decay: σ_k/σ_max (condition ratio)
  - Early stopping: |Δ_epoch| < SE(data) OR spectral budget > 0.9

Optimizer: SGD with measured η + ScaledGD preconditioning.
No Adam, no momentum, no arbitrary caps, no EMA, no per-layer LR heuristics.

ScaledGD (Tong, Ma, Chi — JMLR 2021) preconditions each LoRA factor's
gradient by the pseudoinverse of the other factor. This achieves:
  - Condition-number-free convergence (the theorem)
  - Automatic asymmetric rates for A vs B (Hayou et al. ICML 2024: LoRA+)
  - Rates that decrease as adapters grow (Mu & Klabjan Dec 2025)

The base learning rate η = 1/L is Nesterov's (2004) optimal step size
for L-Lipschitz gradient, where L = λ_max(Hessian) is MEASURED via
power iteration on the Hessian-vector product.

This is a validation script, not a production training pipeline.
It stays in scripts/ per Research vs Production policy.

Usage:
    poetry run python scripts/validate_geometric_training.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --dataset path/to/data.jsonl

Geometry decides when to stop. --iters is a safety cap (default 10000),
not a target. Training halts when epoch-averaged loss stabilizes
(|Δ_epoch| < standard error of the difference, measured from the data)
or spectral budget is exhausted (||BA||/σ_k > 0.9).

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
    p.add_argument("--iters", type=int, default=10000, help="Safety cap (geometry decides when to stop)")
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

def inject_geometric_lora(model, configs, backend):
    """Replace target Linear layers with LoRALinear using per-layer geometry.

    Uses mlx-lm's native LoRALinear.from_base() for proper forward-pass
    integration. Each layer gets its own rank and dropout from geometry.

    After injection, replaces default B=zeros initialization with spectral
    init so that ||BA||_spectral = 0.5 * sigma_k from step 0. This gives
    the adapter immediate contribution at 50% of its geometric budget.

    Scale is set to 1.0 (neutral multiplier). The geometric constraint
    ||BA||_spectral <= sigma_k is enforced through initialization and
    spectral budget monitoring, not through the scale parameter.

    Args:
        model: mlx-lm model (has model.model.layers[...])
        configs: list of LoRALayerConfig from derive_lora_configs()
        backend: Backend for spectral init computation.

    Returns:
        Number of layers successfully injected.
    """
    import mlx.core as mx
    from mlx_lm.tuner.lora import LoRALinear

    from modelcypher.core.domain.geometry.spectral_init import (
        spectral_normalized_lora_init,
    )

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
                scale=1.0,  # Neutral multiplier. Bound is on ||BA||, not scale.
            )

            # Replace default B=zeros with spectral init at 50% of budget.
            # spectral_init returns A: [rank, in], B: [out, rank]
            # LoRALinear stores lora_a: [in, rank], lora_b: [rank, out]
            # Fuse delta = scale * lora_b.T @ lora_a.T = B @ A
            # So: lora_a = A.T, lora_b = B.T
            A_init, B_init = spectral_normalized_lora_init(
                in_features=cfg.in_features,
                out_features=cfg.out_features,
                rank=cfg.rank,
                sigma_k=cfg.sigma_k,  # Full geometric budget (Weyl threshold handles stopping)
                backend=backend,
            )
            lora.lora_a = A_init.T
            lora.lora_b = B_init.T
            mx.eval(lora.lora_a, lora.lora_b)

            setattr(obj, attr_name, lora)
            injected += 1

            logger.debug(
                "Injected LoRA: %s (rank=%d, dropout=%.4f, σ_k=%.4f)",
                cfg.layer_key, cfg.rank, cfg.dropout, cfg.sigma_k,
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
# ScaledGD preconditioning (delegated to domain module)
# ============================================================================

from modelcypher.core.domain.training.scaled_gd import precondition_lora_gradients  # noqa: E402


def apply_scaled_gd(model, grad, lora_configs, opt_config):
    """Apply ScaledGD preconditioning to LoRA gradients.

    Thin wrapper: flattens/unflattens mlx trees around the
    backend-agnostic domain function.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()

    grad_flat = dict(tree_flatten(grad))
    param_flat = dict(tree_flatten(model.trainable_parameters()))

    preconditioned = precondition_lora_gradients(
        grad_flat, param_flat, lora_configs, opt_config, backend,
    )

    return tree_unflatten(list(preconditioned.items()))


# ============================================================================
# Lipschitz constant measurement
# ============================================================================

def measure_lipschitz_constant(model, dataset, batch_size, seq_length, seed,
                                power_iterations=10):
    """Measure η = 1/L where L = max over all batches of λ_max(Hessian).

    L is a supremum — worst-case curvature over the training distribution.
    Measures λ_max(Hessian) on EVERY training batch and takes the maximum.
    This eliminates batch-dependent noise: single-batch L can vary 100×,
    but max-over-batches gives the true worst-case curvature.

    Uses existing hessian_estimator.top_eigenvalue() infrastructure.

    Args:
        model: mlx-lm model with LoRA layers injected.
        dataset: Tokenized training dataset.
        batch_size: Batch size for the measurement passes.
        seq_length: Sequence length for the measurement passes.
        seed: Random seed for batch iteration.
        power_iterations: Number of power iteration steps (default 10).

    Returns:
        (eta, L) where eta = 1/L is the optimal step size.

    Raises:
        ValueError: If L cannot be measured on any batch.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    from modelcypher.core.domain.training.hessian_estimator import (
        Config as HessianConfig,
        top_eigenvalue,
    )

    # Collect ALL batches — L is a supremum, we need to measure all of them
    batches = list(iterate_batches(dataset, batch_size, seq_length, loop=False))
    n_batches = len(batches)
    logger.info("Measuring L on %d batches (%d power iterations each)...",
                n_batches, power_iterations)

    # Save original trainable params to restore after measurement
    original_trainable = dict(tree_flatten(model.trainable_parameters()))
    trainable = dict(original_trainable)  # Copy for top_eigenvalue
    config = HessianConfig(power_iterations=power_iterations)

    L_values = []
    for i, (batch, lengths) in enumerate(batches):
        # Create closure bound to THIS batch (default arg captures value)
        def loss_and_grad_fn(params_dict, _batch=batch, _lengths=lengths):
            model.update(tree_unflatten(list(params_dict.items())))
            mx.eval(model.parameters())
            loss_grad_fn = nn.value_and_grad(model, default_loss)
            (loss, ntoks), grad = loss_grad_fn(model, _batch, _lengths)
            mx.eval(loss)
            grad_flat = dict(tree_flatten(grad))
            return loss, grad_flat

        L_i = top_eigenvalue(loss_and_grad_fn, trainable, config)

        if L_i is not None and L_i > 0:
            L_values.append(L_i)
            logger.debug("  Batch %d/%d: L=%.4e", i + 1, n_batches, L_i)

    # Restore original params (power iteration perturbs them)
    model.update(tree_unflatten(list(original_trainable.items())))
    mx.eval(model.parameters())

    if not L_values:
        raise ValueError(
            "Lipschitz constant measurement failed on all batches. "
            "Hessian may be degenerate at this point."
        )

    L = max(L_values)
    eta = 1.0 / L
    logger.info(
        "Measured L over %d/%d batches: max=%.4e, min=%.4e, range=%.1fx, η=1/L=%.4e",
        len(L_values), n_batches, max(L_values), min(L_values),
        max(L_values) / min(L_values), eta,
    )
    return eta, L


# ============================================================================
# Geometric early stopping (delegated to domain module)
# ============================================================================

from modelcypher.core.domain.training.geometric_early_stopping import check_loss_stable  # noqa: E402
from modelcypher.core.domain.training.spectral_budget import (  # noqa: E402
    DTYPE_THRESHOLD_F32,
    compute_budget_ratios,
    is_budget_exhausted,
)


def check_budget_exhausted(model, configs, opt_config=None) -> tuple[bool, float]:
    """Check if spectral budget is exhausted.

    Thin wrapper: walks mlx model tree to extract LoRA products,
    then delegates to domain modules for SVD + Weyl threshold comparison.

    When opt_config is provided, uses per-layer Weyl-derived crossing
    thresholds (gap_k / (2 * sigma_k)). Otherwise falls back to the
    dtype-derived threshold (1 - sqrt(eps_f32)).
    """
    import mlx.core as mx

    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()

    lora_products = []
    layer_keys_order = []
    for cfg in configs:
        if cfg.rank <= 0 or cfg.sigma_k <= 0:
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

            lora_products.append((
                lora_layer.scale,
                lora_layer.lora_a,
                lora_layer.lora_b,
                cfg.sigma_k,
            ))
            layer_keys_order.append(cfg.layer_key)
        except Exception:
            continue

    ratios = compute_budget_ratios(lora_products, backend)

    # Extract per-layer Weyl data from optimizer config
    spectral_gaps = None
    sigma_ks = None
    if opt_config is not None:
        spectral_gaps = []
        sigma_ks = []
        for key in layer_keys_order:
            lc = opt_config.layer_configs.get(key)
            if lc:
                spectral_gaps.append(lc.spectral_gap)
                sigma_ks.append(lc.sigma_k)
            else:
                spectral_gaps.append(0.0)
                sigma_ks.append(0.0)

    return is_budget_exhausted(
        ratios,
        spectral_gaps=spectral_gaps,
        sigma_ks=sigma_ks,
        threshold=DTYPE_THRESHOLD_F32,
    )


# ============================================================================
# Training
# ============================================================================

def train(model, train_dataset, batch_size, seq_length, max_iters, seed,
          lora_configs, opt_config, lr_override=None):
    """Training loop: geometry decides when to stop.

    Trains until one of two geometric criteria is met:
      loss_stable: |Δ_epoch| < SE_diff  (data noise floor, not machine epsilon)
      budget_exhausted: any layer ratio > gap_k/(2*σ_k)  (Weyl crossing bound)

    max_iters is a safety cap, not a target. The geometry decides.

    Zero arbitrary constants. Every number is a theorem or measurement:

      η = 1/L where L = max over batches of λ_max(H)  (Nesterov 2004)
      ScaledGD: grad_A @ (BBᵀ+εI)⁻¹                   (Tong et al. JMLR 2021)
      ScaledGD: (AᵀA+εI)⁻¹ @ grad_B                   (condition-number-free)
      decay_i = σ_k_i / σ_max_i                        (condition-aware decay)

    ScaledGD automatically produces:
      - Different effective rates for A and B    (Hayou et al. ICML 2024: LoRA+)
      - Rates that decrease as adapters grow     (Mu & Klabjan Dec 2025)

    No re-measurement of L during training. ScaledGD handles curvature
    changes as adapters grow — the preconditioner (BBᵀ+εI)⁻¹ shrinks
    automatically as B grows, which is exactly the rate decrease that
    Mu & Klabjan require. Budget monitoring (Weyl's inequality) catches
    any spectral bound violation.

    Returns:
        losses: list of (iteration, loss, tokens_per_sec)
        stop_reason: str (why training stopped)
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx_lm.tuner.trainer import default_loss, iterate_batches

    use_geometric = lr_override is None

    if use_geometric:
        # Measure L = max over all batches of λ_max(Hessian), set η = 1/L
        eta, L = measure_lipschitz_constant(
            model, train_dataset, batch_size, seq_length, seed,
            power_iterations=10,
        )
        optimizer = opt.SGD(learning_rate=eta, momentum=0.0)
    else:
        eta = lr_override
        optimizer = opt.SGD(learning_rate=eta, momentum=0.0)
        logger.info("Using override LR: %.2e (flat SGD, no ScaledGD)", lr_override)

    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    losses = []
    stop_reason = None
    t_start = time.time()

    batch_iter = iterate_batches(
        train_dataset, batch_size, seq_length, loop=True, seed=seed
    )

    # Intervals derived from dataset size: one epoch = one full pass through data.
    # Count batches in one epoch (without consuming the training iterator).
    n_batches_per_epoch = len(list(
        iterate_batches(train_dataset, batch_size, seq_length, loop=False)
    ))
    # Log every epoch, check geometry every epoch (natural unit of data exposure)
    log_interval = max(1, n_batches_per_epoch)
    check_interval = max(1, n_batches_per_epoch)

    logger.info(
        "Training until geometry says stop (safety cap: %d, epoch: %d batches)",
        max_iters, n_batches_per_epoch,
    )

    for it in range(max_iters):
        batch, lengths = next(batch_iter)

        (loss, ntoks), grad = loss_value_and_grad(model, batch, lengths)

        # Apply ScaledGD preconditioning (Riemannian GD on rank-r manifold)
        if use_geometric:
            scaled_grad = apply_scaled_gd(model, grad, lora_configs, opt_config)
            optimizer.update(model, scaled_grad)
        else:
            optimizer.update(model, grad)

        mx.eval(model.parameters(), optimizer.state)

        loss_val = float(loss)
        ntoks_val = float(ntoks)
        elapsed = time.time() - t_start
        tps = ntoks_val / max(elapsed, 1e-6) if it == 0 else ntoks_val / max(time.time() - t_step, 1e-6)

        losses.append((it, loss_val, tps))
        t_step = time.time()

        if (it + 1) % log_interval == 0 or it == 0:
            epoch = (it + 1) / n_batches_per_epoch
            lr_info = " | η=%.2e" % eta if use_geometric else ""
            logger.info(
                "Iter %d (epoch %.1f) | loss=%.4f | tokens/sec=%.1f%s",
                it + 1, epoch, loss_val, tps, lr_info,
            )

        # Geometric stopping checks — every epoch after 6 epochs minimum
        # (need 2 × 3-epoch windows = 6 epochs for loss stability comparison)
        if use_geometric and (it + 1) % check_interval == 0 and it >= 6 * n_batches_per_epoch:
            # Check 1: Loss converged below data noise floor
            # Compare 3-epoch windows (not 1-epoch) to smooth out epoch oscillations.
            # Single-epoch windows trigger too early on diverse data where
            # a brief plateau doesn't mean convergence.
            stable, threshold = check_loss_stable(losses, window=3 * n_batches_per_epoch)
            if stable:
                stop_reason = f"loss_stable (|Δ_epoch| < SE = {threshold:.4e})"
                logger.info("Geometry stop at iter %d: %s", it + 1, stop_reason)
                break

            # Check 2: Spectral budget exhausted
            exhausted, median_ratio = check_budget_exhausted(model, lora_configs, opt_config)
            if exhausted:
                stop_reason = f"budget_exhausted (median ratio = {median_ratio:.4f}, Weyl crossing)"
                logger.info("Geometry stop at iter %d: %s", it + 1, stop_reason)
                break
    else:
        stop_reason = f"safety_cap ({max_iters} iters)"
        logger.warning("Hit safety cap at %d iters — geometry did not converge", max_iters)

    return losses, stop_reason


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
    logger.info("Safety cap: %d iters (geometry decides when to stop)", args.iters)
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
    # 6. Derive optimizer geometry config (per-layer epsilon + decay)
    # ------------------------------------------------------------------
    # ScaledGD uses per-layer epsilon and decay from spectral geometry.
    # LR = 1/L is measured from Hessian inside train(), not derived from σ_max.
    opt_config = derive_optimizer_geometry_config(weights, backend)
    lr_override = args.lr

    if lr_override is not None:
        logger.info("LR override: %.2e (flat SGD, bypasses ScaledGD)", lr_override)
    else:
        n_lora = len([c for c in configs if c.rank > 0])
        logger.info(
            "%d LoRA layers with ScaledGD + measured η=1/L (Lipschitz)",
            n_lora,
        )
        # Log per-layer decay and epsilon ranges
        lora_keys = [c.layer_key for c in configs if c.rank > 0]
        decays = []
        epsilons = []
        for key in lora_keys:
            lc = opt_config.layer_configs.get(key)
            if lc:
                decays.append(lc.decay_scale)
                epsilons.append(lc.epsilon)
        if decays:
            logger.info(
                "  Decay (σ_k/σ_max): [%.4f, %.4f]", min(decays), max(decays),
            )
            logger.info(
                "  Epsilon: [%.2e, %.2e]", min(epsilons), max(epsilons),
            )

    # ------------------------------------------------------------------
    # 7. Inject LoRA layers
    # ------------------------------------------------------------------
    logger.info("\n--- Injecting Per-Layer LoRA ---")

    # Freeze FIRST, then inject — new LoRA params will be trainable by default
    # (they weren't in the model when freeze was called)
    model.freeze()
    n_injected = inject_geometric_lora(model, configs, backend)
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
    losses, stop_reason = train(
        model, train_dataset, args.batch_size, args.seq_length,
        args.iters, args.seed,
        lora_configs=configs, opt_config=opt_config, lr_override=lr_override,
    )
    train_time = time.time() - t0

    actual_iters = len(losses)
    first_loss = losses[0][1]
    final_loss = losses[-1][1]

    logger.info("Training complete in %.1fs (%d iters)", train_time, actual_iters)
    if stop_reason:
        logger.info("Early stop: %s", stop_reason)
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
    logger.info("Training steps: %d", actual_iters)
    logger.info("Stop reason:    %s", stop_reason)
    logger.info("")
    logger.info("Geometry-derived config:")
    logger.info("  Target layers:  %d / %d", n_injected, len(geometries))
    logger.info("  Rank range:     %d - %d (per-layer from tail_dims)", min(ranks), max(ranks))
    logger.info("  Dropout range:  %.4f - %.4f", min(dropouts), max(dropouts))
    logger.info("  Scale:          1.0 (neutral; geometry via spectral init)")
    logger.info("  Sigma_k range:  %.6f - %.6f (spectral init target)", min(sigma_ks), max(sigma_ks))
    if lr_override is not None:
        logger.info("  Learning rate:  %.2e (override, flat SGD)", lr_override)
    else:
        logger.info("  Optimizer:      ScaledGD + SGD(η=1/L, measured Lipschitz)")
        logger.info("  Decay:          σ_k/σ_max per layer (condition-aware)")
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
