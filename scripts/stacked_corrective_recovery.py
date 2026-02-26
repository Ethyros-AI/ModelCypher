#!/usr/bin/env python3
"""Experiment 4: Stacked Corrective Recovery.

Iteratively trains corrective LoRA adapters to recover full-precision behavior
from a quantized model. After each round:
1. Train corrective LoRA (MSE on logits vs bf16 reference)
2. Fuse adapter delta into model weights (dequantizing quantized layers)
3. Run RMT decomposition on the weight-space residual
4. If signal_rank = 0 (residual in MP noise bulk): stop — recovery complete

The convergence criterion is geometric: each round targets the systematic
(above-MP-bulk) component of the remaining error. When no signal remains,
the residual is genuinely random and uncorrectable by low-rank methods.

Usage:
    poetry run python scripts/stacked_corrective_recovery.py

    # Custom rounds and iterations
    poetry run python scripts/stacked_corrective_recovery.py \
        --max-rounds 5 --max-iters 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import statistics
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
logger = logging.getLogger("stacked_recovery")

# Default paths
DEFAULT_QUANTIZED = (
    "results/feasibility_map/20260225T160732Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-8bit-g64-affine"
)
DEFAULT_FP = "/Volumes/CodeCypher/models/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_TRAIN = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"

BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10000


# ── Utilities ────────────────────────────────────────────────────────────


def _clear_gpu_cache() -> None:
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the mean. Returns (mean, lower, upper)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    observed_mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    alpha = 1.0 - ci
    lo = means[max(0, int(alpha / 2 * n_bootstrap))]
    hi = means[min(n_bootstrap - 1, int((1.0 - alpha / 2) * n_bootstrap))]
    return observed_mean, lo, hi


# ── Weight extraction and RMT ───────────────────────────────────────────


def _extract_weights_streaming(model, adapter) -> dict[str, Any]:
    """Extract dequantized weight matrices from model, one layer at a time."""
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise ValueError("Model has no .layers attribute")

    weights: dict[str, Any] = {}
    for layer_idx, layer in enumerate(base.layers):
        for block_name, proj_names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("up_proj", "down_proj", "gate_proj")),
        ):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in proj_names:
                proj = getattr(block, proj_name, None)
                if proj is None:
                    continue
                key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                # Handle NBLoRALinear: get effective weight (base + delta)
                if hasattr(proj, "linear"):
                    # It's an NBLoRALinear — get base + LoRA delta
                    base_w = adapter._dequantize_weight(proj.linear)
                    A, B = proj._cayley_transform()
                    S = mx.clip(proj.S_raw, 0.0, proj._scale_bound)
                    delta = 2.0 * B.T @ mx.diag(S) @ A  # [out, in]
                    w = base_w + delta
                    mx.eval(w)
                    del base_w, delta, A, B, S
                elif hasattr(proj, "weight"):
                    w = adapter._dequantize_weight(proj)
                else:
                    continue
                weights[key] = w

    return weights


def _analyze_residual_rmt(
    corrected_weights: dict[str, Any],
    fp_weights: dict[str, Any],
    backend: Any,
) -> dict[str, Any]:
    """Run RMT decomposition on the residual E = W_fp - W_corrected."""
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )

    common_keys = sorted(set(corrected_weights.keys()) & set(fp_weights.keys()))
    per_layer: list[dict[str, Any]] = []
    n_svd_failures = 0

    for key in common_keys:
        fp_w = fp_weights[key]
        c_w = corrected_weights[key]

        # Residual in float32
        E = fp_w.astype(mx.float32) - c_w.astype(mx.float32)
        mx.eval(E)

        m, n = int(E.shape[0]), int(E.shape[1])
        frob_norm = float(mx.sqrt(mx.sum(E * E)).item())

        signal_rank = 0
        noise_rank = min(m, n)
        signal_variance_fraction = 0.0
        mp_upper_edge = 0.0
        spectral_norm = 0.0
        svd_success = True

        try:
            S = mx.linalg.svd(E, compute_uv=False, stream=mx.cpu)
            mx.eval(S)

            n_sv = int(S.shape[0])
            spectral_norm = float(S[0].item()) if n_sv > 0 else 0.0

            rmt_result = compute_signal_rank_from_singular_values(
                S, n_samples=m, n_features=n,
                backend=backend, center_correction=True,
            )
            signal_rank = rmt_result.signal_rank
            noise_rank = rmt_result.noise_rank
            signal_variance_fraction = rmt_result.signal_variance_fraction
            mp_upper_edge = rmt_result.mp_upper_edge
            del S

        except Exception as exc:
            logger.warning("  SVD failed for %s: %s", key, exc)
            svd_success = False
            n_svd_failures += 1

        del E
        gc.collect()

        per_layer.append({
            "layer_key": key,
            "shape": [m, n],
            "signal_rank": signal_rank,
            "noise_rank": noise_rank,
            "signal_variance_fraction": signal_variance_fraction,
            "mp_upper_edge": mp_upper_edge,
            "error_spectral_norm": spectral_norm,
            "error_frobenius_norm": frob_norm,
            "svd_success": svd_success,
        })

    # Aggregate
    valid_layers = [l for l in per_layer if l["svd_success"]]
    signal_ranks = [l["signal_rank"] for l in valid_layers]
    sv_fracs = [l["signal_variance_fraction"] for l in valid_layers]
    frob_norms = [l["error_frobenius_norm"] for l in valid_layers]

    sr_mean, sr_ci_lo, sr_ci_hi = _bootstrap_ci(
        [float(r) for r in signal_ranks]
    )
    svf_mean, svf_ci_lo, svf_ci_hi = _bootstrap_ci(sv_fracs)

    return {
        "n_layers": len(common_keys),
        "n_svd_failures": n_svd_failures,
        "n_layers_with_signal": sum(1 for r in signal_ranks if r > 0),
        "mean_signal_rank": sr_mean,
        "signal_rank_ci": [sr_ci_lo, sr_ci_hi],
        "mean_signal_variance_fraction": svf_mean,
        "sv_frac_ci": [svf_ci_lo, svf_ci_hi],
        "mean_frobenius_norm": statistics.mean(frob_norms) if frob_norms else 0.0,
        "median_frobenius_norm": statistics.median(frob_norms) if frob_norms else 0.0,
        "signal_rank_zero": sr_ci_hi <= 0.0,  # Conservative: CI upper bound
        "per_layer": per_layer,
    }


# ── LoRA fusion ──────────────────────────────────────────────────────────


def _fuse_lora_into_model(
    model, adapter, fp_weights: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Fuse all NBLoRALinear modules back into plain nn.Linear.

    After fusion, the model has full-precision weights with the LoRA
    correction baked in. Ready for next round of LoRA injection.

    If fp_weights is provided, computes delta diagnostics:
    - ||delta||_F per layer
    - ||E||_F per layer (error = fp - base, before correction)
    - cosine(E, delta) — alignment between error and correction

    Returns (n_fused, delta_stats).
    """
    from modelcypher.backends.mlx_training_adapter_core import NBLoRALinear

    base = getattr(model, "model", model)
    n_fused = 0
    delta_stats: list[dict[str, Any]] = []

    for layer_idx, layer in enumerate(base.layers):
        for block_name in ("self_attn", "mlp"):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ):
                proj = getattr(block, proj_name, None)
                if not isinstance(proj, NBLoRALinear):
                    continue

                # Get base weight (dequantize if needed)
                base_weight = adapter._dequantize_weight(proj.linear)

                # Compute LoRA delta: 2 * B^T @ diag(S) @ A → [out, in]
                A, B = proj._cayley_transform()
                S = mx.clip(proj.S_raw, 0.0, proj._scale_bound)
                delta = 2.0 * B.T @ mx.diag(S) @ A

                # Delta diagnostics
                if fp_weights is not None:
                    key = f"model.layers.{layer_idx}.{block_name}.{proj_name}.weight"
                    fp_w = fp_weights.get(key)
                    if fp_w is not None:
                        delta_f32 = delta.astype(mx.float32)
                        E = fp_w.astype(mx.float32) - base_weight.astype(mx.float32)
                        mx.eval(delta_f32, E)

                        delta_frob = float(mx.sqrt(mx.sum(delta_f32 * delta_f32)).item())
                        E_frob = float(mx.sqrt(mx.sum(E * E)).item())

                        # Cosine similarity: <E, delta> / (||E|| * ||delta||)
                        inner = float(mx.sum(E * delta_f32).item())
                        denom = E_frob * delta_frob
                        cosine = inner / denom if denom > 0 else 0.0

                        delta_stats.append({
                            "layer_key": key,
                            "delta_frob": delta_frob,
                            "error_frob": E_frob,
                            "ratio": delta_frob / E_frob if E_frob > 0 else 0.0,
                            "cosine_E_delta": cosine,
                        })
                        del delta_f32, E

                # Fused weight
                fused_weight = (base_weight + delta).astype(base_weight.dtype)
                mx.eval(fused_weight)

                # Create plain Linear replacement
                new_linear = nn.Linear(
                    proj._in_features, proj._out_features, bias=False,
                )
                new_linear.weight = fused_weight

                # Copy bias if present
                if hasattr(proj.linear, "bias") and proj.linear.bias is not None:
                    new_linear.bias = proj.linear.bias

                setattr(block, proj_name, new_linear)
                n_fused += 1

                del base_weight, delta, fused_weight, A, B, S

    gc.collect()
    _clear_gpu_cache()
    return n_fused, delta_stats


# ── Dequantize model ─────────────────────────────────────────────────────


def _dequantize_model(model, adapter) -> int:
    """Convert all QuantizedLinear modules to plain Linear.

    This is done once at the start so that subsequent LoRA injection
    and fusion operate on float weights.

    Returns number of layers dequantized.
    """
    base = getattr(model, "model", model)
    n_deq = 0

    for layer_idx, layer in enumerate(base.layers):
        for block_name in ("self_attn", "mlp"):
            block = getattr(layer, block_name, None)
            if block is None:
                continue
            for proj_name in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ):
                proj = getattr(block, proj_name, None)
                if proj is None or not isinstance(proj, nn.QuantizedLinear):
                    continue

                # Dequantize
                w = adapter._dequantize_weight(proj)
                in_features = int(w.shape[1])
                out_features = int(w.shape[0])

                new_linear = nn.Linear(in_features, out_features, bias=False)
                new_linear.weight = w
                mx.eval(new_linear.weight)

                if hasattr(proj, "bias") and proj.bias is not None:
                    new_linear.bias = proj.bias

                setattr(block, proj_name, new_linear)
                n_deq += 1

    gc.collect()
    _clear_gpu_cache()
    return n_deq


# ── Functional error fraction ────────────────────────────────────────────


def _compute_functional_fractions(
    q_model, fp_weights: dict[str, Any], tokenizer, eval_texts: list[str],
    n_samples: int = 30, max_len: int = 128,
) -> list[dict[str, Any]]:
    """Measure what fraction of E's energy is in high-energy activation directions.

    For each layer, collects input activations X, computes the eigendecomposition
    of X^T @ X (activation covariance), then measures how much of the weight error
    E = W_fp - W_q projects onto the top-k eigenvectors.

    The "functional fraction at effective dimension" tells us what fraction of the
    weight error is in the directions that activations concentrate in.

    Only computed for qkv projections (which share the layer input as activation).
    """
    base = getattr(q_model, "model", q_model)
    if not hasattr(base, "layers"):
        return []

    # Tokenize evaluation texts
    all_tokens = []
    for text in eval_texts[:n_samples]:
        tokens = tokenizer.encode(text)
        all_tokens.append(mx.array(tokens[:max_len]))

    # Pad and stack
    max_seq = max(t.shape[0] for t in all_tokens)
    padded = []
    for t in all_tokens:
        if t.shape[0] < max_seq:
            pad_len = max_seq - t.shape[0]
            t = mx.concatenate([t, mx.zeros(pad_len, dtype=t.dtype)])
        padded.append(t)
    batch = mx.stack(padded)

    # Run embedding
    h = base.embed_tokens(batch)
    mx.eval(h)

    results = []

    for layer_idx, layer in enumerate(base.layers):
        # h is the input to this layer; qkv projections receive input_layernorm(h)
        # LayerNorm preserves the subspace (just rescales), so h spans the same space

        # Flatten: [n_samples, seq_len, D] → [N, D]
        X = h.reshape(-1, h.shape[-1]).astype(mx.float32)
        N_tok, D = int(X.shape[0]), int(X.shape[1])
        mx.eval(X)

        # Activation covariance: X^T X / N
        XtX = (X.T @ X) / N_tok
        mx.eval(XtX)

        try:
            eigvals, eigvecs = mx.linalg.eigh(XtX, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
        except Exception as exc:
            logger.warning("  eigh failed for layer %d: %s, skipping", layer_idx, exc)
            h = layer(h)
            mx.eval(h)
            del X, XtX
            gc.collect()
            continue

        # eigh returns ascending order; flip to descending
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Effective dimensionality (participation ratio)
        total_var = float(mx.sum(eigvals).item())
        sum_sq = float(mx.sum(eigvals * eigvals).item())
        D_eff = total_var ** 2 / sum_sq if sum_sq > 0 else float(D)

        # For each qkv projection, compute functional fraction
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
            fp_w = fp_weights.get(key)
            if fp_w is None:
                continue

            proj = getattr(getattr(layer, "self_attn", None), proj_name, None)
            if proj is None or not hasattr(proj, "weight"):
                continue

            q_w = proj.weight.astype(mx.float32)
            E = fp_w.astype(mx.float32) - q_w  # [out, in]
            mx.eval(E)

            E_frob_sq = float(mx.sum(E * E).item())

            if E_frob_sq <= 0:
                del E, q_w
                continue

            # Project E onto each eigenvector: p_i = ||E @ v_i||²
            # E @ eigvecs → [out, D], then sum squares per column
            E_proj = E @ eigvecs
            mx.eval(E_proj)
            proj_energy = mx.sum(E_proj * E_proj, axis=0)  # [D]
            mx.eval(proj_energy)
            pe = proj_energy.tolist()

            # Cumulative fraction at effective dimension
            k_eff = max(1, int(round(D_eff)))
            frac_at_eff = sum(pe[:k_eff]) / E_frob_sq
            # Also compute at k=D/10 and k=D/100
            k_10pct = max(1, D // 10)
            k_1pct = max(1, D // 100)
            frac_at_10pct = sum(pe[:k_10pct]) / E_frob_sq
            frac_at_1pct = sum(pe[:k_1pct]) / E_frob_sq
            # Baseline: if E were isotropic, fraction = k/D
            iso_frac_eff = k_eff / D
            iso_frac_10pct = k_10pct / D

            results.append({
                "layer_key": key,
                "layer_idx": layer_idx,
                "error_frob": math.sqrt(E_frob_sq),
                "D": D,
                "D_eff": D_eff,
                "k_eff": k_eff,
                "frac_at_eff_dim": frac_at_eff,
                "frac_at_10pct": frac_at_10pct,
                "frac_at_1pct": frac_at_1pct,
                "iso_frac_eff": iso_frac_eff,
                "iso_frac_10pct": iso_frac_10pct,
            })

            del E, E_proj, q_w, proj_energy

        # Advance to next layer
        h = layer(h)
        mx.eval(h)

        del X, XtX, eigvals, eigvecs
        gc.collect()
        _clear_gpu_cache()

    return results


# ── CKA measurement ─────────────────────────────────────────────────────


def _collect_activations(
    model, tokenizer, texts, backend, n_samples=30,
) -> dict[int, list]:
    """Collect per-layer mean-pooled activations for CKA."""
    activations: dict[int, list] = {}
    for text in texts[:n_samples]:
        acts = backend.collect_hidden_activations(model, tokenizer, [text])
        for layer_idx, act in acts.items():
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(pooled)
    return activations


def _compute_cka(acts_a, acts_b, backend) -> dict[str, Any]:
    """Linear CKA between two sets of per-layer activations."""
    from modelcypher.core.domain.geometry.cka import (
        compute_linear_cka_from_activations,
    )

    per_layer: dict[int, float] = {}
    common_layers = sorted(set(acts_a.keys()) & set(acts_b.keys()))

    for layer_idx in common_layers:
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


# ── Training loop (single round) ────────────────────────────────────────


def _train_corrective_round(
    q_model,
    fp_model,
    tokenizer,
    adapter,
    backend,
    train_dataset: str,
    max_iters: int,
    batch_size: int,
    seq_length: int,
    seed: int,
    noise_fraction: float = 0.0,
) -> dict[str, Any]:
    """Train one round of corrective LoRA. Returns training metrics.

    Args:
        noise_fraction: RMT noise fraction (1 - sv_frac) for f* derivation.
            f* = initial_loss × noise_fraction is the irreducible loss floor.
            If 0.0, uses f*=0 (original SPS).
    """
    from modelcypher.core.domain.training.geometric_lora import (
        select_target_modules,
    )
    from modelcypher.core.domain.training.mass_step_size import (
        compute_per_step_rates,
        derive_spectral_ceiling,
    )

    # Geometry analysis
    logger.info("  Analyzing model geometry...")
    geometries = adapter.analyze_model_geometry_streaming(
        q_model, use_randomized=True,
    )
    target_modules = select_target_modules(geometries)
    logger.info("  Targeting %d modules for corrective LoRA", len(target_modules))

    # Inject NB-LoRA
    n_injected = adapter.inject_nb_lora(q_model, geometries, target_modules)
    adapter.freeze_and_apply_lora(q_model)
    logger.info("  Injected %d NB-LoRA layers", n_injected)

    n_trainable = sum(
        p.size for _, p in mlx.utils.tree_flatten(q_model.trainable_parameters())
    )

    # MASS step size
    sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)
    sigma_k_vals = [g.sigma_k for g in geometries.values() if g.sigma_k > 0]
    sigma_k_min = min(sigma_k_vals)
    eta_ceiling = derive_spectral_ceiling(
        sigma_k_min=sigma_k_min, sigma_max_global=sigma_max,
    )

    # Load training data
    tokenized = []
    with open(train_dataset, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        tokens = tokenizer.encode(text)
                        if len(tokens) >= 2:
                            tokenized.append(tokens)
                except json.JSONDecodeError:
                    continue

    n_batches = max(1, len(tokenized) // batch_size)
    if n_batches > 1:
        eta_ceiling = eta_ceiling / math.sqrt(n_batches)

    logger.info(
        "  MASS: sigma_max=%.4e, sigma_k_min=%.4e, eta=%.4e",
        sigma_max, sigma_k_min, eta_ceiling,
    )

    optimizer = opt.SGD(learning_rate=eta_ceiling, momentum=0.0)

    # Corrective MSE loss
    def corrective_loss_fn(q_model, batch):
        q_logits = q_model(batch)[:, :-1, :]
        fp_logits = mx.stop_gradient(fp_model(batch)[:, :-1, :])
        diff = q_logits - fp_logits
        mse = mx.mean(diff * diff)
        ntoks = batch.shape[0] * (batch.shape[1] - 1)
        return mse, mx.array(float(ntoks))

    loss_and_grad = nn.value_and_grad(q_model, corrective_loss_fn)

    # Training
    backend.random_seed(seed)
    losses: list[tuple[int, float]] = []
    f_star = 0.0
    best_loss = float("inf")
    best_iter = -1
    start_time = time.monotonic()

    for it in range(max_iters):
        indices = [i % len(tokenized) for i in range(
            it * batch_size, (it + 1) * batch_size,
        )]
        batch_tokens = [tokenized[idx] for idx in indices]

        padded = []
        for tokens in batch_tokens:
            if len(tokens) > seq_length:
                padded.append(tokens[:seq_length])
            else:
                padded.append(tokens + [0] * (seq_length - len(tokens)))
        batch = mx.array(padded)

        (loss, ntoks), grad = loss_and_grad(q_model, batch)

        flat_grads = [
            p.reshape(-1) for _, p in mlx.utils.tree_flatten(grad) if p.size > 0
        ]
        d_norm_sq = sum(mx.sum(p * p) for p in flat_grads)
        mx.eval(d_norm_sq, loss)
        d_norm = float(mx.sqrt(d_norm_sq).item())
        loss_val = float(loss.item())

        # Derive f* from initial loss + RMT noise fraction
        if it == 0 and noise_fraction > 0:
            f_star = loss_val * noise_fraction
            logger.info(
                "  Derived f*=%.6f (initial_loss=%.6f × noise_fraction=%.4f)",
                f_star, loss_val, noise_fraction,
            )

        eta_step, _, _, _, _ = compute_per_step_rates(
            loss_val, d_norm, sigma_k_min, eta_ceiling,
            f_star=f_star,
        )
        optimizer.learning_rate = mx.array(eta_step)

        optimizer.update(q_model, grad)
        mx.eval(q_model.parameters(), optimizer.state)

        for _, module in adapter._iter_nb_lora_modules(q_model):
            module.clamp_scale()
            mx.eval(module.S_raw)

        if loss_val < best_loss:
            best_loss = loss_val
            best_iter = it

        losses.append((it, loss_val))

        if it % 10 == 0 or it == max_iters - 1:
            elapsed = time.monotonic() - start_time
            logger.info(
                "  iter %d/%d: loss=%.6f, eta=%.4e, d_norm=%.4e, elapsed=%.1fs",
                it, max_iters, loss_val, eta_step, d_norm, elapsed,
            )

    training_time = time.monotonic() - start_time

    return {
        "n_iters": len(losses),
        "initial_loss": losses[0][1] if losses else 0.0,
        "final_loss": losses[-1][1] if losses else 0.0,
        "best_loss": best_loss,
        "best_iter": best_iter,
        "f_star": f_star,
        "training_time_seconds": training_time,
        "n_trainable_params": n_trainable,
        "n_lora_layers": n_injected,
        "eta_ceiling": eta_ceiling,
        "sigma_max": sigma_max,
        "sigma_k_min": sigma_k_min,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    args = _parse_args()

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    adapter = MLXTrainingAdapter(backend)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stacked Corrective Recovery — run_id=%s", run_id)
    logger.info("Quantized model: %s", args.quantized_model)
    logger.info("FP model: %s", args.fp_model)
    logger.info("Max rounds: %d, iters/round: %d", args.max_rounds, args.max_iters)
    logger.info("Output: %s", output_dir)

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment": "stacked_corrective_recovery",
        "config": {
            "quantized_model": args.quantized_model,
            "fp_model": args.fp_model,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "max_rounds": args.max_rounds,
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "seed": args.seed,
            "n_cka_probes": args.n_cka_probes,
        },
    }

    # ── Load FP model (stays in memory for MSE loss + RMT comparison) ──
    logger.info("Loading bf16 reference model...")
    fp_model, tokenizer = backend.load_model(str(args.fp_model))

    # Extract FP weights once for RMT comparison
    logger.info("Extracting FP reference weights...")
    fp_weights = _extract_weights_streaming(fp_model, adapter)
    logger.info("  Extracted %d layer weights", len(fp_weights))

    # Collect FP activations once for CKA
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

    logger.info("Collecting FP reference activations (%d probes)...", args.n_cka_probes)
    fp_activations = _collect_activations(
        fp_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
    )

    # ── Load Q model and dequantize to float ──
    logger.info("Loading quantized model...")
    q_model, _ = backend.load_model(str(args.quantized_model))

    logger.info("Dequantizing model weights to float...")
    n_deq = _dequantize_model(q_model, adapter)
    logger.info("  Dequantized %d QuantizedLinear modules", n_deq)

    # ── Measure initial CKA (quantized vs FP) ──
    logger.info("Measuring initial CKA (dequantized Q vs FP)...")
    q_activations = _collect_activations(
        q_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
    )
    initial_cka = _compute_cka(fp_activations, q_activations, backend)
    logger.info(
        "Initial CKA: min=%.4f, mean=%.4f",
        initial_cka["min_cka"], initial_cka["mean_cka"],
    )
    del q_activations
    gc.collect()

    results["initial_cka"] = initial_cka

    # ── Optional: Functional error fraction analysis ──
    if args.functional_fraction:
        logger.info("Computing functional error fraction (E projected onto activation subspace)...")
        func_fracs = _compute_functional_fractions(
            q_model, fp_weights, tokenizer, eval_texts,
            n_samples=args.n_cka_probes, max_len=128,
        )
        if func_fracs:
            d_effs = [f["D_eff"] for f in func_fracs]
            fracs_eff = [f["frac_at_eff_dim"] for f in func_fracs]
            fracs_10 = [f["frac_at_10pct"] for f in func_fracs]
            fracs_1 = [f["frac_at_1pct"] for f in func_fracs]
            iso = [f["iso_frac_eff"] for f in func_fracs]
            logger.info(
                "  Functional fraction: D_eff=%.1f, "
                "frac_at_eff=%.4f (iso=%.4f), "
                "frac_at_10%%=%.4f, frac_at_1%%=%.4f "
                "(%d projections)",
                sum(d_effs) / len(d_effs),
                sum(fracs_eff) / len(fracs_eff),
                sum(iso) / len(iso),
                sum(fracs_10) / len(fracs_10),
                sum(fracs_1) / len(fracs_1),
                len(func_fracs),
            )
            # Log per-layer summary (one entry per layer, averaging across qkv)
            by_layer: dict[int, list[float]] = {}
            for f in func_fracs:
                by_layer.setdefault(f["layer_idx"], []).append(f["frac_at_eff_dim"])
            for lidx in sorted(by_layer.keys()):
                vals = by_layer[lidx]
                logger.info(
                    "    Layer %2d: D_eff=%.1f, frac_at_eff=%.4f (mean of %d projections)",
                    lidx,
                    next(f["D_eff"] for f in func_fracs if f["layer_idx"] == lidx),
                    sum(vals) / len(vals),
                    len(vals),
                )
            results["functional_fractions"] = func_fracs
        gc.collect()
        _clear_gpu_cache()

    # ── Stacking loop ──
    rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds_reached"
    cka_history: list[float] = [initial_cka["mean_cka"]]

    for round_idx in range(args.max_rounds):
        round_start = time.monotonic()
        logger.info("")
        logger.info("=" * 60)
        logger.info("ROUND %d / %d", round_idx + 1, args.max_rounds)
        logger.info("=" * 60)

        # ── Pre-training RMT on current residual ──
        logger.info("Analyzing residual (RMT)...")
        q_weights = _extract_weights_streaming(q_model, adapter)
        rmt_before = _analyze_residual_rmt(q_weights, fp_weights, backend)
        del q_weights
        gc.collect()
        _clear_gpu_cache()

        logger.info(
            "  Residual: %d/%d layers with signal, mean_signal_rank=%.1f "
            "CI=[%.1f, %.1f], mean_sv_frac=%.4f, mean_||E||_F=%.6f",
            rmt_before["n_layers_with_signal"],
            rmt_before["n_layers"],
            rmt_before["mean_signal_rank"],
            rmt_before["signal_rank_ci"][0],
            rmt_before["signal_rank_ci"][1],
            rmt_before["mean_signal_variance_fraction"],
            rmt_before["mean_frobenius_norm"],
        )

        # Diagnostic: log if signal_rank enters bulk (won't happen —
        # signal_rank is invariant to activation-space correction, see Experiment 4)
        if round_idx > 0 and rmt_before["signal_rank_zero"]:
            logger.info(
                "  NOTE: signal_rank CI upper bound <= 0 (unexpected — "
                "signal_rank should be invariant to correction)"
            )

        # ── Train corrective LoRA ──
        # Derive noise fraction from RMT: f* = initial_loss × (1 - sv_frac)
        sv_frac = rmt_before["mean_signal_variance_fraction"]
        round_noise_fraction = 1.0 - sv_frac if sv_frac > 0 else 0.0
        logger.info(
            "Training corrective LoRA (round %d, sv_frac=%.4f, noise=%.4f)...",
            round_idx + 1, sv_frac, round_noise_fraction,
        )
        training_result = _train_corrective_round(
            q_model=q_model,
            fp_model=fp_model,
            tokenizer=tokenizer,
            adapter=adapter,
            backend=backend,
            train_dataset=args.train_dataset,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed + round_idx,  # Different seed each round
            noise_fraction=round_noise_fraction,
        )

        # ── Save this round's adapter ──
        round_adapter_path = output_dir / f"adapter_round_{round_idx + 1}"
        adapter.save_adapter(q_model, round_adapter_path)
        logger.info("  Adapter saved: %s", round_adapter_path)

        # ── Fuse LoRA into weights (with delta diagnostics) ──
        logger.info("Fusing LoRA delta into model weights...")
        n_fused, delta_stats = _fuse_lora_into_model(
            q_model, adapter, fp_weights=fp_weights,
        )
        logger.info("  Fused %d modules", n_fused)
        if delta_stats:
            ratios = [s["ratio"] for s in delta_stats]
            cosines = [s["cosine_E_delta"] for s in delta_stats]
            mean_ratio = sum(ratios) / len(ratios)
            mean_cosine = sum(cosines) / len(cosines)
            max_ratio = max(ratios)
            logger.info(
                "  Delta diagnostics: mean ||delta||/||E|| = %.6f, "
                "max = %.6f, mean cos(E,delta) = %+.4f",
                mean_ratio, max_ratio, mean_cosine,
            )

        # ── Post-training RMT ──
        logger.info("Analyzing post-training residual (RMT)...")
        q_weights_after = _extract_weights_streaming(q_model, adapter)
        rmt_after = _analyze_residual_rmt(q_weights_after, fp_weights, backend)
        del q_weights_after
        gc.collect()
        _clear_gpu_cache()

        logger.info(
            "  Post-correction: %d/%d layers with signal, "
            "mean_signal_rank=%.1f CI=[%.1f, %.1f], "
            "mean_||E||_F=%.6f",
            rmt_after["n_layers_with_signal"],
            rmt_after["n_layers"],
            rmt_after["mean_signal_rank"],
            rmt_after["signal_rank_ci"][0],
            rmt_after["signal_rank_ci"][1],
            rmt_after["mean_frobenius_norm"],
        )

        # ── Post-training CKA ──
        logger.info("Measuring post-round CKA...")
        q_acts = _collect_activations(
            q_model, tokenizer, eval_texts, backend, n_samples=args.n_cka_probes,
        )
        round_cka = _compute_cka(fp_activations, q_acts, backend)
        logger.info(
            "  CKA: min=%.4f, mean=%.4f",
            round_cka["min_cka"], round_cka["mean_cka"],
        )
        del q_acts
        gc.collect()

        round_time = time.monotonic() - round_start

        round_result = {
            "round": round_idx + 1,
            "rmt_before_training": rmt_before,
            "training": training_result,
            "adapter_path": str(round_adapter_path),
            "n_fused": n_fused,
            "delta_stats": delta_stats,
            "rmt_after_training": rmt_after,
            "cka_after_round": round_cka,
            "round_time_seconds": round_time,
            "stopped_before_training": False,
        }
        rounds.append(round_result)

        # Log improvement
        sr_reduction = (
            rmt_before["mean_signal_rank"] - rmt_after["mean_signal_rank"]
        )
        frob_reduction = (
            rmt_before["mean_frobenius_norm"] - rmt_after["mean_frobenius_norm"]
        )
        logger.info(
            "  Round %d summary: signal_rank %.1f → %.1f (Δ=%.1f), "
            "||E||_F %.6f → %.6f (Δ=%.6f), CKA=%.4f, time=%.1fs",
            round_idx + 1,
            rmt_before["mean_signal_rank"], rmt_after["mean_signal_rank"],
            sr_reduction,
            rmt_before["mean_frobenius_norm"], rmt_after["mean_frobenius_norm"],
            frob_reduction,
            round_cka["mean_cka"],
            round_time,
        )

        # ── CKA plateau detection ──
        # TODO: derive plateau threshold from CKA measurement variance
        # (empirical: run CKA N times on same model, compute std of mean_cka;
        #  plateau = delta < k * std for some justified k, e.g. 2-sigma).
        # Until derived, log deltas for analysis but do NOT auto-stop.
        cka_history.append(round_cka["mean_cka"])
        if len(cka_history) >= 3:
            recent_deltas = [
                cka_history[-i] - cka_history[-i - 1] for i in range(1, 3)
            ]
            logger.info(
                "  CKA deltas (last 2 rounds): %s",
                ", ".join(f"{d:+.4f}" for d in reversed(recent_deltas)),
            )

    # ── Final summary ──
    results["rounds"] = rounds
    results["stop_reason"] = stop_reason
    results["n_rounds_completed"] = len(rounds)

    # Summary trajectory
    cka_trajectory = [initial_cka["mean_cka"]]
    signal_rank_trajectory = []
    frob_trajectory = []
    for r in rounds:
        if "cka_after_round" in r:
            cka_trajectory.append(r["cka_after_round"]["mean_cka"])
        if "rmt_after_training" in r:
            signal_rank_trajectory.append(
                r["rmt_after_training"]["mean_signal_rank"]
            )
            frob_trajectory.append(
                r["rmt_after_training"]["mean_frobenius_norm"]
            )

    results["summary"] = {
        "initial_cka": initial_cka["mean_cka"],
        "final_cka": cka_trajectory[-1] if cka_trajectory else 0.0,
        "cka_improvement": (
            cka_trajectory[-1] - cka_trajectory[0]
            if len(cka_trajectory) >= 2 else 0.0
        ),
        "cka_trajectory": cka_trajectory,
        "signal_rank_trajectory": signal_rank_trajectory,
        "frobenius_trajectory": frob_trajectory,
        "stop_reason": stop_reason,
    }

    # Verdict
    cka_improvement = results["summary"]["cka_improvement"]
    if stop_reason == "noise_floor_reached":
        verdict = (
            f"CONVERGED: Residual entered MP noise bulk after "
            f"{len(rounds)} rounds. CKA: {initial_cka['mean_cka']:.4f} → "
            f"{cka_trajectory[-1]:.4f} (Δ={cka_improvement:+.4f}). "
            "Stacked correction exhausted systematic error."
        )
    elif cka_improvement > 0.01:
        verdict = (
            f"PARTIAL: CKA improved by {cka_improvement:+.4f} over "
            f"{len(rounds)} rounds but signal remains. "
            "More rounds may help."
        )
    elif cka_improvement > 0:
        verdict = (
            f"MARGINAL: CKA improved by {cka_improvement:+.4f}. "
            "Correction is working but slowly."
        )
    else:
        verdict = (
            f"FAILED: CKA did not improve ({cka_improvement:+.4f}). "
            "Corrective LoRA training is not effective."
        )
    results["verdict"] = verdict

    # Write results
    output_path = output_dir / "stacked_recovery.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results written to %s", output_path)

    # Pretty print
    print("\n" + "=" * 72)
    print("STACKED CORRECTIVE RECOVERY — SUMMARY")
    print("=" * 72)
    print(f"  Rounds completed:   {len(rounds)}")
    print(f"  Stop reason:        {stop_reason}")
    print()
    print(f"  CKA trajectory:     {' → '.join(f'{c:.4f}' for c in cka_trajectory)}")
    print(f"  Signal rank:        {' → '.join(f'{s:.1f}' for s in signal_rank_trajectory)}")
    print(f"  ||E||_F:            {' → '.join(f'{f:.6f}' for f in frob_trajectory)}")
    print()
    print(f"  CKA improvement:    {cka_improvement:+.4f}")
    print()
    print(f"  VERDICT: {verdict}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Stacked Corrective Recovery",
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
        help="Training dataset (JSONL)",
    )
    parser.add_argument(
        "--eval-dataset",
        default=DEFAULT_EVAL,
        help="Evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/stacked_corrective_recovery",
        help="Base output directory",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum stacking rounds",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Training iterations per round",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (small: 2 models in memory)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (incremented per round)",
    )
    parser.add_argument(
        "--n-cka-probes",
        type=int,
        default=30,
        help="Number of probes for CKA measurement",
    )
    parser.add_argument(
        "--functional-fraction",
        action="store_true",
        help="Compute functional error fraction (E projected onto activation subspace)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
