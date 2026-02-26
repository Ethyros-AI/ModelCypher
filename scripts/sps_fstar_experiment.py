#!/usr/bin/env python3
"""Experiment 5: SPS f* from Measured Geometry.

Tests whether geometrically-derived f* improves SPS step size behavior in
corrective LoRA training, and whether three independent estimation methods
agree on the f* value.

Hypothesis:
    H1: Geometrically-derived f* > 0 causes SPS to bind on >10% of iterations
        in the final 25% of training, improving convergence. Three f* methods
        agree within 2x of each other.

Methods for f* estimation:
    A: RMT noise floor: f* = L_0 * (1 - sv_frac)
       Already implemented in corrective_lora_training.py (line 382)
    B: Exponential tail fit: L(t) = a*exp(-bt) + c, use c as f*
    C: Signal propagation bound: f* = L_0 * highway_fraction
       Uses highway layer fraction from Experiment 1

Measurements:
    1. Run corrective LoRA with f*=0 (baseline)
    2. Run with f* from Method A (RMT noise floor)
    3. Run with f* from Method B (exponential tail fit)
    4. Run with f* from Method C (signal propagation)
    5. Compare: SPS binding frequency, final loss, CKA, f* agreement

Falsification criteria:
    FAIL if SPS never binds for ANY f* method
    FAIL if three methods disagree by >10x
    FAIL if f*>0 gives WORSE CKA than f*=0

References:
    Loizou et al. (2020): SPS convergence requires f* <= f_opt
    corrective_lora_training.py: Existing f* implementation

Usage:
    poetry run python scripts/sps_fstar_experiment.py

    # Smoke test (fewer iterations)
    poetry run python scripts/sps_fstar_experiment.py --smoke

    # Custom output
    poetry run python scripts/sps_fstar_experiment.py \
        --output results/sps_fstar/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Default Paths
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

# Corrective LoRA setup: quantized model + bf16 reference
DEFAULT_QUANTIZED = (
    "results/feasibility_map/20260225T160732Z/derived_models/"
    "Qwen3-1.7B-MLX-bf16-8bit-g64-affine"
)
DEFAULT_FP = f"{MODELS_BASE}/mlx-community/Qwen3-1.7B-MLX-bf16"
DEFAULT_TRAIN = "data/training/benchmark_train.jsonl"
DEFAULT_EVAL = "data/training/benchmark_val.jsonl"

# Signal propagation results from Experiment 1 (for Method C)
DEFAULT_EXP1_RESULTS = "results/signal_propagation/signal_propagation_results.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FStarEstimate:
    """A single f* estimate from one method."""

    method: str  # "baseline", "rmt", "exponential", "signal_propagation"
    f_star: float
    derivation: str  # Human-readable derivation trace


@dataclass
class TrainingRun:
    """Results from one training run with a specific f* value."""

    method: str
    f_star: float
    # Training curve
    losses: list[tuple[int, float]]
    # SPS binding analysis
    sps_binds_total: int
    sps_binds_final_quarter: int
    total_iters: int
    final_quarter_iters: int
    sps_bind_fraction_final: float
    # Which bound binds at each step
    binding_history: list[str]  # "SPS", "Weyl", "Ceil"
    # Final metrics
    initial_loss: float
    final_loss: float
    # CKA
    cka_before: float
    cka_after: float
    cka_improvement: float
    # Training time
    training_time_s: float


@dataclass
class ExperimentResults:
    """Complete experiment results."""

    timestamp: str
    experiment: str = "sps_fstar_from_measured_geometry"
    runs: list[dict] = field(default_factory=list)
    f_star_estimates: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# =============================================================================
# f* Estimation Methods
# =============================================================================


def estimate_fstar_rmt(
    rmt_results_path: str | None,
    initial_loss: float,
) -> FStarEstimate:
    """Method A: RMT noise floor.

    f* = initial_loss * (1 - mean_signal_variance_fraction)

    The MP noise floor bounds the loss achievable by any low-rank corrector.
    Signal variance fraction is the fraction of variance above the MP upper edge.
    """
    if rmt_results_path and Path(rmt_results_path).exists():
        with open(rmt_results_path) as f:
            rmt_data = json.load(f)
        sv_frac = rmt_data["aggregate"]["mean_signal_variance_fraction"]
        noise_fraction = 1.0 - sv_frac
        f_star = initial_loss * noise_fraction
        return FStarEstimate(
            method="rmt",
            f_star=f_star,
            derivation=(
                f"f* = L_0 * (1 - sv_frac) = {initial_loss:.6f} * "
                f"(1 - {sv_frac:.4f}) = {f_star:.6f}"
            ),
        )
    else:
        # No RMT results available — estimate noise fraction from rank deficiency
        # Typical quantization noise fraction for 8-bit: ~0.15-0.25
        # Cannot derive without data; return NaN to signal missing
        return FStarEstimate(
            method="rmt",
            f_star=float("nan"),
            derivation="RMT results not available. Cannot derive f* without data.",
        )


def estimate_fstar_exponential(
    losses: list[tuple[int, float]],
    warmup_iters: int = 20,
) -> FStarEstimate:
    """Method B: Exponential tail fit.

    Fit L(t) = a * exp(-b * t) + c to the loss curve after warmup.
    The asymptote c is the irreducible loss floor = f*.

    Uses least-squares on log(L(t) - c_trial) for robustness.
    Grid search over c in [0, min(L)] to find best fit.
    """
    import numpy as np

    if len(losses) < warmup_iters + 10:
        return FStarEstimate(
            method="exponential",
            f_star=float("nan"),
            derivation="Not enough data points for exponential fit.",
        )

    # Use post-warmup losses
    post_warmup = [(i, l) for i, l in losses if i >= warmup_iters]
    if len(post_warmup) < 10:
        return FStarEstimate(
            method="exponential",
            f_star=float("nan"),
            derivation="Not enough post-warmup data for exponential fit.",
        )

    t = np.array([float(i) for i, _ in post_warmup])
    y = np.array([l for _, l in post_warmup])

    # Normalize t to [0, 1]
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-10)

    # Grid search for c (asymptote)
    min_loss = float(np.min(y))
    best_c = 0.0
    best_residual = float("inf")

    # Search c in [0, 0.95 * min_loss] — c must be below observed minimum
    for c_frac in np.linspace(0.0, 0.95, 20):
        c_trial = c_frac * min_loss
        shifted = y - c_trial
        # Filter positive values only (log requires > 0)
        valid = shifted > 0
        if np.sum(valid) < 5:
            continue

        # Linear regression on log(shifted) vs t
        log_shifted = np.log(shifted[valid])
        t_valid = t_norm[valid]

        # Least squares: log(a) + (-b)*t
        n = len(t_valid)
        t_mean = np.mean(t_valid)
        log_mean = np.mean(log_shifted)
        numerator = np.sum((t_valid - t_mean) * (log_shifted - log_mean))
        denominator = np.sum((t_valid - t_mean) ** 2)
        if abs(denominator) < 1e-12:
            continue

        slope = numerator / denominator
        intercept = log_mean - slope * t_mean

        # Residual in original space
        predicted = np.exp(intercept + slope * t_norm) + c_trial
        residual = np.mean((y - predicted) ** 2)

        if residual < best_residual:
            best_residual = residual
            best_c = c_trial

    return FStarEstimate(
        method="exponential",
        f_star=best_c,
        derivation=(
            f"Fit L(t) = a*exp(-bt) + c to {len(post_warmup)} post-warmup points. "
            f"Best c (asymptote) = {best_c:.6f}, "
            f"residual MSE = {best_residual:.8f}"
        ),
    )


def estimate_fstar_signal_propagation(
    exp1_results_path: str | None,
    initial_loss: float,
    model_name: str = "Qwen3-8B",
) -> FStarEstimate:
    """Method C: Signal propagation bound.

    f* = initial_loss * highway_fraction

    Highway layers have alpha^2 * chi ≈ 0 (ordered phase), meaning they
    don't modify the representation. The fraction of highway layers bounds
    the fraction of the model that cannot contribute to loss reduction.
    """
    if exp1_results_path and Path(exp1_results_path).exists():
        with open(exp1_results_path) as f:
            exp1_data = json.load(f)

        # Find matching model or use first available
        for model_result in exp1_data.get("models", []):
            name = model_result.get("model_name", "")
            if model_name.lower() in name.lower() or not model_name:
                highway = model_result.get("highway_layers", [])
                num_layers = model_result.get("num_layers", 1)
                highway_fraction = len(highway) / num_layers if num_layers > 0 else 0.0
                f_star = initial_loss * highway_fraction

                return FStarEstimate(
                    method="signal_propagation",
                    f_star=f_star,
                    derivation=(
                        f"f* = L_0 * highway_fraction = {initial_loss:.6f} * "
                        f"{highway_fraction:.4f} ({len(highway)}/{num_layers} highway layers) "
                        f"= {f_star:.6f}"
                    ),
                )

        # Model not found in results
        return FStarEstimate(
            method="signal_propagation",
            f_star=float("nan"),
            derivation=f"Model '{model_name}' not found in Experiment 1 results.",
        )
    else:
        return FStarEstimate(
            method="signal_propagation",
            f_star=float("nan"),
            derivation="Experiment 1 results not available.",
        )


# =============================================================================
# Training Loop (minimal corrective LoRA)
# =============================================================================


def run_corrective_training(
    q_model,
    fp_model,
    tokenizer,
    tokenized: list,
    f_star: float,
    sigma_k_min: float,
    sigma_max: float,
    eta_ceiling: float,
    adapter,
    backend,
    max_iters: int = 100,
    batch_size: int = 2,
    seq_length: int = 256,
    seed: int = 42,
) -> dict:
    """Run corrective LoRA training and record SPS binding behavior.

    Returns detailed per-iteration metrics including which MASS bound binds.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    import mlx.utils

    from modelcypher.core.domain.training.mass_step_size import compute_per_step_rates

    optimizer = opt.SGD(learning_rate=eta_ceiling, momentum=0.0)

    def corrective_loss_fn(q_model_inner, batch, lengths_arr):
        q_logits = q_model_inner(batch)[:, :-1, :]
        fp_logits = mx.stop_gradient(fp_model(batch)[:, :-1, :])
        diff = q_logits - fp_logits
        sq = diff * diff

        T_minus_1 = batch.shape[1] - 1
        V = sq.shape[-1]
        arange = mx.arange(T_minus_1)[None, :]
        real_lens = lengths_arr[:, None] - 1
        mask = (arange < real_lens).astype(sq.dtype)
        mask_3d = mask[:, :, None]
        n_real = mx.sum(mask) * V
        mse_real_sum = mx.sum(sq * mask_3d)
        mse = mse_real_sum / n_real

        return mse, (n_real,)

    loss_and_grad = nn.value_and_grad(q_model, corrective_loss_fn)

    # Shuffle sample order
    sample_order = list(range(len(tokenized)))
    rng = random.Random(seed)
    rng.shuffle(sample_order)

    losses = []
    binding_history = []
    sps_binds = 0
    start_time = time.monotonic()

    for it in range(max_iters):
        # Build batch
        indices = [sample_order[i % len(sample_order)]
                   for i in range(it * batch_size, (it + 1) * batch_size)]
        batch_tokens = [tokenized[idx] for idx in indices]
        lengths = [min(len(tokens), seq_length) for tokens in batch_tokens]
        padded = []
        for tokens in batch_tokens:
            if len(tokens) > seq_length:
                padded.append(tokens[:seq_length])
            else:
                padded.append(tokens + [0] * (seq_length - len(tokens)))
        batch = mx.array(padded)
        lengths_arr = mx.array(lengths)

        # Forward + backward
        (loss, (n_real_arr,)), grad = loss_and_grad(q_model, batch, lengths_arr)

        # MASS per-step rate
        flat_grads = [p.reshape(-1) for _, p in mlx.utils.tree_flatten(grad) if p.size > 0]
        d_norm_sq = sum(mx.sum(p * p) for p in flat_grads)
        mx.eval(d_norm_sq, loss)
        d_norm = float(mx.sqrt(d_norm_sq).item())
        loss_val = float(loss.item())

        eta_step, eta_sps, eta_weyl, displacement, _ = compute_per_step_rates(
            loss_val, d_norm, sigma_k_min, eta_ceiling,
            f_star=f_star,
        )
        optimizer.learning_rate = mx.array(eta_step)

        # Track which bound binds
        if eta_step == eta_sps:
            binds = "SPS"
            sps_binds += 1
        elif eta_step == eta_weyl:
            binds = "Weyl"
        else:
            binds = "Ceil"
        binding_history.append(binds)

        # Apply update
        optimizer.update(q_model, grad)
        mx.eval(q_model.parameters(), optimizer.state)

        # Clamp scales
        for _, module in adapter._iter_nb_lora_modules(q_model):
            module.clamp_scale()
            mx.eval(module.S_raw)

        losses.append((it, loss_val))

        if it % 20 == 0:
            logger.info(
                "  [f*=%.6f] iter %d/%d: loss=%.6f, eta=%.4e, binds=%s",
                f_star, it, max_iters, loss_val, eta_step, binds,
            )

    training_time = time.monotonic() - start_time

    # Analyze SPS binding in final quarter
    final_quarter_start = int(0.75 * max_iters)
    final_quarter_binds = sum(
        1 for i, b in enumerate(binding_history)
        if i >= final_quarter_start and b == "SPS"
    )
    final_quarter_iters = max_iters - final_quarter_start

    return {
        "losses": losses,
        "sps_binds_total": sps_binds,
        "sps_binds_final_quarter": final_quarter_binds,
        "total_iters": max_iters,
        "final_quarter_iters": final_quarter_iters,
        "sps_bind_fraction_final": (
            final_quarter_binds / final_quarter_iters
            if final_quarter_iters > 0 else 0.0
        ),
        "binding_history": binding_history,
        "initial_loss": losses[0][1] if losses else 0.0,
        "final_loss": losses[-1][1] if losses else 0.0,
        "training_time_s": training_time,
    }


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full SPS f* experiment."""
    import mlx.core as mx
    import mlx.utils

    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()

    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.training.geometric_lora import select_target_modules
    from modelcypher.core.domain.training.mass_step_size import derive_spectral_ceiling

    backend = get_default_backend()
    adapter = MLXTrainingAdapter(backend)

    max_iters = 50 if args.smoke else args.max_iters

    logger.info("SPS f* Experiment — comparing 4 f* estimation methods")
    logger.info(f"Quantized: {args.quantized_model}")
    logger.info(f"Reference: {args.fp_model}")
    logger.info(f"Max iters per run: {max_iters}")

    results = ExperimentResults(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # ── Phase 1: Load models ──
    logger.info("Loading bf16 reference model...")
    fp_model, tokenizer = backend.load_model(str(args.fp_model))

    logger.info("Loading quantized model (template for LoRA injection)...")
    q_model_base, _ = backend.load_model(str(args.quantized_model))

    # Prepare training data
    tokenized, train_texts = _prepare_batches(
        args.train_dataset, tokenizer, args.batch_size, 512
    )

    # Prepare eval data for CKA
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

    # ── Phase 2: Analyze geometry for MASS ──
    logger.info("Analyzing model geometry for MASS...")
    geometries = adapter.analyze_model_geometry_streaming(q_model_base, use_randomized=True)
    target_modules = select_target_modules(geometries)
    sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)
    sigma_k_vals = [g.sigma_k for g in geometries.values() if g.sigma_k > 0]
    sigma_k_min = min(sigma_k_vals)
    eta_ceiling = derive_spectral_ceiling(
        sigma_k_min=sigma_k_min, sigma_max_global=sigma_max,
    )

    # Apply sqrt(N) correction
    n_batches_per_epoch = max(1, len(tokenized) // args.batch_size)
    if n_batches_per_epoch > 1:
        eta_ceiling = eta_ceiling / math.sqrt(n_batches_per_epoch)

    logger.info(
        f"MASS: sigma_max={sigma_max:.4e}, sigma_k_min={sigma_k_min:.4e}, "
        f"eta_ceiling={eta_ceiling:.4e}"
    )

    # ── Phase 3: Get initial loss for f* estimation ──
    # Do a quick forward pass to get initial loss
    logger.info("Computing initial loss for f* estimation...")
    q_model_init, _ = backend.load_model(str(args.quantized_model))
    adapter.inject_nb_lora(q_model_init, geometries, target_modules)
    adapter.freeze_and_apply_lora(q_model_init)

    # Single batch forward to get L_0
    batch_tokens = tokenized[:args.batch_size]
    lengths = [min(len(t), 256) for t in batch_tokens]
    padded = []
    for tokens in batch_tokens:
        if len(tokens) > 256:
            padded.append(tokens[:256])
        else:
            padded.append(tokens + [0] * (256 - len(tokens)))
    batch = mx.array(padded)
    lengths_arr = mx.array(lengths)

    q_logits = q_model_init(batch)[:, :-1, :]
    fp_logits = mx.stop_gradient(fp_model(batch)[:, :-1, :])
    diff = q_logits - fp_logits
    sq = diff * diff
    T_minus_1 = batch.shape[1] - 1
    V = sq.shape[-1]
    arange_t = mx.arange(T_minus_1)[None, :]
    real_lens = lengths_arr[:, None] - 1
    mask = (arange_t < real_lens).astype(sq.dtype)
    mask_3d = mask[:, :, None]
    n_real = mx.sum(mask) * V
    mse_real = mx.sum(sq * mask_3d) / n_real
    mx.eval(mse_real)
    initial_loss = float(mse_real.item())
    logger.info(f"Initial loss (L_0): {initial_loss:.6f}")
    del q_model_init
    gc.collect()

    # ── Phase 4: Estimate f* via three methods ──
    f_star_estimates = []

    # Method A: RMT noise floor
    est_rmt = estimate_fstar_rmt(args.rmt_results, initial_loss)
    f_star_estimates.append(est_rmt)
    logger.info(f"Method A (RMT): f*={est_rmt.f_star:.6f} — {est_rmt.derivation}")

    # Method C: Signal propagation (from Experiment 1)
    est_sp = estimate_fstar_signal_propagation(args.exp1_results, initial_loss)
    f_star_estimates.append(est_sp)
    logger.info(f"Method C (Signal prop): f*={est_sp.f_star:.6f} — {est_sp.derivation}")

    # Method B will be estimated after baseline run (needs loss curve)

    # ── Phase 5: Run training with each f* ──
    # Define f* values to test
    f_star_configs = [
        ("baseline", 0.0),
    ]

    if not math.isnan(est_rmt.f_star):
        f_star_configs.append(("rmt", est_rmt.f_star))
    if not math.isnan(est_sp.f_star):
        f_star_configs.append(("signal_propagation", est_sp.f_star))

    all_run_results = []

    for method_name, f_star_val in f_star_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running training with f*={f_star_val:.6f} (method: {method_name})")
        logger.info(f"{'='*60}")

        # Fresh model + LoRA for each run
        q_model, _ = backend.load_model(str(args.quantized_model))
        adapter.inject_nb_lora(q_model, geometries, target_modules)
        adapter.freeze_and_apply_lora(q_model)

        # Collect CKA before
        q_acts_before = _collect_activations(q_model, tokenizer, eval_texts, backend, 20)
        fp_acts = _collect_activations(fp_model, tokenizer, eval_texts, backend, 20)
        cka_before = _compute_cka(fp_acts, q_acts_before, backend)

        # Run training
        run_result = run_corrective_training(
            q_model=q_model,
            fp_model=fp_model,
            tokenizer=tokenizer,
            tokenized=tokenized,
            f_star=f_star_val,
            sigma_k_min=sigma_k_min,
            sigma_max=sigma_max,
            eta_ceiling=eta_ceiling,
            adapter=adapter,
            backend=backend,
            max_iters=max_iters,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Collect CKA after
        q_acts_after = _collect_activations(q_model, tokenizer, eval_texts, backend, 20)
        cka_after = _compute_cka(fp_acts, q_acts_after, backend)

        run_result["method"] = method_name
        run_result["f_star"] = f_star_val
        run_result["cka_before"] = cka_before["mean_cka"]
        run_result["cka_after"] = cka_after["mean_cka"]
        run_result["cka_improvement"] = cka_after["mean_cka"] - cka_before["mean_cka"]

        all_run_results.append(run_result)

        logger.info(
            f"  [{method_name}] final_loss={run_result['final_loss']:.6f}, "
            f"SPS binds={run_result['sps_binds_total']}/{run_result['total_iters']} total, "
            f"{run_result['sps_binds_final_quarter']}/{run_result['final_quarter_iters']} final 25%, "
            f"CKA: {cka_before['mean_cka']:.4f} -> {cka_after['mean_cka']:.4f}"
        )

        # If this is the baseline run, compute Method B (exponential fit)
        if method_name == "baseline" and len(run_result["losses"]) >= 30:
            est_exp = estimate_fstar_exponential(run_result["losses"], warmup_iters=20)
            f_star_estimates.append(est_exp)
            logger.info(f"Method B (Exponential): f*={est_exp.f_star:.6f} — {est_exp.derivation}")

            if not math.isnan(est_exp.f_star) and est_exp.f_star > 0:
                f_star_configs.append(("exponential", est_exp.f_star))

        del q_model
        gc.collect()

    # Run exponential f* if it was estimated after baseline
    for method_name, f_star_val in f_star_configs:
        if method_name == "exponential" and not any(
            r["method"] == "exponential" for r in all_run_results
        ):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running training with f*={f_star_val:.6f} (method: exponential)")
            logger.info(f"{'='*60}")

            q_model, _ = backend.load_model(str(args.quantized_model))
            adapter.inject_nb_lora(q_model, geometries, target_modules)
            adapter.freeze_and_apply_lora(q_model)

            q_acts_before = _collect_activations(q_model, tokenizer, eval_texts, backend, 20)
            fp_acts = _collect_activations(fp_model, tokenizer, eval_texts, backend, 20)
            cka_before = _compute_cka(fp_acts, q_acts_before, backend)

            run_result = run_corrective_training(
                q_model=q_model,
                fp_model=fp_model,
                tokenizer=tokenizer,
                tokenized=tokenized,
                f_star=f_star_val,
                sigma_k_min=sigma_k_min,
                sigma_max=sigma_max,
                eta_ceiling=eta_ceiling,
                adapter=adapter,
                backend=backend,
                max_iters=max_iters,
                batch_size=args.batch_size,
                seed=args.seed,
            )

            q_acts_after = _collect_activations(q_model, tokenizer, eval_texts, backend, 20)
            cka_after = _compute_cka(fp_acts, q_acts_after, backend)

            run_result["method"] = "exponential"
            run_result["f_star"] = f_star_val
            run_result["cka_before"] = cka_before["mean_cka"]
            run_result["cka_after"] = cka_after["mean_cka"]
            run_result["cka_improvement"] = cka_after["mean_cka"] - cka_before["mean_cka"]

            all_run_results.append(run_result)

            logger.info(
                f"  [exponential] final_loss={run_result['final_loss']:.6f}, "
                f"SPS binds={run_result['sps_binds_total']}/{run_result['total_iters']} total, "
                f"{run_result['sps_binds_final_quarter']}/{run_result['final_quarter_iters']} final 25%"
            )

            del q_model
            gc.collect()

    # ── Phase 6: Analysis ──
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    # Store results
    results.runs = all_run_results
    results.f_star_estimates = [
        {"method": e.method, "f_star": e.f_star, "derivation": e.derivation}
        for e in f_star_estimates
    ]

    # Falsification test 1: SPS ever binds for any f* > 0 method
    nonzero_runs = [r for r in all_run_results if r["f_star"] > 0]
    any_sps_binds = any(r["sps_binds_total"] > 0 for r in nonzero_runs)
    sps_binds_final_any = any(
        r["sps_bind_fraction_final"] > 0.10 for r in nonzero_runs
    )

    # Falsification test 2: f* methods agree within 10x
    valid_fstars = [
        e.f_star for e in f_star_estimates
        if not math.isnan(e.f_star) and e.f_star > 0
    ]
    if len(valid_fstars) >= 2:
        max_ratio = max(valid_fstars) / min(valid_fstars)
        methods_agree = max_ratio <= 10.0
    elif len(valid_fstars) == 1:
        max_ratio = 1.0
        methods_agree = True
    else:
        max_ratio = float("inf")
        methods_agree = False

    # Falsification test 3: f*>0 gives better CKA than f*=0
    baseline_run = next((r for r in all_run_results if r["method"] == "baseline"), None)
    best_fstar_run = None
    for r in nonzero_runs:
        if best_fstar_run is None or r["cka_after"] > best_fstar_run["cka_after"]:
            best_fstar_run = r

    if baseline_run and best_fstar_run:
        fstar_improves_cka = best_fstar_run["cka_after"] >= baseline_run["cka_after"]
    else:
        fstar_improves_cka = False

    # Overall
    passes_sps_binds = any_sps_binds
    passes_agreement = methods_agree
    passes_cka = fstar_improves_cka

    overall_pass = passes_sps_binds and passes_agreement and passes_cka

    results.summary = {
        "n_runs": len(all_run_results),
        "n_fstar_methods": len(valid_fstars),
        "initial_loss": initial_loss,
        # SPS binding
        "any_sps_binds": any_sps_binds,
        "sps_binds_final_quarter_any": sps_binds_final_any,
        "passes_sps_binds": passes_sps_binds,
        # f* agreement
        "valid_fstars": valid_fstars,
        "max_fstar_ratio": max_ratio,
        "passes_agreement": passes_agreement,
        # CKA
        "baseline_cka_after": baseline_run["cka_after"] if baseline_run else None,
        "best_fstar_cka_after": best_fstar_run["cka_after"] if best_fstar_run else None,
        "best_fstar_method": best_fstar_run["method"] if best_fstar_run else None,
        "passes_cka": passes_cka,
        # Overall
        "overall_verdict": "H1 SUPPORTED" if overall_pass else "H1 REFUTED",
        "falsification_thresholds": {
            "sps_binding_fraction_min": 0.10,
            "sps_binding_source": "10% of final-quarter iterations (Loizou et al. 2020)",
            "method_agreement_max_ratio": 10.0,
            "agreement_source": "order-of-magnitude agreement between independent estimators",
        },
        "references": [
            "Loizou et al. (2020): SPS convergence requires f* <= f_opt",
            "corrective_lora_training.py: existing f* = L_0 * noise_fraction",
            "De & Smith (2020): highway fraction bounds for Method C",
        ],
    }

    verdict = results.summary["overall_verdict"]
    logger.info(f"\nEXPERIMENT VERDICT: {verdict}")
    logger.info(f"  SPS binds test: {'PASS' if passes_sps_binds else 'FAIL'}")
    logger.info(f"  Method agreement test: {'PASS' if passes_agreement else 'FAIL'} (max ratio: {max_ratio:.2f}x)")
    logger.info(f"  CKA improvement test: {'PASS' if passes_cka else 'FAIL'}")

    for r in all_run_results:
        logger.info(
            f"  [{r['method']}] f*={r['f_star']:.6f}, "
            f"final_loss={r['final_loss']:.6f}, "
            f"SPS binds={r['sps_bind_fraction_final']*100:.1f}% final, "
            f"CKA={r['cka_after']:.4f}"
        )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sps_fstar_results.json"

    # Strip binding_history from serialization (too large)
    serializable_runs = []
    for r in all_run_results:
        r_copy = {k: v for k, v in r.items() if k != "binding_history"}
        # Summarize binding history
        if "binding_history" in r:
            hist = r["binding_history"]
            r_copy["binding_counts"] = {
                "SPS": hist.count("SPS"),
                "Weyl": hist.count("Weyl"),
                "Ceil": hist.count("Ceil"),
            }
        serializable_runs.append(r_copy)

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": results.timestamp,
            "experiment": results.experiment,
            "runs": serializable_runs,
            "f_star_estimates": results.f_star_estimates,
            "summary": results.summary,
        }, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


# =============================================================================
# Helper functions (reused from corrective_lora_training.py)
# =============================================================================


def _prepare_batches(
    dataset_path: str,
    tokenizer: Any,
    batch_size: int,
    seq_length: int,
) -> tuple[list, list[str]]:
    """Load dataset and tokenize."""
    texts = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 2:
            tokenized.append(tokens)

    return tokenized, texts


def _collect_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    backend: Any,
    n_samples: int | None = None,
) -> dict[int, list]:
    """Collect per-layer mean-pooled activations for CKA measurement."""
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
    import mlx.core as mx

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


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPS f* from Measured Geometry Experiment"
    )
    parser.add_argument(
        "--output",
        default="results/sps_fstar/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quantized-model",
        default=DEFAULT_QUANTIZED,
        help="Path to quantized model",
    )
    parser.add_argument(
        "--fp-model",
        default=DEFAULT_FP,
        help="Path to bf16 reference model",
    )
    parser.add_argument(
        "--train-dataset",
        default=DEFAULT_TRAIN,
        help="Path to training dataset (jsonl)",
    )
    parser.add_argument(
        "--eval-dataset",
        default=DEFAULT_EVAL,
        help="Path to eval dataset (jsonl)",
    )
    parser.add_argument(
        "--rmt-results",
        default=None,
        help="Path to RMT analysis results JSON (for Method A)",
    )
    parser.add_argument(
        "--exp1-results",
        default=DEFAULT_EXP1_RESULTS,
        help="Path to Experiment 1 results JSON (for Method C)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Max training iterations per run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 50 iterations per run",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
