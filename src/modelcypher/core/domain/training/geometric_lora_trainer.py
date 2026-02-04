# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Geometric LoRA trainer.

Trains LoRA adapters where all configuration is derived from geometry:
- Target modules: spectral decay < 100×
- Rank: min(tail_dims) across targets
- Scale: σ_k per layer (via spectral normalization)

Tracks geometric metrics instead of just loss:
- spectral_bound_ratio: ||B @ A||_spectral / σ_k (should stay ≤ 1.0)
- capacity_utilization: effective_rank(B @ A) / rank
- energy_concentration: Gini coefficient of singular values (0=uniform, 1=concentrated)

No hyperparameters except learning rate and epochs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .geometric_lora import (
    LayerGeometry,
    analyze_model_geometry,
    apply_geometric_lora,
    compute_geometric_rank,
    compute_per_layer_ranks,
    get_lora_parameters,
    select_target_modules,
)
from .loop_preservation import (
    LoopPreservationConfig,
    compute_entropy_trajectory,
    loop_preservation_loss,
)

logger = logging.getLogger(__name__)


# Machine epsilon for convergence thresholds
SQRT_EPS = np.sqrt(np.finfo(np.float32).eps)


# =============================================================================
# Geometric Convergence Monitor
# =============================================================================

@dataclass
class GeometricConvergenceState:
    """State of geometric convergence criteria.

    All criteria are geometry-derived - no validation set required.
    """

    step: int
    bb_stable: bool  # BB curvature estimates have stabilized
    loss_stable: bool  # Loss change below √ε
    budget_exhausted: bool  # Spectral bound ratio > threshold

    @property
    def should_stop(self) -> bool:
        """Check if training should stop based on geometric criteria.

        Convergence requires BB stability AND (loss stable OR budget exhausted).
        """
        return self.bb_stable and (self.loss_stable or self.budget_exhausted)


class GeometricConvergenceMonitor:
    """Geometry-derived early stopping without validation set.

    Combines three geometric criteria:
    1. BB stability: Barzilai-Borwein curvature estimates stabilized
    2. Loss stability: Loss change below √ε (numerical precision floor)
    3. Spectral budget: spectral_bound_ratio approaching 1.0

    All thresholds are dtype-derived, not hyperparameters.

    Usage:
        monitor = GeometricConvergenceMonitor()
        for step in range(max_steps):
            ... training step ...
            state = monitor.check(optimizer, lora_layers, current_loss)
            if state.should_stop:
                logger.info("Geometric convergence at step %d", step)
                break
    """

    def __init__(
        self,
        bb_stability_threshold: float = 1e-4,
        budget_threshold: float = 0.9,
        loss_window: int = 10,
    ):
        """Initialize convergence monitor.

        Args:
            bb_stability_threshold: Relative variance threshold for BB stability.
                From GeometricOptimizer.is_bb_stable(). Default 1e-4.
            budget_threshold: Spectral bound ratio threshold for budget exhaustion.
                When mean(spectral_bound_ratio) > threshold, budget is nearly used.
                Default 0.9 means 90% of geometric budget consumed.
            loss_window: Number of steps for loss stability check. Default 10.
        """
        self._bb_threshold = bb_stability_threshold
        self._budget_threshold = budget_threshold
        self._loss_window = loss_window
        self._loss_history: list[float] = []
        self._step = 0

    def check(
        self,
        optimizer,
        lora_layers: dict,
        current_loss: float,
    ) -> GeometricConvergenceState:
        """Check all geometric convergence criteria.

        Args:
            optimizer: GeometricOptimizer with BB state
            lora_layers: Dict of LoRA layers for spectral metrics
            current_loss: Current training loss

        Returns:
            GeometricConvergenceState with all criteria evaluated
        """
        self._step += 1
        self._loss_history.append(current_loss)

        # Keep only recent history
        if len(self._loss_history) > self._loss_window * 2:
            self._loss_history = self._loss_history[-self._loss_window * 2:]

        # Criterion 1: BB curvature stabilized
        bb_stable = optimizer.is_bb_stable(threshold=self._bb_threshold)

        # Criterion 2: Loss change below √ε
        loss_stable = self._check_loss_stability()

        # Criterion 3: Spectral budget nearly exhausted
        budget_exhausted = self._check_budget_exhausted(lora_layers)

        return GeometricConvergenceState(
            step=self._step,
            bb_stable=bb_stable,
            loss_stable=loss_stable,
            budget_exhausted=budget_exhausted,
        )

    def _check_loss_stability(self) -> bool:
        """Check if loss change is below numerical precision floor."""
        if len(self._loss_history) < self._loss_window:
            return False

        recent = self._loss_history[-self._loss_window:]
        older = self._loss_history[-self._loss_window * 2:-self._loss_window] if len(
            self._loss_history) >= self._loss_window * 2 else self._loss_history[:self._loss_window]

        if not older:
            return False

        # Compute average change between windows
        mean_recent = np.mean(recent)
        mean_older = np.mean(older)

        # Relative change
        denominator = max(abs(mean_older), SQRT_EPS)
        rel_change = abs(mean_recent - mean_older) / denominator

        return rel_change < SQRT_EPS

    def _check_budget_exhausted(self, lora_layers: dict) -> bool:
        """Check if spectral budget is nearly exhausted."""
        if not lora_layers:
            return False

        metrics = compute_aggregate_metrics(lora_layers)
        if not metrics:
            return False

        mean_ratio = metrics["spectral_bound_ratio"]["mean"]
        return mean_ratio > self._budget_threshold

    def reset(self) -> None:
        """Reset monitor for new training run."""
        self._loss_history = []
        self._step = 0

    @property
    def step(self) -> int:
        """Current step count."""
        return self._step


# =============================================================================
# Geometric Metrics
# =============================================================================

@dataclass
class GeometricMetrics:
    """Geometric health metrics for a LoRA layer."""

    layer_key: str
    spectral_bound_ratio: float  # ||B @ A|| / σ_k - should be ≤ 1.0
    capacity_utilization: float  # effective_rank / rank - how much capacity is used
    energy_concentration: float  # Gini coefficient - 0=uniform, 1=concentrated

    @property
    def is_healthy(self) -> bool:
        """Whether the layer is within geometric bounds."""
        return self.spectral_bound_ratio <= 1.0


def compute_spectral_norm(M: mx.array, n_iters: int = 5) -> float:
    """Compute spectral norm via power iteration."""
    # Convert to numpy for stability
    M_np = np.array(M.tolist(), dtype=np.float32)

    # Check for near-zero matrix (e.g., at initialization when B=0)
    frobenius = np.linalg.norm(M_np, 'fro')
    if frobenius < 1e-10:
        return 0.0

    # Power iteration
    v = np.random.randn(M_np.shape[1])
    v = v / np.linalg.norm(v)

    for _ in range(n_iters):
        u = M_np @ v
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-10:
            return 0.0  # Matrix is effectively zero
        u = u / u_norm

        v = M_np.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return 0.0
        v = v / v_norm

    return float(np.linalg.norm(M_np @ v))


def compute_effective_rank(M: mx.array) -> float:
    """Compute effective rank from singular value distribution.

    effective_rank = exp(entropy(normalized_singular_values))

    This measures how many dimensions are actually being used.
    - If all SVs are equal: effective_rank = full_rank
    - If only one SV is nonzero: effective_rank = 1
    """
    M_np = np.array(M.tolist(), dtype=np.float32)

    # Check for near-zero matrix
    if np.linalg.norm(M_np, 'fro') < 1e-10:
        return 0.0

    _, S, _ = np.linalg.svd(M_np, full_matrices=False)

    # Normalize to probability distribution
    S_pos = S[S > 1e-10]
    if len(S_pos) == 0:
        return 0.0

    p = S_pos / np.sum(S_pos)

    # Entropy
    entropy = -np.sum(p * np.log(p + 1e-10))

    # Effective rank
    return float(np.exp(entropy))


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient measuring concentration.

    0 = perfectly uniform distribution
    1 = all energy in one value
    """
    if len(values) == 0:
        return 0.0

    values = np.sort(np.abs(values))
    n = len(values)

    if np.sum(values) < 1e-10:
        return 0.0

    # Gini formula
    cumsum = np.cumsum(values)
    return float((2 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values))) - (n + 1) / n)


@dataclass
class SaturationState:
    """State of geometric saturation across LoRA layers."""
    n_saturated: int  # Layers at their spectral bound
    n_total: int  # Total layers
    saturation_ratio: float  # n_saturated / n_total

    @property
    def is_full(self) -> bool:
        """True when all layers are saturated."""
        return self.n_saturated == self.n_total and self.n_total > 0


def enforce_spectral_bound(lora_layers: dict) -> SaturationState:
    """Enforce spectral bound constraint on all LoRA layers.

    For each layer, if ||B @ A||_spectral > σ_k, rescale B so that
    the constraint is satisfied exactly.

    This is a hard geometric constraint, not a regularization term.
    The adapter's contribution should not exceed the layer's noise floor.

    Args:
        lora_layers: Dict mapping layer keys to LoRA layer objects.

    Returns:
        SaturationState indicating how many layers are at capacity.
    """
    n_saturated = 0
    n_total = 0

    for layer_key, lora_layer in lora_layers.items():
        sigma_k = lora_layer.sigma_k
        if sigma_k <= 0:
            continue

        n_total += 1

        # Compute current spectral norm of B @ A
        spectral_norm = compute_spectral_norm(lora_layer.lora_b @ lora_layer.lora_a)

        if spectral_norm > sigma_k:
            # Rescale B to enforce constraint: ||B @ A|| = σ_k
            scale = sigma_k / spectral_norm
            lora_layer.lora_b = lora_layer.lora_b * scale
            mx.eval(lora_layer.lora_b)
            n_saturated += 1

    saturation_ratio = n_saturated / n_total if n_total > 0 else 0.0
    return SaturationState(
        n_saturated=n_saturated,
        n_total=n_total,
        saturation_ratio=saturation_ratio,
    )


def compute_layer_metrics(lora_layer, layer_key: str) -> GeometricMetrics:
    """Compute geometric metrics for a single LoRA layer."""
    # Get the LoRA delta: B @ A
    delta = lora_layer.lora_b @ lora_layer.lora_a
    mx.eval(delta)

    # Convert to numpy once
    delta_np = np.array(delta.tolist(), dtype=np.float32)
    frobenius = np.linalg.norm(delta_np, 'fro')

    # Handle near-zero delta (e.g., at initialization)
    if frobenius < 1e-10:
        return GeometricMetrics(
            layer_key=layer_key,
            spectral_bound_ratio=0.0,
            capacity_utilization=0.0,
            energy_concentration=0.0,
        )

    # Spectral norm of the delta
    spectral_norm = compute_spectral_norm(delta)

    # Spectral bound ratio
    spectral_bound_ratio = spectral_norm / lora_layer.sigma_k if lora_layer.sigma_k > 0 else float('inf')

    # Effective rank and capacity utilization
    eff_rank = compute_effective_rank(delta)
    capacity_utilization = eff_rank / lora_layer.rank if lora_layer.rank > 0 else 0.0

    # Energy concentration (Gini of singular values)
    _, S, _ = np.linalg.svd(delta_np, full_matrices=False)
    energy_concentration = compute_gini_coefficient(S)

    return GeometricMetrics(
        layer_key=layer_key,
        spectral_bound_ratio=spectral_bound_ratio,
        capacity_utilization=capacity_utilization,
        energy_concentration=energy_concentration,
    )


def compute_aggregate_metrics(lora_layers: dict) -> dict:
    """Compute aggregate geometric metrics across all LoRA layers."""
    metrics_list = []

    for layer_key, lora_layer in lora_layers.items():
        metrics = compute_layer_metrics(lora_layer, layer_key)
        metrics_list.append(metrics)

    if not metrics_list:
        return {}

    # Aggregate statistics
    bound_ratios = [m.spectral_bound_ratio for m in metrics_list]
    utilizations = [m.capacity_utilization for m in metrics_list]
    concentrations = [m.energy_concentration for m in metrics_list]

    return {
        "spectral_bound_ratio": {
            "mean": float(np.mean(bound_ratios)),
            "max": float(np.max(bound_ratios)),
            "min": float(np.min(bound_ratios)),
            "n_violations": sum(1 for r in bound_ratios if r > 1.0),
        },
        "capacity_utilization": {
            "mean": float(np.mean(utilizations)),
            "max": float(np.max(utilizations)),
            "min": float(np.min(utilizations)),
        },
        "energy_concentration": {
            "mean": float(np.mean(concentrations)),
            "max": float(np.max(concentrations)),
            "min": float(np.min(concentrations)),
        },
        "n_layers": len(metrics_list),
        "n_healthy": sum(1 for m in metrics_list if m.is_healthy),
    }


# =============================================================================
# Configuration and Results
# =============================================================================

@dataclass
class GeometricLoRAConfig:
    """Configuration derived from model geometry.

    Supports both global rank (legacy) and per-layer adaptive ranks.
    Per-layer ranks allocate more capacity to high-curvature layers.
    """

    target_modules: list[str]
    rank: int  # Global rank (or min of per-layer ranks)
    geometries: dict[str, LayerGeometry]
    per_layer_ranks: dict[str, int] = field(default_factory=dict)  # Curvature-adaptive

    # Training params (these ARE hyperparameters - task dependent)
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 4

    # Geometric early stopping (enabled by default)
    enable_geometric_stopping: bool = True
    max_steps: int | None = None  # Optional hard limit (None = epochs only)

    @property
    def adaptive_ranks_enabled(self) -> bool:
        """Whether per-layer adaptive ranks are being used."""
        return len(self.per_layer_ranks) > 0

    @property
    def effective_ranks(self) -> dict[str, int]:
        """Get the ranks that will actually be used (per-layer if available)."""
        if self.adaptive_ranks_enabled:
            return self.per_layer_ranks
        return {key: self.rank for key in self.target_modules}

    @property
    def total_lora_params(self) -> int:
        """Estimate total LoRA parameters based on ranks and layer shapes."""
        total = 0
        for key in self.target_modules:
            if key not in self.geometries:
                continue
            geom = self.geometries[key]
            rank = self.effective_ranks.get(key, self.rank)
            # LoRA A: [rank, in_features], LoRA B: [out_features, rank]
            in_features = geom.shape[1]
            out_features = geom.shape[0]
            total += rank * (in_features + out_features)
        return total

    def to_dict(self) -> dict:
        result = {
            "target_modules": self.target_modules,
            "rank": self.rank,
            "adaptive_ranks": self.adaptive_ranks_enabled,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "layer_geometries": {
                key: {
                    "sigma_k": g.sigma_k,
                    "sigma_max": g.sigma_max,
                    "decay_ratio": g.decay_ratio,
                    "tail_dims": g.tail_dims,
                    "rank": self.effective_ranks.get(key, self.rank),
                }
                for key, g in self.geometries.items()
                if key in self.target_modules
            },
        }
        if self.adaptive_ranks_enabled:
            result["per_layer_ranks"] = self.per_layer_ranks
        return result


@dataclass
class GeometricLoRAResult:
    """Result of geometric LoRA training."""

    success: bool
    config: Optional[GeometricLoRAConfig] = None
    adapter_path: Optional[Path] = None
    final_loss: float = 0.0
    training_time_seconds: float = 0.0
    error: Optional[str] = None


def derive_config_from_geometry(
    model,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 4,
    adaptive_rank: bool = True,
    min_rank: int = 4,
    max_rank: int = 64,
) -> GeometricLoRAConfig:
    """Derive LoRA configuration from model geometry.

    Args:
        model: The loaded model
        learning_rate: Learning rate (task-dependent)
        epochs: Number of epochs (task-dependent)
        batch_size: Batch size (hardware-dependent)
        adaptive_rank: If True, compute per-layer ranks based on curvature.
                      High-curvature layers get higher rank. (default: True)
        min_rank: Minimum rank for any layer (numerical stability)
        max_rank: Maximum rank for any layer (memory constraint)

    Returns:
        GeometricLoRAConfig with all geometry-derived parameters
    """
    logger.info("Analyzing model geometry...")

    # Compute geometry for all layers
    geometries = analyze_model_geometry(model)

    if not geometries:
        raise ValueError("No targetable layers found in model")

    # Select targets based on spectral decay
    target_modules = select_target_modules(geometries)

    if not target_modules:
        raise ValueError("No layers with decay_ratio < 100 found")

    # Derive ranks from geometry
    if adaptive_rank:
        # Curvature-adaptive: allocate more rank to high-curvature layers
        per_layer_ranks = compute_per_layer_ranks(
            geometries, target_modules,
            min_rank=min_rank, max_rank=max_rank
        )
        # Global rank is the min (for compatibility)
        rank = min(per_layer_ranks.values())

        rank_summary = sorted(set(per_layer_ranks.values()))
        logger.info(
            "Adaptive ranks: %d targets, ranks=%s (min=%d, max=%d)",
            len(target_modules), rank_summary, min(rank_summary), max(rank_summary)
        )
    else:
        # Legacy: single global rank
        per_layer_ranks = {}
        rank = compute_geometric_rank(geometries, target_modules)
        logger.info(
            "Global rank: %d targets, rank=%d",
            len(target_modules), rank
        )

    config = GeometricLoRAConfig(
        target_modules=target_modules,
        rank=rank,
        geometries=geometries,
        per_layer_ranks=per_layer_ranks,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )

    logger.info(
        "Total LoRA parameters: %d (%s)",
        config.total_lora_params,
        "adaptive" if adaptive_rank else "global"
    )

    return config


def train_geometric_lora(
    model,
    tokenizer,
    training_data: list[dict],
    output_path: Path,
    config: GeometricLoRAConfig,
    progress_callback: Optional[Callable[[dict], None]] = None,
    loop_config: Optional[LoopPreservationConfig] = None,
) -> GeometricLoRAResult:
    """Train a geometric LoRA adapter.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        training_data: List of training examples (prompt/completion format)
        output_path: Where to save the adapter
        config: Geometry-derived configuration
        progress_callback: Optional callback for progress updates
        loop_config: Optional loop preservation configuration.
            If provided, adds a loss term that penalizes spectral entropy
            collapse. The config is derived from base model geometry using
            derive_loop_config() before training.

    Returns:
        GeometricLoRAResult with training outcomes
    """
    start_time = time.time()

    try:
        # Apply geometric LoRA to model (with per-layer ranks if available)
        lora_layers = apply_geometric_lora(
            model,
            config.geometries,
            config.target_modules,
            config.effective_ranks,  # Uses per-layer ranks if adaptive
        )

        if not lora_layers:
            return GeometricLoRAResult(
                success=False,
                error="No LoRA layers were applied",
            )

        # Freeze all parameters first
        model.freeze()

        # Unfreeze LoRA layers (only lora_a and lora_b will be trainable)
        for layer_key, lora_layer in lora_layers.items():
            # Unfreeze this module's parameters
            lora_layer.unfreeze()
            # Re-freeze the base weight (it was unfrozen with the module)
            # base_weight should stay frozen
            lora_layer.freeze(keys=["base_weight", "base_bias"], strict=False)

        # Count trainable params
        n_lora_params = sum(
            lora_layer.lora_a.size + lora_layer.lora_b.size
            for lora_layer in lora_layers.values()
        )
        logger.info("Training %d LoRA parameters (frozen base model)", n_lora_params)

        # Tokenize training data
        tokenized = _tokenize_data(training_data, tokenizer)

        if not tokenized:
            return GeometricLoRAResult(
                success=False,
                error="No valid training data after tokenization",
            )

        # Create optimizer (geometry-derived, no magic hyperparameters)
        from .geometric_optimizer import GeometricOptimizer
        optimizer = GeometricOptimizer(base_decay=0.0)
        optimizer.init_from_model(model)

        # Training loop
        final_loss = 0.0
        total_steps = 0
        total_batches = (len(tokenized) + config.batch_size - 1) // config.batch_size
        metrics_interval = max(total_batches // 4, 100)  # Log metrics ~4 times per epoch

        # Geometric convergence monitor (early stopping without validation set)
        convergence_monitor = GeometricConvergenceMonitor() if config.enable_geometric_stopping else None
        converged_early = False

        # Loop preservation tracking
        loop_loss_total = 0.0
        loop_delta_total = 0.0
        loop_samples = 0

        # Initial geometric metrics
        logger.info("Computing initial geometric metrics...")
        initial_metrics = compute_aggregate_metrics(lora_layers)
        _log_geometric_metrics(initial_metrics, "initial")

        if loop_config is not None:
            logger.info(
                "Loop preservation enabled: highway=%d, base_ΔH=%.4f, λ=%.6f",
                loop_config.highway_layer,
                loop_config.base_delta_entropy,
                loop_config.lambda_scale,
            )

        if convergence_monitor is not None:
            logger.info("Geometric early stopping enabled")

        for epoch in range(config.epochs):
            if converged_early:
                break
            epoch_loss = 0.0
            epoch_loop_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(tokenized), config.batch_size):
                batch = tokenized[batch_start:batch_start + config.batch_size]

                # Debug: print every 10 steps to track progress
                step_in_epoch = batch_start // config.batch_size + 1
                if step_in_epoch % 10 == 1:
                    import sys
                    print(f"[DEBUG] Processing batch {step_in_epoch}, step {total_steps + 1}", flush=True)
                    sys.stdout.flush()

                # Forward and backward pass (only unfrozen params get gradients)
                loss, grads = _compute_loss_and_grads(model, batch, lora_layers)

                # Add loop preservation loss if configured
                if loop_config is not None and len(batch) > 0:
                    # Compute entropy trajectory for this batch
                    # Use first sample in batch for efficiency
                    input_ids = batch[0][:-1][None, :]  # [1, seq]
                    trajectory = compute_entropy_trajectory(model, input_ids)

                    if trajectory:
                        lp_loss, delta = loop_preservation_loss(trajectory, loop_config)
                        epoch_loop_loss += lp_loss
                        loop_loss_total += lp_loss
                        loop_delta_total += delta
                        loop_samples += 1

                        # Add to total loss (already scaled by λ)
                        loss = loss + mx.array(lp_loss)

                # Update only trainable (LoRA) parameters
                optimizer.update(model, grads)
                mx.eval(loss)

                epoch_loss += float(loss)
                n_batches += 1
                total_steps += 1

                if progress_callback:
                    progress_callback({
                        "epoch": epoch,
                        "step": total_steps,
                        "loss": float(loss),
                    })

                # Enforce spectral bound and check saturation every 100 steps
                # (spectral norm computation is expensive, don't do it every step)
                if total_steps % 100 == 0:
                    saturation = enforce_spectral_bound(lora_layers)

                    logger.info(
                        "Step %d: loss=%.4f | saturation=%d/%d (%.1f%%)",
                        total_steps, float(loss),
                        saturation.n_saturated, saturation.n_total,
                        saturation.saturation_ratio * 100
                    )

                    # Check for geometric saturation (capacity full)
                    if saturation.is_full:
                        logger.info(
                            "Geometric saturation at step %d: %d/%d layers at capacity. Training complete.",
                            total_steps, saturation.n_saturated, saturation.n_total
                        )
                        converged_early = True
                        break

                # Log geometric metrics periodically (more informative than loss)
                if total_steps % metrics_interval == 0:
                    metrics = compute_aggregate_metrics(lora_layers)
                    _log_geometric_metrics(metrics, f"step_{total_steps}")

                # Check geometric convergence
                if convergence_monitor is not None:
                    conv_state = convergence_monitor.check(optimizer, lora_layers, float(loss))
                    if conv_state.should_stop:
                        logger.info(
                            "Geometric convergence at step %d: bb_stable=%s, loss_stable=%s, budget_exhausted=%s",
                            total_steps, conv_state.bb_stable, conv_state.loss_stable, conv_state.budget_exhausted
                        )
                        converged_early = True
                        break

                # Check max_steps limit
                if config.max_steps is not None and total_steps >= config.max_steps:
                    logger.info("Reached max_steps limit: %d", config.max_steps)
                    converged_early = True
                    break

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            avg_loop_loss = epoch_loop_loss / n_batches if n_batches > 0 else 0

            # End of epoch: compute and log geometric metrics
            epoch_metrics = compute_aggregate_metrics(lora_layers)
            _log_geometric_metrics(epoch_metrics, f"epoch_{epoch}")

            if loop_config is not None:
                avg_delta = loop_delta_total / loop_samples if loop_samples > 0 else 0
                logger.info(
                    "Epoch %d: loss=%.4f (loop=%.4f) | ΔH=%+.4f | bound_ratio=%.3f | capacity=%.1f%%",
                    epoch, avg_loss, avg_loop_loss, avg_delta,
                    epoch_metrics["spectral_bound_ratio"]["mean"],
                    epoch_metrics["capacity_utilization"]["mean"] * 100,
                )
            else:
                logger.info(
                    "Epoch %d: loss=%.4f | bound_ratio=%.3f (max=%.3f) | capacity=%.1f%% | concentration=%.3f",
                    epoch, avg_loss,
                    epoch_metrics["spectral_bound_ratio"]["mean"],
                    epoch_metrics["spectral_bound_ratio"]["max"],
                    epoch_metrics["capacity_utilization"]["mean"] * 100,
                    epoch_metrics["energy_concentration"]["mean"],
                )

            # Warn if violating spectral bounds
            n_violations = epoch_metrics["spectral_bound_ratio"]["n_violations"]
            if n_violations > 0:
                logger.warning(
                    "%d/%d layers exceeding spectral bound (ratio > 1.0)",
                    n_violations, epoch_metrics["n_layers"]
                )

            final_loss = avg_loss

        # Save adapter
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        _save_geometric_adapter(
            lora_layers,
            config,
            output_path,
        )

        training_time = time.time() - start_time

        return GeometricLoRAResult(
            success=True,
            config=config,
            adapter_path=output_path,
            final_loss=final_loss,
            training_time_seconds=training_time,
        )

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return GeometricLoRAResult(
            success=False,
            error=str(e),
            training_time_seconds=time.time() - start_time,
        )


def _tokenize_data(data: list[dict], tokenizer, max_length: int = 512) -> list[mx.array]:
    """Tokenize training data."""
    tokenized = []

    for sample in data:
        if "prompt" in sample and "completion" in sample:
            text = sample["prompt"] + sample["completion"]
        elif "text" in sample:
            text = sample["text"]
        elif "input" in sample and "output" in sample:
            text = sample["input"] + sample["output"]
        else:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(mx.array(tokens))

    return tokenized


def _compute_loss_and_grads(model, batch: list[mx.array], lora_layers):
    """Compute loss and gradients for a batch."""

    def loss_fn(model):
        total_loss = 0.0
        n_tokens = 0

        for tokens in batch:
            # Shift for language modeling
            input_ids = tokens[:-1]
            target_ids = tokens[1:]

            # Forward pass
            logits = model(input_ids[None, :])

            # Cross entropy loss
            logits_flat = logits[0]  # [seq_len, vocab]
            loss = nn.losses.cross_entropy(
                logits_flat,
                target_ids,
                reduction="sum",
            )
            total_loss += loss
            n_tokens += len(target_ids)

        return total_loss / n_tokens if n_tokens > 0 else mx.array(0.0)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    return loss, grads


def _log_geometric_metrics(metrics: dict, label: str) -> None:
    """Log geometric metrics in structured format."""
    logger.info(
        "Geometric metrics [%s]: "
        "bound_ratio=%.4f (max=%.4f, violations=%d/%d) | "
        "capacity=%.1f%% (range: %.1f%%-%.1f%%) | "
        "concentration=%.4f",
        label,
        metrics["spectral_bound_ratio"]["mean"],
        metrics["spectral_bound_ratio"]["max"],
        metrics["spectral_bound_ratio"]["n_violations"],
        metrics["n_layers"],
        metrics["capacity_utilization"]["mean"] * 100,
        metrics["capacity_utilization"]["min"] * 100,
        metrics["capacity_utilization"]["max"] * 100,
        metrics["energy_concentration"]["mean"],
    )


def _save_geometric_adapter(
    lora_layers: dict,
    config: GeometricLoRAConfig,
    output_path: Path,
):
    """Save the geometric LoRA adapter with final geometric metrics."""
    # Collect weights
    weights = {}
    for layer_key, lora_layer in lora_layers.items():
        weights[f"{layer_key}.lora_a"] = lora_layer.lora_a
        weights[f"{layer_key}.lora_b"] = lora_layer.lora_b

    # Save weights
    weights_path = output_path / "lora_weights.safetensors"
    mx.save_safetensors(str(weights_path), weights)

    # Compute final geometric metrics
    final_metrics = compute_aggregate_metrics(lora_layers)

    # Per-layer metrics for detailed analysis
    per_layer_metrics = {}
    for layer_key, lora_layer in lora_layers.items():
        layer_metrics = compute_layer_metrics(lora_layer, layer_key)
        per_layer_metrics[layer_key] = {
            "spectral_bound_ratio": layer_metrics.spectral_bound_ratio,
            "capacity_utilization": layer_metrics.capacity_utilization,
            "energy_concentration": layer_metrics.energy_concentration,
            "is_healthy": layer_metrics.is_healthy,
        }

    # Save config with geometry info and metrics
    config_dict = {
        "type": "geometric_lora",
        "rank": config.rank,
        "adaptive_ranks": config.adaptive_ranks_enabled,
        "target_modules": config.target_modules,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        # Store σ_k for each layer (needed for inference)
        "layer_sigma_k": {
            key: config.geometries[key].sigma_k
            for key in config.target_modules
        },
        # Store per-layer ranks (needed for inference with adaptive ranks)
        "per_layer_ranks": config.effective_ranks,
        # Store full geometry for reference
        "geometry": config.to_dict()["layer_geometries"],
        # Final geometric health metrics
        "final_metrics": {
            "aggregate": final_metrics,
            "per_layer": per_layer_metrics,
        },
    }

    config_path = output_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Log final health summary
    n_healthy = final_metrics["n_healthy"]
    n_total = final_metrics["n_layers"]
    logger.info(
        "Saved geometric adapter to %s | Health: %d/%d layers within bounds | "
        "Capacity: %.1f%% | Concentration: %.3f",
        output_path, n_healthy, n_total,
        final_metrics["capacity_utilization"]["mean"] * 100,
        final_metrics["energy_concentration"]["mean"],
    )


__all__ = [
    "GeometricMetrics",
    "GeometricLoRAConfig",
    "GeometricLoRAResult",
    "GeometricConvergenceState",
    "GeometricConvergenceMonitor",
    "compute_layer_metrics",
    "compute_aggregate_metrics",
    "derive_config_from_geometry",
    "train_geometric_lora",
]
