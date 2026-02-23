# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Data-derived early stopping and geometric stopping certificate.

Pure Python — zero framework dependencies.

Two stopping mechanisms:

1. **Heuristic (legacy):** ``check_val_loss_converged()`` compares windowed
   loss means against the SE of the difference. Used as fallback when no
   eval dataset is provided.

2. **Certificate:** ``check_stopping_certificate()`` verifies four
   mathematically sufficient conditions for convergence:

   - Stationarity: Riemannian gradient norm at numerical floor.
   - Improvement bound: best local val improvement below sampling uncertainty.
   - Worst-group: no single batch can improve more than its noise.
   - No mechanism drift: entropy / repetition within dtype-derived bounds.

   When all four hold, no measurable local improvement remains under the
   current geometry and finite-sample uncertainty.

Machine epsilon is retained as a lower bound for the degenerate case
where per-batch variance -> 0 (perfectly uniform dataset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# sqrt(eps) for float32: numerical significance threshold
_SQRT_EPS = math.sqrt(math.ldexp(1.0, -23))  # ~3.45e-4


def check_loss_stable(
    losses: list[tuple[int, float, float]],
    window: int | None = None,
    numeric_floor: float = _SQRT_EPS,
) -> tuple[bool, float]:
    """Check if loss has converged: |delta_epoch_mean| < standard error of the difference.

    Compares mean of last ``window`` losses to mean of previous ``window`` losses.
    If ``window`` is None, it is derived from the observed trajectory by splitting
    the available history into two equal windows.
    Threshold is the standard error of the difference of the two epoch means:

        SE_diff = sqrt(var_recent/N + var_earlier/N)

    This is a MEASUREMENT of the data's own noise floor, not a fixed constant.
    On homogeneous data (40 samples), per-batch variance is low -> tight threshold.
    On diverse data (724 samples), per-batch variance is high -> threshold accounts
    for batch-to-batch noise that would mask convergence.

    Args:
        losses: List of (iteration, loss_value, tokens_per_sec) tuples.
        window: Number of recent entries to compare. If None, derived from
            the available trajectory length.
        numeric_floor: Numerical lower bound for distinguishability.

    Returns:
        (is_stable, threshold) -- threshold is the SE_diff actually used.
    """
    resolved_window = window if window is not None else len(losses) // 2
    if resolved_window < 2:
        return False, 0.0

    if len(losses) < 2 * resolved_window:
        return False, 0.0

    recent = [entry[1] for entry in losses[-resolved_window:]]
    earlier = [entry[1] for entry in losses[-2 * resolved_window : -resolved_window]]

    n = len(recent)
    mean_recent = sum(recent) / n
    mean_earlier = sum(earlier) / n

    # Variance of each epoch's per-batch losses
    var_recent = sum((x - mean_recent) ** 2 for x in recent) / n
    var_earlier = sum((x - mean_earlier) ** 2 for x in earlier) / n

    # Standard error of the difference of two means
    se_diff = math.sqrt((var_recent + var_earlier) / n)

    # Use data-derived SE, but never below machine epsilon
    threshold = max(se_diff, numeric_floor)

    return abs(mean_recent - mean_earlier) < threshold, threshold


def check_val_loss_converged(
    val_losses: list[float],
    window: int | None = None,
    numeric_floor: float = _SQRT_EPS,
) -> tuple[bool, str, float]:
    """Check if validation loss has converged or is degrading.

    Compares mean of last ``window`` validation loss measurements to the
    mean of the previous ``window`` measurements. Two stopping conditions:

    1. **Stable** (converged): |delta| < SE_diff — no more useful learning.
    2. **Increasing** (overfitting): delta > SE_diff — generalization degrading.

    Validation loss is the unbiased estimate of generalization error. Training
    loss measures memorization. This function watches generalization.

    Args:
        val_losses: List of per-epoch validation loss values.
        window: Number of recent epochs to compare.
        numeric_floor: Numerical lower bound for distinguishability.

    Returns:
        (should_stop, reason, threshold) where reason is one of:
        - ``""`` — don't stop, still improving
        - ``"val_stable"`` — validation loss converged (no more improvement)
        - ``"val_increasing"`` — validation loss increasing (overfitting)
    """
    resolved_window = window if window is not None else len(val_losses) // 2
    if resolved_window < 2:
        return False, "", 0.0

    if len(val_losses) < 2 * resolved_window:
        return False, "", 0.0

    recent = val_losses[-resolved_window:]
    earlier = val_losses[-2 * resolved_window : -resolved_window]

    n = len(recent)
    mean_recent = sum(recent) / n
    mean_earlier = sum(earlier) / n

    var_recent = sum((x - mean_recent) ** 2 for x in recent) / n
    var_earlier = sum((x - mean_earlier) ** 2 for x in earlier) / n

    se_diff = math.sqrt((var_recent + var_earlier) / n)
    threshold = max(se_diff, numeric_floor)

    delta = mean_recent - mean_earlier

    # Val loss increasing → overfitting
    if delta > threshold:
        return True, "val_increasing", threshold

    # Val loss stable → converged
    if abs(delta) < threshold:
        return True, "val_stable", threshold

    # Val loss still decreasing → keep training
    return False, "", threshold


# ── Geometric Stopping Certificate ──────────────────────────────────


def check_grad_norm_stable(
    grad_norms: list[float],
    window: int | None = None,
    numeric_floor: float = _SQRT_EPS,
) -> tuple[bool, float]:
    """Check if preconditioned gradient norm has converged to its stochastic floor.

    In stochastic (mini-batch) optimization, the gradient norm does not converge
    to zero. It oscillates around an irreducible noise floor determined by batch
    variance and learning rate. This function detects when that oscillation has
    stabilized — the norm is no longer decreasing, meaning further training
    cannot reduce the Riemannian gradient below the stochastic noise floor.

    Same windowed-SE test as ``check_loss_stable()``, applied to gradient norms.

    Args:
        grad_norms: Per-epoch preconditioned gradient norms ``||P^{1/2} g||``.
        window: Number of recent epochs to compare.
        numeric_floor: Numerical lower bound for distinguishability.

    Returns:
        (is_stable, threshold) — ``is_stable`` is True when the gradient norm
        has converged to its stochastic noise floor.
    """
    resolved_window = window if window is not None else len(grad_norms) // 2
    if resolved_window < 2:
        return False, 0.0

    if len(grad_norms) < 2 * resolved_window:
        return False, 0.0

    recent = grad_norms[-resolved_window:]
    earlier = grad_norms[-2 * resolved_window : -resolved_window]

    n = len(recent)
    mean_recent = sum(recent) / n
    mean_earlier = sum(earlier) / n

    var_recent = sum((x - mean_recent) ** 2 for x in recent) / n
    var_earlier = sum((x - mean_earlier) ** 2 for x in earlier) / n

    se_diff = math.sqrt((var_recent + var_earlier) / n)
    threshold = max(se_diff, numeric_floor)

    return abs(mean_recent - mean_earlier) < threshold, threshold


@dataclass(frozen=True)
class StoppingCertificate:
    """Geometric stopping certificate for Riemannian training.

    All four conditions must hold simultaneously to certify that no
    measurable local improvement remains:

    1. **Stationarity:** ``||P^{1/2} ∇L_train||`` has converged to its
       stochastic noise floor (windowed SE test on gradient norm history).
    2. **Improvement bound:** ``Δ_max_val < CI_half_width(val)``
    3. **Worst-group:** ``max_i Δ_max_i < CI_half_width_i``
    4. **No drift:** entropy not collapsed, repetition not spiked.

    All thresholds are dtype-derived or measured from data. No patience
    counters, no fixed hyperparameters.
    """

    # Condition 1: Stationarity on the train manifold
    precond_grad_norm: float       # ||P^{1/2} ∇L_train|| = sqrt(g^T P g)
    stationarity_floor: float      # SE threshold from gradient norm history
    stationarity_met: bool

    # Condition 2: Improvement bound vs validation uncertainty
    alignment: float               # a_t = ∇L_val^T d_t
    curvature: float               # b_t = d_t^T H_val d_t
    delta_max_val: float           # a_t² / (2 b_t) when a>0, b>0; else 0
    val_ci_half_width: float       # bootstrap CI half-width
    improvement_bound_met: bool

    # Condition 3: Worst-group bound
    delta_max_worst: float         # max per-batch Δ_max_i
    worst_ci_half_width: float     # CI half-width for the worst batch
    worst_group_met: bool

    # Condition 4: Mechanism drift
    entropy_collapsed: bool        # entropy < sqrt(ε_f32)
    entropy_expanding: bool        # entropy > baseline * (1 + sqrt(ε_f32)) — SFT entropy-seeking
    repetition_spiked: bool        # repetition > 1 - sqrt(ε_f32)
    no_drift: bool

    # Aggregate
    all_conditions_met: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for logging / metrics."""
        return {k: v for k, v in self.__dict__.items()}


def _improvement_bound(alignment: float, curvature: float) -> float:
    """Compute local improvement bound for L(θ - η d).

    Let ``a = ∇L^T d`` and ``b = d^T H d``. The second-order model gives:

    ``Δ(η) = L(θ) - L(θ-ηd) ≈ η a - 0.5 η² b``.

    - If ``a <= 0``, this direction is not descent for validation loss;
      no local improvement is certified, so return 0.
    - If ``a > 0`` and ``b > 0``, the maximal quadratic improvement is
      ``a² / (2b)``.
    - If ``a > 0`` and ``b <= 0``, the quadratic model does not provide a
      finite upper bound along this direction; treat as unbounded (``inf``).
    """
    if alignment <= 0.0:
        return 0.0
    if curvature <= 0.0:
        return float("inf")
    return alignment * alignment / (2.0 * curvature)


def check_stopping_certificate(
    # Condition 1: Stationarity
    precond_grad_norm: float,
    grad_norm_history: list[float] | None = None,
    numeric_floor: float = _SQRT_EPS,
    # Condition 2: Improvement bound
    alignment: float = 0.0,
    curvature: float = 0.0,
    val_ci_half_width: float = 0.0,
    # Condition 3: Worst-group
    per_batch_alignments: list[float] | None = None,
    per_batch_curvatures: list[float] | None = None,
    per_batch_ci_half_widths: list[float] | None = None,
    # Condition 4: Mechanism drift
    mean_token_entropy: float | None = None,
    baseline_entropy: float | None = None,
    repetition_rate: float | None = None,
) -> StoppingCertificate:
    """Evaluate the geometric stopping certificate.

    Pure Python. Takes precomputed scalars from the adapter, returns a
    frozen ``StoppingCertificate`` with per-condition verdicts and the
    aggregate decision.

    Args:
        precond_grad_norm: ||P^{1/2} ∇L_train|| (Riemannian gradient norm).
        grad_norm_history: Per-epoch gradient norms for stochastic stationarity.
            When provided, stationarity is checked via ``check_grad_norm_stable()``
            (has the norm converged to its stochastic floor?). When None, falls
            back to ``precond_grad_norm < numeric_floor`` (deterministic case).
        numeric_floor: dtype-derived floor (default: sqrt(ε_f32)).
        alignment: a_t = ∇L_val^T d_t.
        curvature: b_t = d_t^T H_val d_t.
        val_ci_half_width: bootstrap CI half-width of mean val loss.
            When 0, conditions 2 & 3 are vacuously satisfied (no val data).
        per_batch_alignments: per-batch a_i values.
        per_batch_curvatures: per-batch b_i values.
        per_batch_ci_half_widths: per-batch CI half-widths.
        mean_token_entropy: mean Shannon entropy of generated tokens.
        baseline_entropy: pre-training entropy (measured before adaptation).
            When provided, detects entropy expansion (SFT entropy-seeking).
        repetition_rate: fraction of repeated 4-grams.

    Returns:
        ``StoppingCertificate`` with all four conditions evaluated.
    """
    # ── Condition 1: Stationarity ──
    # Stochastic case: gradient norm history available → check if norm
    # has converged to its stochastic noise floor via windowed SE test.
    # Deterministic case (no history): fall back to machine-epsilon floor.
    # Include current epoch's norm in the stationarity test; otherwise a spike
    # in the current step can be ignored by construction.
    history = (list(grad_norm_history) if grad_norm_history is not None else [])
    history.append(precond_grad_norm)
    if len(history) >= 4:
        stationarity_met, stationarity_floor = check_grad_norm_stable(
            history, window=None, numeric_floor=numeric_floor,
        )
    else:
        stationarity_met = precond_grad_norm < numeric_floor
        stationarity_floor = numeric_floor

    # ── Condition 2: Improvement bound ──
    delta_max_val = _improvement_bound(alignment, curvature)

    # val_ci_half_width <= 0 means no val data → vacuously satisfied
    if val_ci_half_width <= 0.0:
        improvement_bound_met = True
    else:
        improvement_bound_met = delta_max_val < val_ci_half_width

    # ── Condition 3: Worst-group ──
    delta_max_worst = 0.0
    worst_ci = 0.0

    if (
        per_batch_alignments is not None
        and per_batch_curvatures is not None
        and per_batch_ci_half_widths is not None
        and len(per_batch_alignments) > 0
    ):
        worst_group_met = True
        for a_i, b_i, ci_i in zip(
            per_batch_alignments,
            per_batch_curvatures,
            per_batch_ci_half_widths,
        ):
            d_i = _improvement_bound(a_i, b_i)
            if d_i > delta_max_worst:
                delta_max_worst = d_i
                worst_ci = ci_i
            if ci_i > 0.0 and d_i >= ci_i:
                worst_group_met = False
    else:
        # Fallback: use aggregate bound
        delta_max_worst = delta_max_val
        worst_ci = val_ci_half_width
        worst_group_met = improvement_bound_met

    # ── Condition 4: Mechanism drift ──
    entropy_collapsed = (
        mean_token_entropy is not None and mean_token_entropy < numeric_floor
    )
    # Entropy expansion: SFT can trigger entropy-seeking in small models.
    # Detect if current entropy exceeds baseline by more than sqrt(eps_f32).
    # Threshold: baseline * (1 + sqrt(eps_f32)). Derived from IEEE 754.
    entropy_expanding = (
        mean_token_entropy is not None
        and baseline_entropy is not None
        and baseline_entropy > 0.0
        and mean_token_entropy > baseline_entropy * (1.0 + numeric_floor)
    )
    repetition_spiked = (
        repetition_rate is not None and repetition_rate > 1.0 - numeric_floor
    )
    no_drift = not entropy_collapsed and not entropy_expanding and not repetition_spiked

    # ── Aggregate ──
    all_met = (
        stationarity_met
        and improvement_bound_met
        and worst_group_met
        and no_drift
    )

    return StoppingCertificate(
        precond_grad_norm=precond_grad_norm,
        stationarity_floor=stationarity_floor,
        stationarity_met=stationarity_met,
        alignment=alignment,
        curvature=curvature,
        delta_max_val=delta_max_val,
        val_ci_half_width=val_ci_half_width,
        improvement_bound_met=improvement_bound_met,
        delta_max_worst=delta_max_worst,
        worst_ci_half_width=worst_ci,
        worst_group_met=worst_group_met,
        entropy_collapsed=entropy_collapsed,
        entropy_expanding=entropy_expanding,
        repetition_spiked=repetition_spiked,
        no_drift=no_drift,
        all_conditions_met=all_met,
    )


__all__ = [
    "_SQRT_EPS",
    "StoppingCertificate",
    "check_grad_norm_stable",
    "check_loss_stable",
    "check_stopping_certificate",
    "check_val_loss_converged",
]
