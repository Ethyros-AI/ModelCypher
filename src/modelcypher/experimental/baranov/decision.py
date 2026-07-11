"""Pre-registered decision logic for Baranov Track A.

EXPERIMENTAL: Not validated for production use.

Determines pass/fail/inconclusive for alignment-tax claims based on:
- Delta recall (post - pre) across seeds per model
- CKA drift (geometry distortion)
- No-op noise floor (measurement pipeline artifact bound)
- Scaling trend across model sizes

All significance thresholds are derived from the data:
- Alpha = 1/n_facts (Clopper-Pearson convention from recall_evaluator)
- Noise floor = max |delta| from no_op control
- No heuristic constants.

Decision criteria (protocol §4.3, pre-registered):
    Replication pass:
        Directional trend of recall suppression with scale in at least
        one mode split, with CI excluding zero effect for that split.
    Mechanism pass:
        Suppression co-occurs with geometry signatures (CKA drift > 0
        and/or preserved-fraction collapse).
    Fail:
        No consistent trend, trend reverses under controls, or CIs
        include no-effect across all splits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain.statistics import clopper_pearson_interval

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeVerdict:
    """Decision for a single mode (raw_completion or chat_template) for one model.

    Attributes
    ----------
    mode:
        ``raw_completion`` or ``chat_template``.
    pre_recall:
        Pre-intervention recall rate (constant across seeds).
    post_recall_per_seed:
        Post-intervention recall rate for each seed.
    delta_mean:
        Mean delta (post - pre) across seeds.
    delta_ci:
        Confidence interval on the delta.
    noise_floor:
        Maximum |delta| observed in no_op control for this mode.
    significant:
        Whether delta CI excludes the noise floor band.
    verdict:
        ``effective`` (injection worked), ``ineffective`` (no signal),
        or ``degraded`` (recall worsened).
    reason:
        Human-readable explanation.
    """

    mode: str
    pre_recall: float
    post_recall_per_seed: tuple[float, ...]
    delta_mean: float
    delta_ci: tuple[float, float]
    noise_floor: float
    significant: bool
    verdict: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pre_recall": self.pre_recall,
            "post_recall_per_seed": list(self.post_recall_per_seed),
            "delta_mean": round(self.delta_mean, 6),
            "delta_ci": [round(self.delta_ci[0], 6), round(self.delta_ci[1], 6)],
            "noise_floor": round(self.noise_floor, 6),
            "significant": self.significant,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModelVerdict:
    """Decision for a single model across modes and seeds.

    Attributes
    ----------
    model:
        Model name.
    n_seeds:
        Number of seeds used.
    raw:
        Verdict for raw_completion mode.
    chat:
        Verdict for chat_template mode.
    cka_drift_mean:
        Mean CKA drift across seeds.
    preserved_fraction_mean:
        Mean preserved fraction across seeds.
    verdict:
        Overall model verdict (``effective``, ``ineffective``, ``degraded``).
    reason:
        Human-readable explanation.
    """

    model: str
    n_seeds: int
    raw: ModeVerdict
    chat: ModeVerdict
    cka_drift_mean: float
    preserved_fraction_mean: float
    verdict: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_seeds": self.n_seeds,
            "raw": self.raw.as_dict(),
            "chat": self.chat.as_dict(),
            "cka_drift_mean": round(self.cka_drift_mean, 6),
            "preserved_fraction_mean": round(self.preserved_fraction_mean, 6),
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrackADecision:
    """Overall Track A decision across models.

    This is the final pre-registered decision artifact.

    Attributes
    ----------
    per_model:
        Per-model verdicts keyed by model name.
    scaling_verdict:
        ``pass``, ``fail``, or ``inconclusive`` for the scaling claim.
    scaling_reason:
        Human-readable explanation of the scaling analysis.
    geometry_co_occurrence:
        Whether geometry signatures (CKA drift) co-occur with
        recall suppression.
    overall_verdict:
        Final Track A verdict per protocol §4.3.
    overall_reason:
        Combined explanation.
    """

    per_model: dict[str, ModelVerdict]
    scaling_verdict: str
    scaling_reason: str
    geometry_co_occurrence: bool
    overall_verdict: str
    overall_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_model": {k: v.as_dict() for k, v in self.per_model.items()},
            "scaling_verdict": self.scaling_verdict,
            "scaling_reason": self.scaling_reason,
            "geometry_co_occurrence": self.geometry_co_occurrence,
            "overall_verdict": self.overall_verdict,
            "overall_reason": self.overall_reason,
        }


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseFloor:
    """Noise floor from no_op control measurements.

    If no no_op data is available, floor defaults to 0.0 (greedy
    decoding on the same model should produce identical outputs).
    """

    raw: float = 0.0
    chat: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"raw": self.raw, "chat": self.chat}


def compute_noise_floor(
    noop_results: list[dict[str, Any]],
) -> NoiseFloor:
    """Compute noise floor from no_op control results.

    The noise floor is the maximum absolute delta observed across
    all models in the no_op condition.  With greedy decoding, this
    should be exactly 0.0.
    """
    if not noop_results:
        return NoiseFloor()

    max_raw = 0.0
    max_chat = 0.0
    for r in noop_results:
        if r.get("deltas") is not None:
            max_raw = max(max_raw, abs(r["deltas"]["delta_raw_recall"]))
            max_chat = max(max_chat, abs(r["deltas"]["delta_chat_recall"]))

    return NoiseFloor(raw=max_raw, chat=max_chat)


# ---------------------------------------------------------------------------
# Per-mode decision
# ---------------------------------------------------------------------------


def _compute_delta_ci_single_seed(
    pre_rate: float,
    post_rate: float,
    n_facts: int,
    alpha: float,
) -> tuple[float, float]:
    """CI on delta from a single seed via Clopper-Pearson on post count."""
    post_count = round(post_rate * n_facts)
    ci_lo, ci_hi = clopper_pearson_interval(
        n_correct=post_count, n_total=n_facts, alpha=alpha,
    )
    return (ci_lo - pre_rate, ci_hi - pre_rate)


def _compute_delta_ci_multi_seed(
    deltas: list[float],
    alpha: float,
) -> tuple[float, float]:
    """CI on delta from multiple seeds via t-interval."""
    n = len(deltas)
    if n < 2:
        mean_d = deltas[0] if deltas else 0.0
        return (mean_d, mean_d)

    mean_d = sum(deltas) / n
    variance = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    se = math.sqrt(variance / n)

    if se == 0.0:
        return (mean_d, mean_d)

    # t critical value via scipy
    from scipy.stats import t as t_dist

    t_val = t_dist.ppf(1 - alpha / 2, n - 1)
    return (mean_d - t_val * se, mean_d + t_val * se)


def compute_mode_verdict(
    mode: str,
    pre_rate: float,
    post_rates: list[float],
    n_facts: int,
    noise_floor: float,
) -> ModeVerdict:
    """Compute decision for a single mode of a single model.

    Parameters
    ----------
    mode:
        ``raw_completion`` or ``chat_template``.
    pre_rate:
        Pre-intervention recall rate.
    post_rates:
        Post-intervention recall rates (one per seed).
    n_facts:
        Number of facts evaluated.
    noise_floor:
        Maximum |delta| from no_op control for this mode.
    """
    if not post_rates:
        return ModeVerdict(
            mode=mode,
            pre_recall=pre_rate,
            post_recall_per_seed=(),
            delta_mean=0.0,
            delta_ci=(0.0, 0.0),
            noise_floor=noise_floor,
            significant=False,
            verdict="ineffective",
            reason="No post-measurement data.",
        )

    n_seeds = len(post_rates)
    deltas = [p - pre_rate for p in post_rates]
    delta_mean = sum(deltas) / n_seeds

    # Alpha derived from n_facts (1/n, matching recall_evaluator convention)
    alpha = 1.0 / n_facts if n_facts > 0 else 0.05

    if n_seeds == 1:
        delta_ci = _compute_delta_ci_single_seed(
            pre_rate, post_rates[0], n_facts, alpha,
        )
    else:
        delta_ci = _compute_delta_ci_multi_seed(deltas, alpha)

    # Significance: CI excludes noise floor band [-floor, +floor]
    significant = (delta_ci[1] < -noise_floor) or (delta_ci[0] > noise_floor)

    # Verdict
    if not significant:
        verdict = "ineffective"
        reason = (
            f"Delta CI [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}] "
            f"overlaps noise floor +/-{noise_floor:.4f}."
        )
    elif delta_mean > 0:
        verdict = "effective"
        reason = (
            f"Recall increased {delta_mean:+.4f} "
            f"(CI [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}] excludes floor)."
        )
    else:
        verdict = "degraded"
        reason = (
            f"Recall decreased {delta_mean:+.4f} "
            f"(CI [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}] excludes floor)."
        )

    return ModeVerdict(
        mode=mode,
        pre_recall=pre_rate,
        post_recall_per_seed=tuple(post_rates),
        delta_mean=delta_mean,
        delta_ci=delta_ci,
        noise_floor=noise_floor,
        significant=significant,
        verdict=verdict,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Per-model decision
# ---------------------------------------------------------------------------


def compute_model_verdict(
    model: str,
    pre_raw_rate: float,
    pre_chat_rate: float,
    seed_results: list[dict[str, Any]],
    n_facts: int,
    noise_floor: NoiseFloor,
) -> ModelVerdict:
    """Compute decision for a single model across seeds.

    Parameters
    ----------
    model:
        Model name.
    pre_raw_rate:
        Pre-intervention raw recall rate (constant across seeds).
    pre_chat_rate:
        Pre-intervention chat recall rate (constant across seeds).
    seed_results:
        List of per-seed result dicts, each containing:
        - ``post_raw_rate``: float
        - ``post_chat_rate``: float
        - ``cka_drift``: float
        - ``preserved_fraction``: float
    n_facts:
        Number of facts evaluated.
    noise_floor:
        Noise floor from no_op control.
    """
    n_seeds = len(seed_results)

    post_raw_rates = [r["post_raw_rate"] for r in seed_results]
    post_chat_rates = [r["post_chat_rate"] for r in seed_results]
    cka_drifts = [r["cka_drift"] for r in seed_results]
    preserved_fractions = [r["preserved_fraction"] for r in seed_results]

    raw_verdict = compute_mode_verdict(
        "raw_completion", pre_raw_rate, post_raw_rates, n_facts, noise_floor.raw,
    )
    chat_verdict = compute_mode_verdict(
        "chat_template", pre_chat_rate, post_chat_rates, n_facts, noise_floor.chat,
    )

    cka_drift_mean = sum(cka_drifts) / len(cka_drifts) if cka_drifts else 0.0
    preserved_fraction_mean = (
        sum(preserved_fractions) / len(preserved_fractions) if preserved_fractions else 1.0
    )

    # Overall model verdict: best of the two modes
    if raw_verdict.significant or chat_verdict.significant:
        if raw_verdict.verdict == "degraded" or chat_verdict.verdict == "degraded":
            verdict = "degraded"
            reason = "Recall degraded in at least one mode after intervention."
        elif raw_verdict.verdict == "effective" or chat_verdict.verdict == "effective":
            verdict = "effective"
            reason = "Injection effective in at least one mode."
        else:
            verdict = "ineffective"
            reason = "Significant change detected but direction unclear."
    else:
        verdict = "ineffective"
        reason = "No significant change in either mode."

    return ModelVerdict(
        model=model,
        n_seeds=n_seeds,
        raw=raw_verdict,
        chat=chat_verdict,
        cka_drift_mean=cka_drift_mean,
        preserved_fraction_mean=preserved_fraction_mean,
        verdict=verdict,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Scaling analysis (Track A claim)
# ---------------------------------------------------------------------------


def _is_monotone_decreasing(values: list[float]) -> bool:
    """Check if values are strictly non-increasing."""
    return all(a >= b for a, b in zip(values, values[1:]))


def compute_track_a_decision(
    model_verdicts: dict[str, ModelVerdict],
    model_order: list[str],
) -> TrackADecision:
    """Compute overall Track A decision from per-model verdicts.

    Applies protocol §4.3 criteria:
    - Replication pass: directional trend + CI excluding zero in >= 1 mode
    - Mechanism pass: geometry co-occurrence
    - Fail: no trend, trend reverses, or CIs include no-effect

    Parameters
    ----------
    model_verdicts:
        Per-model verdicts keyed by model name.
    model_order:
        Models in ascending scale order (smallest first).
    """
    # Extract deltas in scale order (only models in both verdict dict and order list)
    ordered_models = [m for m in model_order if m in model_verdicts]

    if len(ordered_models) < 2:
        return TrackADecision(
            per_model=model_verdicts,
            scaling_verdict="inconclusive",
            scaling_reason="Fewer than 2 models in scale order — cannot assess scaling trend.",
            geometry_co_occurrence=False,
            overall_verdict="inconclusive",
            overall_reason="Insufficient models for scaling analysis.",
        )
    raw_deltas = [model_verdicts[m].raw.delta_mean for m in ordered_models]
    chat_deltas = [model_verdicts[m].chat.delta_mean for m in ordered_models]

    # Check for suppression trend: delta DECREASING with scale
    # (i.e., larger models gain less recall from LoRA injection)
    raw_trend_decreasing = _is_monotone_decreasing(raw_deltas)
    chat_trend_decreasing = _is_monotone_decreasing(chat_deltas)

    # Check CIs exclude zero in the mode showing the trend
    raw_cis_exclude_zero = all(
        model_verdicts[m].raw.significant for m in ordered_models
    )
    chat_cis_exclude_zero = all(
        model_verdicts[m].chat.significant for m in ordered_models
    )

    # Geometry co-occurrence: CKA drift present where suppression occurs
    geometry_co_occurrence = any(
        model_verdicts[m].cka_drift_mean > 0.0 for m in ordered_models
    )

    # Apply decision criteria
    raw_pass = raw_trend_decreasing and raw_cis_exclude_zero
    chat_pass = chat_trend_decreasing and chat_cis_exclude_zero

    if raw_pass or chat_pass:
        passing_modes = []
        if raw_pass:
            passing_modes.append("raw_completion")
        if chat_pass:
            passing_modes.append("chat_template")

        if geometry_co_occurrence:
            scaling_verdict = "pass"
            scaling_reason = (
                f"Directional suppression trend with scale in "
                f"{', '.join(passing_modes)} with CIs excluding zero. "
                f"Geometry co-occurrence confirmed."
            )
        else:
            scaling_verdict = "inconclusive"
            scaling_reason = (
                f"Directional trend in {', '.join(passing_modes)} "
                f"but no geometry co-occurrence (CKA drift = 0 everywhere). "
                f"Mechanism pass requires geometry signature."
            )
    else:
        # Check if any mode shows significant effects at all
        any_significant = any(
            model_verdicts[m].raw.significant or model_verdicts[m].chat.significant
            for m in ordered_models
        )

        if not any_significant:
            scaling_verdict = "fail"
            scaling_reason = (
                "CIs include no-effect across all models and modes. "
                "No alignment-tax signal detected."
            )
        else:
            # Significant effects but no consistent trend
            raw_desc = (
                "decreasing" if raw_trend_decreasing else
                "non-monotone"
            )
            chat_desc = (
                "decreasing" if chat_trend_decreasing else
                "non-monotone"
            )
            scaling_verdict = "inconclusive"
            scaling_reason = (
                f"Significant effects detected but no consistent scaling trend. "
                f"Raw deltas: {raw_desc} ({[f'{d:.3f}' for d in raw_deltas]}). "
                f"Chat deltas: {chat_desc} ({[f'{d:.3f}' for d in chat_deltas]})."
            )

    # Overall = replication + mechanism
    if scaling_verdict == "pass":
        overall_verdict = "pass"
        overall_reason = scaling_reason
    elif scaling_verdict == "fail":
        overall_verdict = "fail"
        overall_reason = scaling_reason
    else:
        overall_verdict = "inconclusive"
        overall_reason = scaling_reason

    return TrackADecision(
        per_model=model_verdicts,
        scaling_verdict=scaling_verdict,
        scaling_reason=scaling_reason,
        geometry_co_occurrence=geometry_co_occurrence,
        overall_verdict=overall_verdict,
        overall_reason=overall_reason,
    )


__all__ = [
    "ModeVerdict",
    "ModelVerdict",
    "NoiseFloor",
    "TrackADecision",
    "compute_mode_verdict",
    "compute_model_verdict",
    "compute_noise_floor",
    "compute_track_a_decision",
]
