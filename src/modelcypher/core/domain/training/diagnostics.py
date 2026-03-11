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

"""Pure diagnostic interpreter for training results.

Transforms raw training metrics into agent-readable observations and
recommendations. No magic constants — all comparisons against values
already present in the training result dict.
"""

from __future__ import annotations

from typing import Any

from modelcypher.core.domain.agent_protocol import (
    AgentDiagnostics,
    AgentRecommendation,
)


# ---------------------------------------------------------------------------
# Stop reason interpretation
# ---------------------------------------------------------------------------

# Stop reasons are format strings with embedded numbers, e.g.:
#   "certificate (‖g‖=8.17e-01, Δmax=0.00e+00<CI=3.75e-02, epoch=8)"
#   "adapter_saturation_exhausted (Weyl crossing, median_ratio=0.9500, epoch=6)"
#   "safety_cap (1500 iters)"
# We match on the prefix to determine the category.

_STOP_REASON_EXPLANATIONS: list[tuple[str, str]] = [
    (
        "certificate",
        "Gradient norm reached the noise floor and validation loss stabilized. "
        "Training extracted what it can from this data.",
    ),
    (
        "adapter_saturation_exhausted",
        "Adapter spectral capacity fully consumed — the geometric budget is spent. "
        "Further training cannot improve without increasing rank.",
    ),
    (
        "adapter_saturation_cap",
        "Adapter saturation reached the configured cap. "
        "Training stopped to preserve remaining spectral headroom.",
    ),
    (
        "moe_expert_saturation_exhausted",
        "All targeted MoE experts reached spectral capacity.",
    ),
    (
        "max_epochs",
        "Hit the maximum epoch cap. Training may not have fully converged.",
    ),
    (
        "online_eval_degraded_significant",
        "Online evaluation detected significant inference degradation. "
        "Stopped to prevent further damage.",
    ),
    (
        "degeneration_exceeded",
        "Output quality degraded — n-gram repetition rate exceeded baseline. "
        "Stopped to prevent degenerate text generation.",
    ),
    (
        "loss_stable",
        "Training loss stabilized (no validation set available). "
        "Further iterations would not reduce loss.",
    ),
    (
        "safety_cap",
        "Hit the safety iteration limit derived from machine precision. "
        "The geometric stopping certificate failed to fire — this may indicate "
        "a convergence failure.",
    ),
    (
        "val_loss",
        "Validation loss converged — further training won't reduce loss on held-out data.",
    ),
]


def interpret_stop_reason(stop_reason: str | None) -> str:
    """Return a plain-language explanation of a training stop reason."""
    if not stop_reason:
        return "Training completed (no stop reason recorded)."
    for prefix, explanation in _STOP_REASON_EXPLANATIONS:
        if stop_reason.startswith(prefix):
            return explanation
    return f"Training stopped: {stop_reason}"


# ---------------------------------------------------------------------------
# Pipeline gate interpretation
# ---------------------------------------------------------------------------


def interpret_pipeline_gate(gate_checks: dict[str, Any] | None) -> list[str]:
    """Convert pipeline gate checks to agent-readable observations."""
    if not gate_checks:
        return []

    observations: list[str] = []
    for name, check in gate_checks.items():
        if not isinstance(check, dict):
            continue
        status = check.get("status", "unknown")
        message = check.get("message")
        value = check.get("value")

        if status == "pass":
            if value is not None:
                observations.append(f"{name}: passed (value={value})")
            else:
                observations.append(f"{name}: passed")
        elif status == "fail":
            detail = message or check.get("failure_mode", "")
            observations.append(f"{name}: FAILED — {detail}")
        elif status == "unresolved":
            observations.append(f"{name}: unresolved (insufficient data)")
    return observations


# ---------------------------------------------------------------------------
# Observations from training result
# ---------------------------------------------------------------------------


def _build_observations(result: dict[str, Any]) -> list[str]:
    """Extract factual observations from a training result dict."""
    obs: list[str] = []

    # Stop reason
    stop_reason = result.get("stop_reason")
    if stop_reason:
        obs.append(f"Stop reason: {interpret_stop_reason(stop_reason)}")

    # Loss trajectory
    baseline_loss = result.get("baseline_loss")
    post_loss = result.get("post_loss")
    if baseline_loss is not None and post_loss is not None:
        delta = post_loss - baseline_loss
        if delta < 0:
            obs.append(
                f"Loss improved: {baseline_loss:.4f} → {post_loss:.4f} "
                f"(Δ={delta:+.4f})"
            )
        else:
            obs.append(
                f"Loss did not improve: {baseline_loss:.4f} → {post_loss:.4f} "
                f"(Δ={delta:+.4f})"
            )

    # Perplexity
    baseline_ppl = result.get("baseline_perplexity")
    post_ppl = result.get("post_perplexity")
    if baseline_ppl is not None and post_ppl is not None:
        obs.append(f"Perplexity: {baseline_ppl:.2f} → {post_ppl:.2f}")

    # CKA preservation
    min_cka = result.get("min_cka")
    mean_cka = result.get("mean_cka")
    if min_cka is not None:
        obs.append(f"CKA preservation: min={min_cka:.3f}, mean={mean_cka:.3f}")

    # Spectral bounds
    spectral_ok = result.get("spectral_bounds_ok")
    if spectral_ok is not None:
        if spectral_ok:
            obs.append("Spectral bounds: satisfied (by construction)")
        else:
            obs.append("Spectral bounds: VIOLATED")

    # Adapter saturation
    sat_ratio = result.get("adapter_saturation_median_ratio")
    if sat_ratio is not None:
        pct = sat_ratio * 100
        obs.append(f"Adapter saturation: {pct:.1f}% of spectral budget used")

    # Degeneration
    degen_max = result.get("degeneration_max_ngram_repeat")
    if degen_max is not None:
        order = result.get("degeneration_ngram_order", "?")
        obs.append(f"Degeneration: max {order}-gram repeat rate = {degen_max:.3f}")

    # Pipeline gate
    gate_passed = result.get("pipeline_gate_passed")
    if gate_passed is not None:
        if gate_passed:
            obs.append("Pipeline gate: PASSED")
        else:
            failure_modes = result.get("pipeline_gate_failure_modes", [])
            obs.append(f"Pipeline gate: FAILED ({', '.join(failure_modes)})")

    # Benchmark delta
    benchmark_delta = result.get("benchmark_delta")
    if isinstance(benchmark_delta, dict):
        overall = benchmark_delta.get("overall")
        if isinstance(overall, (int, float)):
            obs.append(f"Benchmark delta: {overall:+.4f}")

    # Training stats
    iters = result.get("train_iters")
    time_s = result.get("training_time_seconds")
    if iters is not None:
        parts = [f"Training: {iters} iterations"]
        if time_s is not None:
            parts.append(f"{time_s:.0f}s")
        obs.append(", ".join(parts))

    # Pipeline gate detailed checks
    gate_checks = result.get("pipeline_gate_checks")
    obs.extend(interpret_pipeline_gate(gate_checks))

    return obs


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def suggest_next_steps(
    result: dict[str, Any],
    model_path: str | None = None,
    adapter_path: str | None = None,
) -> list[AgentRecommendation]:
    """Generate concrete next-step recommendations from training result."""
    recs: list[AgentRecommendation] = []

    # If adapter was saved, suggest evaluation
    adapter = adapter_path or result.get("adapter_path")
    if adapter and model_path:
        recs.append(
            AgentRecommendation(
                action="evaluate",
                reason="Assess inference quality on test prompts",
                command=f"mc train evaluate -m {model_path} -a {adapter}",
            )
        )

    # Gate failures
    gate_passed = result.get("pipeline_gate_passed")
    failure_modes = result.get("pipeline_gate_failure_modes", [])

    if gate_passed is False:
        if "degeneration" in str(failure_modes).lower():
            recs.append(
                AgentRecommendation(
                    action="try_different_data",
                    reason="Degeneration detected. Training data may conflict with "
                    "base model behavior. Try more diverse or less domain-specific data.",
                )
            )
        if "cka" in str(failure_modes).lower():
            recs.append(
                AgentRecommendation(
                    action="inspect_cka",
                    reason="CKA preservation failed. The adapter may be distorting "
                    "base model representations too aggressively.",
                )
            )

    # Loss didn't improve
    baseline_loss = result.get("baseline_loss")
    post_loss = result.get("post_loss")
    if baseline_loss is not None and post_loss is not None:
        if post_loss >= baseline_loss:
            recs.append(
                AgentRecommendation(
                    action="check_data_quality",
                    reason="Post-training loss is not better than baseline. "
                    "The training data may not contain learnable signal "
                    "for this model.",
                )
            )

    # Adapter saturated — cannot learn more at this rank
    sat_ratio = result.get("adapter_saturation_median_ratio")
    if sat_ratio is not None and sat_ratio >= 0.95:
        recs.append(
            AgentRecommendation(
                action="note_saturation",
                reason=f"Adapter saturation at {sat_ratio * 100:.1f}% — "
                "near the geometric limit. More data won't help at this rank.",
            )
        )

    # Safety cap hit — convergence failure
    stop_reason = result.get("stop_reason", "")
    if stop_reason.startswith("safety_cap"):
        recs.append(
            AgentRecommendation(
                action="investigate_convergence",
                reason="Safety cap hit. The stopping certificate failed to fire. "
                "This may indicate the data is too noisy or the model architecture "
                "has issues with the current adapter surface.",
            )
        )

    # Degeneration stop
    if stop_reason.startswith("degeneration_exceeded"):
        recs.append(
            AgentRecommendation(
                action="try_different_data",
                reason="Training stopped due to output degeneration. "
                "The training data may be too narrow or conflicting with "
                "base model behavior.",
            )
        )

    return recs


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _build_summary(result: dict[str, Any]) -> str:
    """Generate a one-sentence summary of the training result."""
    stop_reason = result.get("stop_reason", "unknown")
    iters = result.get("train_iters", "?")
    baseline_loss = result.get("baseline_loss")
    post_loss = result.get("post_loss")
    gate_passed = result.get("pipeline_gate_passed")

    # Loss direction
    if baseline_loss is not None and post_loss is not None:
        if post_loss < baseline_loss:
            loss_direction = "loss improved"
        else:
            loss_direction = "loss did not improve"
    else:
        loss_direction = "loss comparison unavailable"

    # Gate status
    if gate_passed is True:
        gate_text = "pipeline gate passed"
    elif gate_passed is False:
        gate_text = "pipeline gate FAILED"
    else:
        gate_text = "pipeline gate not evaluated"

    # Stop category
    stop_text = interpret_stop_reason(stop_reason).split(".")[0]

    return f"Training completed after {iters} iterations. {stop_text}. {loss_direction.capitalize()}, {gate_text}."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def diagnose_training_result(
    result: dict[str, Any],
    model_path: str | None = None,
    adapter_path: str | None = None,
) -> AgentDiagnostics:
    """Generate agent-readable diagnostics from a training result dict.

    Args:
        result: Output of ``DatasetTrainResult.to_dict()``.
        model_path: Model path for generating concrete CLI commands.
        adapter_path: Adapter path override (defaults to result's adapter_path).
    """
    return AgentDiagnostics(
        summary=_build_summary(result),
        observations=_build_observations(result),
        recommendations=suggest_next_steps(result, model_path, adapter_path),
    )
