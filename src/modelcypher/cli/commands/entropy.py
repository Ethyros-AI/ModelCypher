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

"""Entropy analysis CLI commands.

Provides commands for entropy pattern analysis, distress detection,
baseline verification, sliding window tracking, and conversation analysis.

Commands:
    mc entropy analyze <samples>
    mc entropy detect-distress <samples>
    mc entropy verify-baseline --mean ... --std-dev ... --max ... --min ... --observed ...
    mc entropy window --size <n> --threshold <t>
    mc entropy conversation-track --session <file>
    mc entropy dual-path --base <path> --adapter <path>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.entropy_probe_service import (
    EntropyProbeService,
    PatternAnalysisConfig,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def entropy_analyze(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ..., help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
    minimum_samples: int = typer.Option(..., "--min-samples", help="Minimum samples for trend"),
    trend_threshold: float = typer.Option(..., "--trend-threshold", help="Trend slope threshold"),
    distress_correlation_threshold: float = typer.Option(
        ..., "--distress-correlation-threshold", help="Entropy/variance correlation threshold"
    ),
    high_volatility_threshold: float = typer.Option(
        ..., "--high-volatility-threshold", help="Volatility threshold"
    ),
    anomaly_z_score_threshold: float = typer.Option(
        ..., "--anomaly-z-score-threshold", help="Anomaly z-score threshold"
    ),
) -> None:
    """Analyze entropy/variance samples for patterns and trends."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        sample_list = json_lib.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
        parsed_samples = [(float(s[0]), float(s[1])) for s in sample_list]
    except (json_lib.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1050",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of [entropy, variance] pairs",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    config = PatternAnalysisConfig(
        minimum_samples_for_trend=minimum_samples,
        trend_threshold=trend_threshold,
        distress_correlation_threshold=distress_correlation_threshold,
        high_volatility_threshold=high_volatility_threshold,
        anomaly_z_score_threshold=anomaly_z_score_threshold,
    )
    pattern = service.analyze_pattern(parsed_samples, config)
    payload = service.pattern_payload(pattern)

    if context.output_format == "text":
        trend_direction = "rising" if pattern.is_rising else "falling" if pattern.is_falling else "stable"
        lines = [
            "ENTROPY PATTERN ANALYSIS",
            f"Trend Slope: {pattern.trend_slope:.4f} ({trend_direction})",
            f"Volatility: {pattern.volatility:.4f}",
            f"Entropy Mean: {pattern.entropy_mean:.4f}",
            f"Entropy StdDev: {pattern.entropy_std_dev:.4f}",
            f"Variance Mean: {pattern.variance_mean:.4f}",
            f"Entropy-Variance Correlation: {pattern.entropy_variance_correlation:.4f}",
            f"Sustained High Count: {pattern.sustained_high_count}",
            f"Sustained Significance: {pattern.sustained_significance:.2f}",
            f"Peak Entropy: {pattern.peak_entropy:.4f}",
            f"Min Entropy: {pattern.min_entropy:.4f}",
            f"Sample Count: {pattern.sample_count}",
        ]
        if pattern.anomaly_indices:
            lines.append(f"Anomaly Indices: {list(pattern.anomaly_indices)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect-distress")
def entropy_detect_distress(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ..., help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
    minimum_samples: int = typer.Option(..., "--min-samples", help="Minimum samples for trend"),
    trend_threshold: float = typer.Option(..., "--trend-threshold", help="Trend slope threshold"),
    distress_correlation_threshold: float = typer.Option(
        ..., "--distress-correlation-threshold", help="Entropy/variance correlation threshold"
    ),
    high_volatility_threshold: float = typer.Option(
        ..., "--high-volatility-threshold", help="Volatility threshold"
    ),
    anomaly_z_score_threshold: float = typer.Option(
        ..., "--anomaly-z-score-threshold", help="Anomaly z-score threshold"
    ),
) -> None:
    """Detect distress patterns in entropy/variance samples."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        sample_list = json_lib.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
        parsed_samples = [(float(s[0]), float(s[1])) for s in sample_list]
    except (json_lib.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1051",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of [entropy, variance] pairs",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    config = PatternAnalysisConfig(
        minimum_samples_for_trend=minimum_samples,
        trend_threshold=trend_threshold,
        distress_correlation_threshold=distress_correlation_threshold,
        high_volatility_threshold=high_volatility_threshold,
        anomaly_z_score_threshold=anomaly_z_score_threshold,
    )
    distress = service.detect_distress(parsed_samples, config)
    payload = service.distress_payload(distress)

    if context.output_format == "text":
        if distress is None:
            write_output("No distress detected", context.output_format, context.pretty)
            return
        lines = [
            "DISTRESS DETECTION RESULT",
            "Detected: YES",
            f"Confidence: {distress.confidence:.2%}",
            f"Sustained High Count: {distress.sustained_high_count}",
            f"Average Entropy: {distress.average_entropy:.4f}",
            f"Average Variance: {distress.average_variance:.4f}",
            f"Correlation: {distress.correlation:.4f}",
            f"Indicators: {', '.join(distress.indicators)}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("verify-baseline")
def entropy_verify_baseline(
    ctx: typer.Context,
    declared_mean: float = typer.Option(..., "--mean", help="Declared delta mean"),
    declared_std_dev: float = typer.Option(
        ..., "--std-dev", help="Declared delta standard deviation"
    ),
    declared_max: float = typer.Option(..., "--max", help="Declared maximum delta"),
    declared_min: float = typer.Option(..., "--min", help="Declared minimum delta"),
    observed_deltas: str = typer.Option(
        ..., "--observed", help="JSON array of observed delta values, e.g. '[0.1, 0.15, 0.12]'"
    ),
    test_prompts: str = typer.Option(
        ..., "--test-prompts", help="JSON array of verification prompts"
    ),
    failure_z_score: float = typer.Option(..., "--failure-z-score", help="Failure z-score"),
    suspicious_z_score: float = typer.Option(
        ..., "--suspicious-z-score", help="Suspicious z-score"
    ),
    minimum_sample_count: int = typer.Option(
        ..., "--minimum-samples", help="Minimum sample count"
    ),
    include_adversarial: bool = typer.Option(
        ..., "--include-adversarial/--no-include-adversarial", help="Include adversarial prompts"
    ),
    max_tokens_per_prompt: int = typer.Option(
        ..., "--max-tokens-per-prompt", help="Max tokens per prompt"
    ),
    temperature: float = typer.Option(..., "--temperature", help="Generation temperature"),
    prompt_timeout_seconds: float = typer.Option(
        ..., "--prompt-timeout-seconds", help="Timeout per prompt in seconds"
    ),
    base_model_id: str = typer.Option(..., "--base-model", help="Base model identifier"),
    adapter_path: str = typer.Option(..., "--adapter", help="Path to adapter"),
) -> None:
    """Verify observed entropy deltas against declared baseline."""
    context = _context(ctx)
    import json as json_lib

    service = EntropyProbeService()

    try:
        deltas = json_lib.loads(observed_deltas)
        if not isinstance(deltas, list):
            raise ValueError("Observed deltas must be a JSON array")
        parsed_deltas = [float(d) for d in deltas]
        prompt_values = json_lib.loads(test_prompts)
        if not isinstance(prompt_values, list) or not all(
            isinstance(p, str) for p in prompt_values
        ):
            raise ValueError("Test prompts must be a JSON array of strings")
        prompt_tuple = tuple(prompt_values)
    except (json_lib.JSONDecodeError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1052",
            title="Invalid deltas format",
            detail=str(exc),
            hint="Provide observed deltas as JSON array of numbers",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    from modelcypher.core.domain.entropy.baseline_verification_probe import (
        VerificationConfiguration,
    )

    config = VerificationConfiguration.with_statistical_thresholds(
        failure_z_score=failure_z_score,
        suspicious_z_score=suspicious_z_score,
        test_prompts=prompt_tuple,
        include_adversarial=include_adversarial,
        max_tokens_per_prompt=max_tokens_per_prompt,
        minimum_sample_count=minimum_sample_count,
        temperature=temperature,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )

    result = service.verify_baseline(
        declared_mean=declared_mean,
        declared_std_dev=declared_std_dev,
        declared_max=declared_max,
        declared_min=declared_min,
        observed_deltas=parsed_deltas,
        config=config,
        base_model_id=base_model_id,
        adapter_path=adapter_path,
    )
    payload = service.verification_payload(result)

    if context.output_format == "text":
        lines = [
            "BASELINE VERIFICATION",
            f"Declared Δ mean: {result.declared_baseline.delta_mean:.3f} ± {result.declared_baseline.delta_std_dev:.3f}",
            f"Declared Δ range: [{result.declared_baseline.delta_min:.3f}, {result.declared_baseline.delta_max:.3f}]",
            f"Observed Δ mean: {result.observed_baseline.delta_mean:.3f} ± {result.observed_baseline.delta_std_dev:.3f}",
            f"Observed Δ range: [{result.observed_baseline.delta_min:.3f}, {result.observed_baseline.delta_max:.3f}]",
            f"Mean Z-score: {result.comparison.mean_z_score:.2f}",
            f"StdDev ratio: {result.comparison.std_dev_ratio:.2f}",
            f"Range exceeded: {result.comparison.range_exceeded}",
            f"Divergence score: {result.comparison.divergence_score:.3f}",
            f"Sample count sufficient: {result.comparison.sample_count_sufficient}",
            f"Total samples: {result.total_samples}",
            f"Adversarial flags: {len(result.adversarial_flags)}",
            f"Duration: {result.verification_duration:.1f}s",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("window")
def entropy_window(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ..., help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
    size: int = typer.Option(..., "--size", help="Window size for sliding analysis"),
    minimum_samples: int = typer.Option(..., "--minimum-samples", help="Minimum samples"),
    sustained_high_count: int = typer.Option(
        ..., "--sustained-high-count", help="Consecutive high samples threshold"
    ),
    high_threshold: float = typer.Option(..., "--high-threshold", help="High entropy threshold"),
    circuit_threshold: float = typer.Option(
        ..., "--circuit-threshold", help="Circuit breaker threshold"
    ),
) -> None:
    """Analyze entropy using a sliding window tracker.

    Provides real-time entropy monitoring with explicit thresholds
    for detecting anomalies and state transitions.

    Examples:
        mc entropy window '[[3.5, 0.2], [3.6, 0.1], [4.8, 0.5]]' --size 50 --minimum-samples 5 --sustained-high-count 3 --high-threshold 4.0 --circuit-threshold 5.0
    """
    context = _context(ctx)

    from modelcypher.core.domain.entropy.entropy_window import (
        EntropyWindow,
        EntropyWindowConfig,
    )

    try:
        sample_list = json.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
        parsed_samples = [(float(s[0]), float(s[1])) for s in sample_list]
    except (json.JSONDecodeError, IndexError, TypeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1053",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of [entropy, variance] pairs",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    config = EntropyWindowConfig(
        window_size=size,
        minimum_samples=minimum_samples,
        high_entropy_threshold=high_threshold,
        circuit_breaker_threshold=circuit_threshold,
        sustained_high_count=sustained_high_count,
    )
    window = EntropyWindow(config=config)

    # Add all samples to the window
    for idx, (entropy, variance) in enumerate(parsed_samples):
        window.add(entropy=entropy, variance=variance, token_index=idx)

    status = window.status()

    payload = {
        "windowSize": size,
        "sampleCount": status.sample_count,
        "currentEntropy": status.current_entropy,
        "movingAverage": status.moving_average,
        "maxEntropy": status.max_entropy,
        "minEntropy": status.min_entropy,
        "consecutiveHighCount": status.consecutive_high_count,
        "shouldTripCircuitBreaker": status.should_trip_circuit_breaker,
        "tokenStart": status.token_start,
        "tokenEnd": status.token_end,
    }

    if context.output_format == "text":
        lines = [
            "ENTROPY WINDOW ANALYSIS",
            f"Window Size: {size}",
            f"Samples Analyzed: {status.sample_count}",
            "",
            f"Current Entropy: {status.current_entropy:.4f}",
            f"Moving Average: {status.moving_average:.4f}",
            f"Max Entropy: {status.max_entropy:.4f}",
            f"Min Entropy: {status.min_entropy:.4f}",
            f"Consecutive High Count: {status.consecutive_high_count}",
            f"Circuit Breaker: {'TRIPPED' if status.should_trip_circuit_breaker else 'OK'}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("conversation-track")
def entropy_conversation_track(
    ctx: typer.Context,
    session: str = typer.Option(..., "--session", help="Path to session file (JSON with turns)"),
    oscillation_threshold: float = typer.Option(
        ..., "--oscillation-threshold", help="Oscillation amplitude threshold"
    ),
    drift_threshold: float = typer.Option(
        ..., "--drift-threshold", help="Cumulative drift threshold"
    ),
    turn_spike_threshold: float = typer.Option(
        ..., "--turn-spike-threshold", help="Turn-over-turn spike threshold"
    ),
    oscillation_window_size: int = typer.Option(
        ..., "--oscillation-window-size", help="Oscillation window size (turns)"
    ),
    minimum_turns_for_analysis: int = typer.Option(
        ..., "--minimum-turns", help="Minimum turns for analysis"
    ),
    recency_decay: float = typer.Option(
        ..., "--recency-decay", help="Recency decay factor"
    ),
) -> None:
    """Track entropy patterns across a conversation session.

    Analyzes multi-turn conversations for oscillation patterns,
    cumulative drift, and manipulation signals. Session file format:
    {
        "turns": [
            {"token_count": 100, "avg_delta": 0.1, "anomaly_count": 0},
            {"token_count": 50, "avg_delta": 0.15, "anomaly_count": 1}
        ]
    }

    Examples:
        mc entropy conversation-track --session ./session.json --oscillation-threshold 1.0 --drift-threshold 1.5 --turn-spike-threshold 0.4 --oscillation-window-size 5 --minimum-turns 3 --recency-decay 0.9
    """
    context = _context(ctx)

    from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
        ConversationEntropyConfiguration,
        ConversationEntropyTracker,
    )

    session_path = Path(session)
    if not session_path.exists():
        error = ErrorDetail(
            code="MC-1054",
            title="Session file not found",
            detail=f"Session file does not exist: {session}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except json.JSONDecodeError as exc:
        error = ErrorDetail(
            code="MC-1055",
            title="Invalid session format",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse turns from session data
    turns = session_data.get("turns", [])
    if not turns:
        error = ErrorDetail(
            code="MC-1056",
            title="Empty session",
            detail="Session file contains no turns",
            hint="Session file should have a 'turns' array with 'token_count', 'avg_delta' fields",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    config = ConversationEntropyConfiguration.with_thresholds(
        oscillation_threshold=oscillation_threshold,
        drift_threshold=drift_threshold,
        turn_spike_threshold=turn_spike_threshold,
        oscillation_window_size=oscillation_window_size,
        minimum_turns_for_analysis=minimum_turns_for_analysis,
        recency_decay=recency_decay,
    )
    tracker = ConversationEntropyTracker(configuration=config)

    # Process turns
    assessment = None
    from datetime import datetime

    for turn in turns:
        if "token_count" not in turn or "avg_delta" not in turn:
            raise ValueError(
                "Each turn must include 'token_count' and 'avg_delta' fields"
            )
        if "max_anomaly_score" not in turn or "anomaly_count" not in turn:
            raise ValueError(
                "Each turn must include 'max_anomaly_score' and 'anomaly_count' fields"
            )
        if "circuit_breaker_tripped" not in turn or "security_assessment" not in turn:
            raise ValueError(
                "Each turn must include 'circuit_breaker_tripped' and 'security_assessment' fields"
            )
        if "timestamp" not in turn:
            raise ValueError("Each turn must include ISO 'timestamp' field")

        token_count = int(turn["token_count"])
        avg_delta = float(turn["avg_delta"])
        max_anomaly_score = float(turn["max_anomaly_score"])
        anomaly_count = int(turn["anomaly_count"])
        circuit_breaker_tripped = bool(turn["circuit_breaker_tripped"])
        security_assessment = str(turn["security_assessment"])
        timestamp = datetime.fromisoformat(str(turn["timestamp"]))

        assessment = tracker.record_turn(
            token_count=token_count,
            avg_delta=avg_delta,
            max_anomaly_score=max_anomaly_score,
            anomaly_count=anomaly_count,
            circuit_breaker_tripped=circuit_breaker_tripped,
            security_assessment=security_assessment,
            timestamp=timestamp,
        )

    if assessment is None:
        error = ErrorDetail(
            code="MC-1056",
            title="No assessment",
            detail="No turns could be processed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    mc = assessment.manipulation_components
    payload = {
        "sessionPath": str(session_path),
        "turnCount": assessment.turn_count,
        "oscillationAmplitude": assessment.oscillation_amplitude,
        "oscillationFrequency": assessment.oscillation_frequency,
        "cumulativeDrift": assessment.cumulative_drift,
        "recentAnomalyCount": assessment.recent_anomaly_count,
        "assessmentConfidence": assessment.assessment_confidence,
        "isSufficientData": assessment.is_sufficient_data,
        "manipulationComponents": {
            "oscillationAmplitudeScore": mc.oscillation_amplitude_score,
            "oscillationFrequencyScore": mc.oscillation_frequency_score,
            "driftScore": mc.drift_score,
            "anomalyScore": mc.anomaly_score,
            "spikeScore": mc.spike_score,
            "circuitBreakerTripped": mc.circuit_breaker_tripped,
            "baselineOscillationExceeded": mc.baseline_oscillation_exceeded,
        },
    }

    if context.output_format == "text":
        lines = [
            "CONVERSATION ENTROPY TRACKING",
            f"Session: {session_path}",
            f"Turns Analyzed: {assessment.turn_count}",
            f"Sufficient Data: {'YES' if assessment.is_sufficient_data else 'NO'}",
            "",
            f"Oscillation Amplitude: {assessment.oscillation_amplitude:.4f}",
            f"Oscillation Frequency: {assessment.oscillation_frequency:.4f}",
            f"Cumulative Drift: {assessment.cumulative_drift:.4f}",
            f"Recent Anomalies: {assessment.recent_anomaly_count}",
            f"Confidence: {assessment.assessment_confidence:.2%}",
            "",
            "MANIPULATION SIGNAL COMPONENTS:",
            f"  Oscillation Amplitude Score: {mc.oscillation_amplitude_score:.4f}",
            f"  Oscillation Frequency Score: {mc.oscillation_frequency_score:.4f}",
            f"  Drift Score: {mc.drift_score:.4f}",
            f"  Anomaly Score: {mc.anomaly_score:.4f}",
            f"  Spike Score: {mc.spike_score:.4f}",
            f"  Circuit Breaker: {'TRIPPED' if mc.circuit_breaker_tripped else 'OK'}",
            f"  Baseline Oscillation Exceeded: {'YES' if mc.baseline_oscillation_exceeded else 'NO'}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dual-path")
def entropy_dual_path(
    ctx: typer.Context,
    samples: str = typer.Argument(..., help="JSON array of {base: [e, v], adapter: [e, v]} pairs"),
    anomaly_threshold: float = typer.Option(
        ..., "--anomaly-threshold", help="Anomaly score threshold"
    ),
    delta_threshold: float = typer.Option(
        ..., "--delta-threshold", help="Entropy delta threshold"
    ),
    base_entropy_floor: float = typer.Option(
        ..., "--base-entropy-floor", help="Base entropy floor for anomaly scoring"
    ),
) -> None:
    """Analyze entropy divergence between base model and adapter.

    Compares entropy patterns from base model and adapted model
    to detect suspicious divergence that may indicate backdoors.
    Input format: [{"base": [entropy, variance], "adapter": [entropy, variance]}]

    Anomaly scoring: High base entropy + low adapter entropy = suspicious

    Examples:
        mc entropy dual-path '[{"base": [3.5, 0.2], "adapter": [3.8, 0.3]}]' --anomaly-threshold 0.5 --delta-threshold 1.0 --base-entropy-floor 2.0
    """
    context = _context(ctx)

    try:
        sample_list = json.loads(samples)
        if not isinstance(sample_list, list):
            raise ValueError("Samples must be a JSON array")
    except json.JSONDecodeError as exc:
        error = ErrorDetail(
            code="MC-1057",
            title="Invalid samples format",
            detail=str(exc),
            hint="Provide samples as JSON array of {base: [e, v], adapter: [e, v]} objects",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Analyze samples manually (EntropyDeltaTracker requires MLX arrays for live tracking)
    deltas: list[float] = []
    anomaly_scores: list[float] = []
    anomaly_indices: list[int] = []

    for idx, sample in enumerate(sample_list):
        if "base" not in sample or "adapter" not in sample:
            raise ValueError("Each sample must include 'base' and 'adapter' entries")
        base = sample["base"]
        adapter = sample["adapter"]

        base_entropy = float(base[0])
        adapter_entropy = float(adapter[0])

        delta = adapter_entropy - base_entropy
        deltas.append(delta)

        # Anomaly score: high when base is uncertain but adapter is confident
        # (potential backdoor signature)
        if base_entropy > base_entropy_floor and adapter_entropy < base_entropy:
            # Normalized score based on entropy reduction
            anomaly_score = min(1.0, (base_entropy - adapter_entropy) / base_entropy)
        else:
            anomaly_score = 0.0

        anomaly_scores.append(anomaly_score)

        if anomaly_score >= anomaly_threshold:
            anomaly_indices.append(idx)

    # Compute statistics
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    max_anomaly = max(anomaly_scores) if anomaly_scores else 0.0
    anomaly_count = len(anomaly_indices)

    payload = {
        "sampleCount": len(sample_list),
        "averageDelta": avg_delta,
        "maxAnomalyScore": max_anomaly,
        "anomalyCount": anomaly_count,
        "anomalyIndices": anomaly_indices,
        "anomalyThreshold": anomaly_threshold,
    }

    if context.output_format == "text":
        lines = [
            "DUAL-PATH ENTROPY ANALYSIS",
            f"Samples Analyzed: {len(sample_list)}",
            "",
            f"Average Delta: {avg_delta:.4f}",
            f"Max Anomaly Score: {max_anomaly:.4f}",
            f"Anomaly Count: {anomaly_count}",
            f"Anomaly Threshold: {anomaly_threshold:.4f}",
        ]
        if anomaly_indices:
            lines.append(f"Anomaly Indices: {anomaly_indices}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("calibrate")
def entropy_calibrate(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    output: str = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to save calibration JSON (optional)",
    ),
    prompts: str = typer.Option(..., "--prompts", help="Path to prompts JSON array"),
    max_tokens: int = typer.Option(..., "--max-tokens", help="Max tokens per prompt"),
    temperature: float = typer.Option(..., "--temperature", help="Sampling temperature"),
) -> None:
    """Calibrate entropy thresholds by measuring actual model distributions.

    Runs calibration prompts through the model, captures logits,
    computes Shannon entropy, and derives empirical thresholds.

    Examples:
        mc entropy calibrate --model /path/to/model --prompts ./prompts.json --max-tokens 100 --temperature 0.7
        mc entropy calibrate --model /path/to/model --prompts ./prompts.json --max-tokens 100 --temperature 0.7 --output-file ./calibration.json
    """
    context = _context(ctx)

    from modelcypher.core.use_cases.entropy_calibration_service import (
        EntropyCalibrationService,
    )

    service = EntropyCalibrationService()

    try:
        prompt_path = Path(prompts)
        if not prompt_path.exists():
            raise ValueError(f"Prompts file does not exist: {prompt_path}")
        prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(prompt_data, list) or not all(
            isinstance(p, str) for p in prompt_data
        ):
            raise ValueError("Prompts file must contain a JSON array of strings")
        prompt_tuple = tuple(prompt_data)

        result = service.calibrate(
            model_path=model,
            prompts=prompt_tuple,
            max_tokens_per_prompt=max_tokens,
            temperature=temperature,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1055",
            title="Calibration failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        error = ErrorDetail(
            code="MC-1056",
            title="Calibration runtime error",
            detail=str(exc),
            hint="Ensure MLX and mlx-lm are installed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if output is not None:
        try:
            service.save_calibration(result, output)
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1057",
                title="Failed to save calibration",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    payload = {
        "modelId": result.model_id,
        "vocabSize": result.vocab_size,
        "maxTheoreticalEntropy": result.max_theoretical_entropy,
        "sampleCount": result.sample_count,
        "promptCount": result.prompt_count,
        "calibrationPrompts": list(result.calibration_prompts),
        "statistics": {
            "mean": result.mean,
            "stdDev": result.std_dev,
            "min": result.min_value,
            "max": result.max_value,
            "percentile25": result.percentile_25,
            "percentile50": result.percentile_50,
            "percentile75": result.percentile_75,
            "percentile95": result.percentile_95,
        },
        "calibrationDurationSeconds": result.calibration_duration_seconds,
        "calibratedAt": result.calibrated_at,
        "outputPath": output,
    }

    if context.output_format == "text":
        lines = [
            "ENTROPY CALIBRATION COMPLETE",
            "",
            f"Model: {result.model_id}",
            f"Vocab Size: {result.vocab_size}",
            f"Max Theoretical Entropy: {result.max_theoretical_entropy:.3f}",
            "",
            f"Samples Collected: {result.sample_count}",
            f"Prompts Used: {result.prompt_count}",
            f"Duration: {result.calibration_duration_seconds:.1f}s",
            "",
            "MEASURED STATISTICS:",
            f"  Mean:     {result.mean:.4f}",
            f"  Std Dev:  {result.std_dev:.4f}",
            f"  Min:      {result.min_value:.4f}",
            f"  Max:      {result.max_value:.4f}",
            "",
            "PERCENTILES:",
            f"  10th: {result.percentile_10:.4f}",
            f"  25th: {result.percentile_25:.4f}",
            f"  50th: {result.percentile_50:.4f} (median)",
            f"  75th: {result.percentile_75:.4f}",
            f"  90th: {result.percentile_90:.4f}",
            f"  95th: {result.percentile_95:.4f}",
            f"  99th: {result.percentile_99:.4f}",
        ]
        if output is not None:
            lines.extend(["", f"Calibration saved to: {output}"])
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
