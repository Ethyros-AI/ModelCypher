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

All commands return raw statistics computed from the data itself.
No user-configurable thresholds - the geometry IS the signal.

Commands:
    mc entropy analyze <samples>
    mc entropy detect-distress <samples>
    mc entropy verify-baseline --mean ... --std-dev ... --max ... --min ... --observed ...
    mc entropy window <samples> --size <n>
    mc entropy conversation-track --session <file>
    mc entropy dual-path <samples>
    mc entropy calibrate --model <path> --prompts <path> --max-tokens <n> --temperature <t>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
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
) -> None:
    """Analyze entropy/variance samples - returns raw statistics.

    No thresholds required. The data's own statistics are the signal.
    """
    context = _context(ctx)
    import json as json_lib
    import math

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

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

    if not parsed_samples:
        write_output({"error": "No samples provided"}, context.output_format, context.pretty)
        return

    # Extract entropy and variance series
    entropies = [s[0] for s in parsed_samples]
    variances = [s[1] for s in parsed_samples]
    n = len(entropies)

    # Compute raw statistics from the data itself
    entropy_mean = sum(entropies) / n
    variance_mean = sum(variances) / n
    entropy_std = math.sqrt(sum((e - entropy_mean) ** 2 for e in entropies) / n) if n > 1 else 0.0
    variance_std = math.sqrt(sum((v - variance_mean) ** 2 for v in variances) / n) if n > 1 else 0.0

    # Trend via linear regression slope
    if n > 1:
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (e - entropy_mean) for i, e in enumerate(entropies))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        trend_slope = numerator / denominator if denominator > 0 else 0.0
    else:
        trend_slope = 0.0

    # Z-scores for each sample (computed from the data's own distribution)
    z_scores = [(e - entropy_mean) / entropy_std if entropy_std > 0 else 0.0 for e in entropies]

    payload = {
        "sampleCount": n,
        # Raw distribution statistics
        "entropyMean": entropy_mean,
        "entropyStdDev": entropy_std,
        "entropyMin": min(entropies),
        "entropyMax": max(entropies),
        "varianceMean": variance_mean,
        "varianceStdDev": variance_std,
        # Trend (slope of linear fit)
        "trendSlope": trend_slope,
        # Z-scores relative to this data's own distribution
        "zScores": z_scores,
    }

    if context.output_format == "text":
        lines = [
            "ENTROPY ANALYSIS (raw statistics)",
            f"Sample Count: {n}",
            f"Entropy: mean={entropy_mean:.4f}, std={entropy_std:.4f}, min={min(entropies):.4f}, max={max(entropies):.4f}",
            f"Variance: mean={variance_mean:.4f}, std={variance_std:.4f}",
            f"Trend Slope: {trend_slope:.6f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect-distress")
def entropy_detect_distress(
    ctx: typer.Context,
    samples: str = typer.Argument(
        ..., help="JSON array of [entropy, variance] pairs, e.g. '[[3.5, 0.2], [3.6, 0.1]]'"
    ),
) -> None:
    """Analyze entropy/variance samples for distress indicators.

    Returns raw distress metrics computed from the data itself:
    - Trend slope (linear regression)
    - Entropy-variance correlation (Pearson)
    - Volatility (standard deviation of differences)
    - Z-scores for each sample (from data's own distribution)

    No thresholds - the measurements ARE the signal.
    """
    context = _context(ctx)
    import json as json_lib
    import math

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

    if not parsed_samples:
        write_output({"error": "No samples provided"}, context.output_format, context.pretty)
        return

    n = len(parsed_samples)
    entropies = [s[0] for s in parsed_samples]
    variances = [s[1] for s in parsed_samples]

    # Basic statistics
    entropy_mean = sum(entropies) / n
    variance_mean = sum(variances) / n
    entropy_std = math.sqrt(sum((e - entropy_mean) ** 2 for e in entropies) / n) if n > 1 else 0.0
    variance_std = math.sqrt(sum((v - variance_mean) ** 2 for v in variances) / n) if n > 1 else 0.0

    # Trend slope (linear regression on entropy)
    if n > 1:
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (e - entropy_mean) for i, e in enumerate(entropies))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        trend_slope = numerator / denominator if denominator > 0 else 0.0
    else:
        trend_slope = 0.0

    # Pearson correlation between entropy and variance
    if n > 1 and entropy_std > 0 and variance_std > 0:
        covariance = sum(
            (e - entropy_mean) * (v - variance_mean) for e, v in parsed_samples
        ) / n
        correlation = covariance / (entropy_std * variance_std)
    else:
        correlation = 0.0

    # Volatility (std of consecutive differences)
    if n > 1:
        diffs = [entropies[i + 1] - entropies[i] for i in range(n - 1)]
        diff_mean = sum(diffs) / len(diffs)
        volatility = math.sqrt(sum((d - diff_mean) ** 2 for d in diffs) / len(diffs))
    else:
        volatility = 0.0

    # Z-scores for each sample
    z_scores = [(e - entropy_mean) / entropy_std if entropy_std > 0 else 0.0 for e in entropies]

    payload = {
        "sampleCount": n,
        "entropyMean": entropy_mean,
        "entropyStdDev": entropy_std,
        "varianceMean": variance_mean,
        "varianceStdDev": variance_std,
        "trendSlope": trend_slope,
        "correlation": correlation,
        "volatility": volatility,
        "zScores": z_scores,
    }

    if context.output_format == "text":
        lines = [
            "DISTRESS ANALYSIS (raw metrics)",
            f"Sample Count: {n}",
            f"Entropy: mean={entropy_mean:.4f}, std={entropy_std:.4f}",
            f"Variance: mean={variance_mean:.4f}, std={variance_std:.4f}",
            f"Trend Slope: {trend_slope:.6f}",
            f"Entropy-Variance Correlation: {correlation:.4f}",
            f"Volatility: {volatility:.4f}",
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
) -> None:
    """Compare observed entropy deltas against declared baseline.

    Returns raw statistical comparison metrics:
    - Z-score of observed mean vs declared mean
    - Ratio of observed std vs declared std
    - Declared vs observed ranges
    - Per-sample z-scores

    No interpretation thresholds - the statistics ARE the signal.
    """
    context = _context(ctx)
    import json as json_lib
    import math

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

    try:
        deltas = json_lib.loads(observed_deltas)
        if not isinstance(deltas, list):
            raise ValueError("Observed deltas must be a JSON array")
        parsed_deltas = [float(d) for d in deltas]
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

    if not parsed_deltas:
        write_output({"error": "No observed deltas provided"}, context.output_format, context.pretty)
        return

    n = len(parsed_deltas)

    # Compute observed statistics
    observed_mean = sum(parsed_deltas) / n
    observed_std = math.sqrt(sum((d - observed_mean) ** 2 for d in parsed_deltas) / n) if n > 1 else 0.0
    observed_min = min(parsed_deltas)
    observed_max = max(parsed_deltas)

    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))

    mean_z_score = (observed_mean - declared_mean) / max(declared_std_dev, eps)
    std_dev_ratio = observed_std / max(declared_std_dev, eps)

    max_deviation = abs(observed_max - declared_max)
    min_deviation = abs(observed_min - declared_min)
    declared_range = abs(declared_max - declared_min)
    observed_range = abs(observed_max - observed_min)

    sample_z_scores = [(d - declared_mean) / max(declared_std_dev, eps) for d in parsed_deltas]

    payload = {
        "sampleCount": n,
        "declared": {
            "mean": declared_mean,
            "stdDev": declared_std_dev,
            "min": declared_min,
            "max": declared_max,
        },
        "observed": {
            "mean": observed_mean,
            "stdDev": observed_std,
            "min": observed_min,
            "max": observed_max,
        },
        "meanZScore": mean_z_score,
        "stdDevRatio": std_dev_ratio,
        "maxDeviation": max_deviation,
        "minDeviation": min_deviation,
        "declaredRange": declared_range,
        "observedRange": observed_range,
        "sampleZScores": sample_z_scores,
    }

    if context.output_format == "text":
        lines = [
            "BASELINE VERIFICATION (raw comparison)",
            f"Sample Count: {n}",
            "",
            f"Declared: mean={declared_mean:.4f}, std={declared_std_dev:.4f}, range=[{declared_min:.4f}, {declared_max:.4f}]",
            f"Observed: mean={observed_mean:.4f}, std={observed_std:.4f}, range=[{observed_min:.4f}, {observed_max:.4f}]",
            "",
            f"Mean Z-score: {mean_z_score:.4f}",
            f"StdDev Ratio: {std_dev_ratio:.4f}",
            f"Max Deviation: {max_deviation:.4f}",
            f"Min Deviation: {min_deviation:.4f}",
            f"Declared Range: {declared_range:.4f}",
            f"Observed Range: {observed_range:.4f}",
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
    size: int = typer.Option(50, "--size", help="Window size for sliding analysis"),
) -> None:
    """Analyze entropy using a sliding window.

    Returns raw window statistics:
    - Moving average over window
    - Min/max in window
    - Standard deviation
    - Z-scores for each sample (from window's own distribution)
    - Consecutive runs above 1σ

    No thresholds - the window statistics ARE the signal.

    Examples:
        mc entropy window '[[3.5, 0.2], [3.6, 0.1], [4.8, 0.5]]' --size 50
    """
    context = _context(ctx)
    import math

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

    if not parsed_samples:
        write_output({"error": "No samples provided"}, context.output_format, context.pretty)
        return

    # Extract entropies and use sliding window
    entropies = [s[0] for s in parsed_samples]
    variances = [s[1] for s in parsed_samples]
    n = len(entropies)

    # Use window or full data if smaller
    window_data = entropies[-size:] if len(entropies) > size else entropies
    window_n = len(window_data)

    # Window statistics
    window_mean = sum(window_data) / window_n
    window_std = math.sqrt(sum((e - window_mean) ** 2 for e in window_data) / window_n) if window_n > 1 else 0.0
    window_min = min(window_data)
    window_max = max(window_data)

    # Z-scores for each sample in window
    z_scores = [(e - window_mean) / window_std if window_std > 0 else 0.0 for e in window_data]

    # Current (most recent) values
    current_entropy = entropies[-1] if entropies else 0.0
    current_variance = variances[-1] if variances else 0.0
    current_z = z_scores[-1] if z_scores else 0.0

    payload = {
        "windowSize": size,
        "actualWindowSize": window_n,
        "totalSamples": n,
        "currentEntropy": current_entropy,
        "currentVariance": current_variance,
        "currentZScore": current_z,
        "windowMean": window_mean,
        "windowStdDev": window_std,
        "windowMin": window_min,
        "windowMax": window_max,
        "zScores": z_scores,
    }

    if context.output_format == "text":
        lines = [
            "ENTROPY WINDOW ANALYSIS (raw statistics)",
            f"Window Size: {window_n} (of {n} total samples)",
            "",
            f"Current Entropy: {current_entropy:.4f}",
            f"Current Z-score: {current_z:.4f}",
            "",
            f"Window Mean: {window_mean:.4f}",
            f"Window StdDev: {window_std:.4f}",
            f"Window Range: [{window_min:.4f}, {window_max:.4f}]",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("conversation-track")
def entropy_conversation_track(
    ctx: typer.Context,
    session: str = typer.Option(..., "--session", help="Path to session file (JSON with turns)"),
) -> None:
    """Analyze entropy patterns across a conversation session.

    Returns raw conversation statistics:
    - Oscillation amplitude and frequency (from data)
    - Cumulative drift
    - Turn-over-turn deltas
    - Z-scores for each turn (from conversation's own distribution)

    Session file format (simplified):
    {
        "turns": [
            {"avg_delta": 0.1},
            {"avg_delta": 0.15}
        ]
    }

    No thresholds - the conversation statistics ARE the signal.

    Examples:
        mc entropy conversation-track --session ./session.json
    """
    context = _context(ctx)
    import math

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
            hint="Session file should have a 'turns' array with 'avg_delta' field",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Extract deltas from turns
    deltas: list[float] = []
    for turn in turns:
        if "avg_delta" not in turn:
            error = ErrorDetail(
                code="MC-1057",
                title="Missing field",
                detail="Each turn must include 'avg_delta' field",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        deltas.append(float(turn["avg_delta"]))

    n = len(deltas)

    # Basic statistics
    delta_mean = sum(deltas) / n
    delta_std = math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / n) if n > 1 else 0.0

    # Cumulative drift (sum of deltas)
    cumulative_drift = sum(deltas)

    # Turn-over-turn changes
    turn_changes = [deltas[i + 1] - deltas[i] for i in range(n - 1)] if n > 1 else []
    max_spike = max(abs(c) for c in turn_changes) if turn_changes else 0.0

    # Oscillation: count sign changes in turn_changes
    sign_changes = 0
    for i in range(len(turn_changes) - 1):
        if turn_changes[i] * turn_changes[i + 1] < 0:
            sign_changes += 1
    oscillation_frequency = sign_changes / len(turn_changes) if turn_changes else 0.0

    # Oscillation amplitude: std of turn_changes
    if turn_changes:
        tc_mean = sum(turn_changes) / len(turn_changes)
        oscillation_amplitude = math.sqrt(sum((c - tc_mean) ** 2 for c in turn_changes) / len(turn_changes))
    else:
        oscillation_amplitude = 0.0

    # Z-scores for each turn
    z_scores = [(d - delta_mean) / delta_std if delta_std > 0 else 0.0 for d in deltas]

    payload = {
        "sessionPath": str(session_path),
        "turnCount": n,
        "deltaMean": delta_mean,
        "deltaStdDev": delta_std,
        "deltaMin": min(deltas),
        "deltaMax": max(deltas),
        "cumulativeDrift": cumulative_drift,
        "oscillationAmplitude": oscillation_amplitude,
        "oscillationFrequency": oscillation_frequency,
        "maxTurnSpike": max_spike,
        "turnZScores": z_scores,
        "turnChanges": turn_changes,
    }

    if context.output_format == "text":
        lines = [
            "CONVERSATION ENTROPY TRACKING (raw statistics)",
            f"Session: {session_path}",
            f"Turns Analyzed: {n}",
            "",
            f"Delta: mean={delta_mean:.4f}, std={delta_std:.4f}, range=[{min(deltas):.4f}, {max(deltas):.4f}]",
            f"Cumulative Drift: {cumulative_drift:.4f}",
            f"Oscillation Amplitude: {oscillation_amplitude:.4f}",
            f"Oscillation Frequency: {oscillation_frequency:.4f}",
            f"Max Turn Spike: {max_spike:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dual-path")
def entropy_dual_path(
    ctx: typer.Context,
    samples: str = typer.Argument(..., help="JSON array of {base: [e, v], adapter: [e, v]} pairs"),
) -> None:
    """Analyze entropy divergence between base model and adapter.

    Returns raw dual-path statistics:
    - Delta (adapter - base) for each sample
    - Distribution statistics (mean, std, min, max)
    - Z-scores for each sample
    - Entropy reduction ratio where adapter < base

    No thresholds - the deltas ARE the signal.

    Input format: [{"base": [entropy, variance], "adapter": [entropy, variance]}]

    Examples:
        mc entropy dual-path '[{"base": [3.5, 0.2], "adapter": [3.8, 0.3]}]'
    """
    context = _context(ctx)
    import math

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

    if not sample_list:
        write_output({"error": "No samples provided"}, context.output_format, context.pretty)
        return

    # Extract base and adapter entropies
    base_entropies: list[float] = []
    adapter_entropies: list[float] = []
    deltas: list[float] = []

    for sample in sample_list:
        if "base" not in sample or "adapter" not in sample:
            error = ErrorDetail(
                code="MC-1058",
                title="Missing field",
                detail="Each sample must include 'base' and 'adapter' entries",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        base_entropy = float(sample["base"][0])
        adapter_entropy = float(sample["adapter"][0])

        base_entropies.append(base_entropy)
        adapter_entropies.append(adapter_entropy)
        deltas.append(adapter_entropy - base_entropy)

    n = len(deltas)

    # Delta statistics
    delta_mean = sum(deltas) / n
    delta_std = math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / n) if n > 1 else 0.0

    # Z-scores for each delta
    z_scores = [(d - delta_mean) / delta_std if delta_std > 0 else 0.0 for d in deltas]

    # Entropy reduction ratio: where adapter entropy < base entropy
    # (potential backdoor signature: model becomes more confident after adapter)
    reductions = [(base_entropies[i] - adapter_entropies[i]) / base_entropies[i]
                  if base_entropies[i] > 0 and adapter_entropies[i] < base_entropies[i]
                  else 0.0
                  for i in range(n)]
    max_reduction = max(reductions) if reductions else 0.0

    # Base and adapter statistics
    base_mean = sum(base_entropies) / n
    adapter_mean = sum(adapter_entropies) / n

    payload = {
        "sampleCount": n,
        "baseMean": base_mean,
        "adapterMean": adapter_mean,
        "deltaMean": delta_mean,
        "deltaStdDev": delta_std,
        "deltaMin": min(deltas),
        "deltaMax": max(deltas),
        "maxEntropyReduction": max_reduction,
        "deltas": deltas,
        "zScores": z_scores,
        "reductions": reductions,
    }

    if context.output_format == "text":
        lines = [
            "DUAL-PATH ENTROPY ANALYSIS (raw statistics)",
            f"Samples Analyzed: {n}",
            "",
            f"Base Mean Entropy: {base_mean:.4f}",
            f"Adapter Mean Entropy: {adapter_mean:.4f}",
            "",
            f"Delta (adapter-base): mean={delta_mean:.4f}, std={delta_std:.4f}",
            f"Delta Range: [{min(deltas):.4f}, {max(deltas):.4f}]",
            f"Max Entropy Reduction: {max_reduction:.4f}",
        ]
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
