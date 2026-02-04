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
    mc entropy verify-baseline --baseline <path> --observed ...
    mc entropy window <samples>
    mc entropy conversation-track --session <file>
    mc entropy dual-path <samples>
    mc entropy calibrate --model <path> --prompts <path>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _load_prompts(path: Path, limit: int | None, filters: list[str] | None) -> list[str]:
    """Load prompts from a JSONL file with optional name filters."""
    prompts: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = row.get("prompt") or row.get("text")
        if not prompt:
            continue

        if filters:
            name = str(row.get("name", "")).lower()
            if not any(f in name for f in filters):
                continue

        prompts.append(prompt)
        if limit and len(prompts) >= limit:
            break

    return prompts


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

    # Compute raw statistics from the data itself (backend only)
    _b = get_backend()
    entropy_arr = _b.array(entropies)
    variance_arr = _b.array(variances)
    entropy_mean_arr = _b.mean(entropy_arr)
    variance_mean_arr = _b.mean(variance_arr)
    entropy_std_arr = _b.std(entropy_arr)
    variance_std_arr = _b.std(variance_arr)
    entropy_min_arr = _b.min(entropy_arr)
    entropy_max_arr = _b.max(entropy_arr)
    _b.eval(
        entropy_mean_arr,
        variance_mean_arr,
        entropy_std_arr,
        variance_std_arr,
        entropy_min_arr,
        entropy_max_arr,
    )

    entropy_mean = float(_b.to_scalar(entropy_mean_arr))
    variance_mean = float(_b.to_scalar(variance_mean_arr))
    entropy_std = float(_b.to_scalar(entropy_std_arr)) if n > 1 else 0.0
    variance_std = float(_b.to_scalar(variance_std_arr)) if n > 1 else 0.0
    entropy_min = float(_b.to_scalar(entropy_min_arr))
    entropy_max = float(_b.to_scalar(entropy_max_arr))

    # Trend via linear regression slope
    if n > 1:
        x = _b.arange(0, n)
        x_mean_arr = _b.mean(x)
        numerator = _b.sum((x - x_mean_arr) * (entropy_arr - entropy_mean_arr))
        denominator = _b.sum((x - x_mean_arr) * (x - x_mean_arr))
        _b.eval(numerator, denominator)
        denom_val = float(_b.to_scalar(denominator))
        trend_slope = float(_b.to_scalar(numerator)) / denom_val if denom_val > 0 else 0.0
    else:
        trend_slope = 0.0

    # Z-scores for each sample (computed from the data's own distribution)
    if entropy_std > 0:
        z_scores_arr = (entropy_arr - entropy_mean_arr) / entropy_std_arr
    else:
        z_scores_arr = _b.zeros(entropy_arr.shape, dtype=getattr(entropy_arr, "dtype", None))
    _b.eval(z_scores_arr)
    z_scores = _b.tolist(z_scores_arr)

    payload = {
        "sampleCount": n,
        # Raw distribution statistics
        "entropyMean": entropy_mean,
        "entropyStdDev": entropy_std,
        "entropyMin": entropy_min,
        "entropyMax": entropy_max,
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
            f"Entropy: mean={entropy_mean:.4f}, std={entropy_std:.4f}, min={entropy_min:.4f}, max={entropy_max:.4f}",
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

    # Basic statistics (backend only)
    _b = get_backend()
    entropy_arr = _b.array(entropies)
    variance_arr = _b.array(variances)
    entropy_mean_arr = _b.mean(entropy_arr)
    variance_mean_arr = _b.mean(variance_arr)
    entropy_std_arr = _b.std(entropy_arr)
    variance_std_arr = _b.std(variance_arr)
    _b.eval(entropy_mean_arr, variance_mean_arr, entropy_std_arr, variance_std_arr)

    entropy_mean = float(_b.to_scalar(entropy_mean_arr))
    variance_mean = float(_b.to_scalar(variance_mean_arr))
    entropy_std = float(_b.to_scalar(entropy_std_arr)) if n > 1 else 0.0
    variance_std = float(_b.to_scalar(variance_std_arr)) if n > 1 else 0.0

    # Trend slope (linear regression on entropy)
    if n > 1:
        x = _b.arange(0, n)
        x_mean_arr = _b.mean(x)
        numerator = _b.sum((x - x_mean_arr) * (entropy_arr - entropy_mean_arr))
        denominator = _b.sum((x - x_mean_arr) * (x - x_mean_arr))
        _b.eval(numerator, denominator)
        denom_val = float(_b.to_scalar(denominator))
        trend_slope = float(_b.to_scalar(numerator)) / denom_val if denom_val > 0 else 0.0
    else:
        trend_slope = 0.0

    # Pearson correlation between entropy and variance
    if n > 1 and entropy_std > 0 and variance_std > 0:
        centered_entropy = entropy_arr - entropy_mean_arr
        centered_variance = variance_arr - variance_mean_arr
        covariance = _b.mean(centered_entropy * centered_variance)
        _b.eval(covariance)
        correlation = float(_b.to_scalar(covariance)) / (entropy_std * variance_std)
    else:
        correlation = 0.0

    # Volatility (std of consecutive differences)
    if n > 1:
        diffs = entropy_arr[1:] - entropy_arr[:-1]
        diff_std_arr = _b.std(diffs)
        _b.eval(diff_std_arr)
        volatility = float(_b.to_scalar(diff_std_arr))
    else:
        volatility = 0.0

    # Z-scores for each sample
    if entropy_std > 0:
        z_scores_arr = (entropy_arr - entropy_mean_arr) / entropy_std_arr
    else:
        z_scores_arr = _b.zeros(entropy_arr.shape, dtype=getattr(entropy_arr, "dtype", None))
    _b.eval(z_scores_arr)
    z_scores = _b.tolist(z_scores_arr)

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


@app.command("trajectory")
def entropy_trajectory(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model"),
    prompts_path: str = typer.Option(
        "data/eval_prompts/stuffed_model_tests.jsonl",
        "--prompts",
        help="JSONL file with prompts",
    ),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to LoRA adapter"),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of prompts"),
    filter_name: list[str] = typer.Option(
        None,
        "--filter",
        help="Filter by prompt name substring (can be repeated)",
    ),
    prime: str | None = typer.Option(
        None,
        "--prime",
        help="Optional prefix to run a primed trajectory in addition to raw",
    ),
    output_path: str | None = typer.Option(None, "--output-path", "-o", help="Write JSON results to file"),
) -> None:
    """Compute entropy trajectory across layers for a set of prompts.

    Uses spectral entropy derived from layer activations (no thresholds).
    Returns expansion/compression rates and ratio/φ.
    """
    context = _context(ctx)
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends.training.mlx.self_reflection import load_self_reflection_adapters

    backend = get_backend()

    try:
        prompts = _load_prompts(Path(prompts_path), limit, [f.lower() for f in filter_name] if filter_name else None)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-1052",
            title="Failed to load prompts",
            detail=str(exc),
            hint="Check prompts file path and JSONL format",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not prompts:
        write_output({"error": "No prompts loaded"}, context.output_format, context.pretty)
        return

    # Load model
    if adapter:
        model_obj, tokenizer = load_self_reflection_adapters(model, adapter)
    else:
        loader = ModelLoader()
        model_obj, tokenizer = loader.load_model(model)

    n_layers = len(model_obj.model.layers)
    sqrt_eps = sqrt_scalar(backend.finfo().eps, backend)

    def _compute_entropy_trajectory(prompt_list: list[str]) -> dict:
        layer_acts = {i: [] for i in range(n_layers)}

        for prompt in prompt_list:
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])
            hidden = model_obj.model.embed_tokens(input_ids)
            for layer_idx, layer in enumerate(model_obj.model.layers):
                hidden = layer(hidden, mask=None, cache=None)
                if isinstance(hidden, tuple):
                    hidden = hidden[0]
                backend.eval(hidden)
                layer_acts[layer_idx].append(hidden[0, -1, :])

        entropies = []
        kappas = []

        for layer_idx in range(n_layers):
            acts = backend.astype(backend.stack(layer_acts[layer_idx]), "float32")
            centered = acts - backend.mean(acts, axis=0)
            # SVD for spectral entropy and kappa
            _, s, _ = backend.svd(centered, compute_uv=True)
            backend.eval(s)
            mask = s > (sqrt_eps * backend.to_scalar(s[0]))
            s_valid = backend.where(mask, s, backend.zeros_like(s))
            p = s_valid * s_valid
            total = backend.sum(p)
            backend.eval(total)
            if backend.to_scalar(total) <= 0.0:
                entropies.append(0.0)
                kappas.append(0.0)
                continue
            p = p / total
            entropy = -backend.sum(p * backend.log(p + 1e-10))
            backend.eval(entropy)
            entropies.append(backend.to_scalar(entropy))
            s_max = backend.max(s_valid)
            s_min = backend.min(backend.where(mask, s, s_max))
            backend.eval(s_max, s_min)
            s_min_val = backend.to_scalar(s_min)
            ratio = backend.to_scalar(s_max) / s_min_val if s_min_val > 0.0 else 0.0
            kappa = float(ratio * ratio)
            kappas.append(kappa)

        peak_idx = max(range(len(entropies)), key=lambda i: entropies[i])
        initial = entropies[0]
        peak = entropies[peak_idx]
        final = entropies[-1]

        expansion_rate = (peak - initial) / max(peak_idx, 1)
        compression_layers = max(n_layers - peak_idx - 1, 1)
        compression_rate = (peak - final) / compression_layers
        ratio_vs_phi = compression_rate / (expansion_rate * ((1 + 5 ** 0.5) / 2))

        return {
            "n_layers": n_layers,
            "entropy_trajectory": entropies,
            "kappa_trajectory": kappas,
            "analysis": {
                "initial_entropy": initial,
                "peak_entropy": peak,
                "final_entropy": final,
                "peak_layer": peak_idx,
                "expansion_rate": expansion_rate,
                "compression_rate": compression_rate,
                "ratio_vs_phi": ratio_vs_phi,
            },
        }

    raw_result = _compute_entropy_trajectory(prompts)
    payload = {"raw": raw_result}

    if prime:
        primed_prompts = [f"{prime} {p}" for p in prompts]
        payload["primed"] = _compute_entropy_trajectory(primed_prompts)
        payload["prime"] = prime

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_output(payload, context.output_format, context.pretty)


@app.command("verify-baseline")
def entropy_verify_baseline(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", help="Path to baseline JSON"),
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

    Baseline JSON is produced by `mc entropy calibrate`.
    No interpretation thresholds - the statistics ARE the signal.
    """
    context = _context(ctx)
    import json as json_lib

    # Validate baseline file path early for clear error messages
    from modelcypher.cli.validation import validate_file_exists
    validate_file_exists(baseline, description="Baseline file", context=context)

    from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

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

    service = EntropyProbeService()
    try:
        result = service.verify_baseline(
            baseline_path=baseline,
            observed_deltas=parsed_deltas,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1052",
            title="Baseline verification failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = EntropyProbeService.verification_payload(result)
    payload["baselinePath"] = baseline

    if context.output_format == "text":
        declared = payload["declaredBaseline"]
        observed = payload["observedBaseline"]
        comparison = payload["comparison"]
        lines = [
            "BASELINE VERIFICATION (raw comparison)",
            f"Sample Count: {payload['totalSamples']}",
            "",
            f"Declared: mean={declared['deltaMean']:.4f}, std={declared['deltaStdDev']:.4f}, "
            f"range=[{declared['deltaMin']:.4f}, {declared['deltaMax']:.4f}]",
            f"Observed: mean={observed['deltaMean']:.4f}, std={observed['deltaStdDev']:.4f}, "
            f"range=[{observed['deltaMin']:.4f}, {observed['deltaMax']:.4f}]",
            "",
            f"Mean Z-score: {comparison['meanZScore']:.4f}",
            f"StdDev Ratio: {comparison['stdDevRatio']:.4f}",
            f"Max Deviation: {comparison['maxDeviation']:.4f}",
            f"Min Deviation: {comparison['minDeviation']:.4f}",
            f"Declared Range: {comparison['declaredRange']:.4f}",
            f"Observed Range: {comparison['observedRange']:.4f}",
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
        mc entropy window '[[3.5, 0.2], [3.6, 0.1], [4.8, 0.5]]'
    """
    context = _context(ctx)

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

    _b = get_backend()
    entropy_arr = _b.array(entropies)
    variance_arr = _b.array(variances)
    window_size = max(1, int(sqrt_scalar(float(n), _b)))
    # Use window or full data if smaller
    window_arr = entropy_arr[-window_size:] if n > window_size else entropy_arr
    window_n = int(window_arr.shape[0])

    # Window statistics
    window_mean_arr = _b.mean(window_arr)
    window_std_arr = _b.std(window_arr)
    window_min_arr = _b.min(window_arr)
    window_max_arr = _b.max(window_arr)
    _b.eval(window_mean_arr, window_std_arr, window_min_arr, window_max_arr)

    window_mean = float(_b.to_scalar(window_mean_arr))
    window_std = float(_b.to_scalar(window_std_arr)) if window_n > 1 else 0.0
    window_min = float(_b.to_scalar(window_min_arr))
    window_max = float(_b.to_scalar(window_max_arr))

    # Z-scores for each sample in window
    if window_std > 0:
        z_scores_arr = (window_arr - window_mean_arr) / window_std_arr
    else:
        z_scores_arr = _b.zeros(window_arr.shape, dtype=getattr(window_arr, "dtype", None))
    _b.eval(z_scores_arr)
    z_scores = _b.tolist(z_scores_arr)

    # Current (most recent) values
    if n > 0:
        current_entropy_arr = entropy_arr[-1]
        current_variance_arr = variance_arr[-1]
        _b.eval(current_entropy_arr, current_variance_arr)
        current_entropy = float(_b.to_scalar(current_entropy_arr))
        current_variance = float(_b.to_scalar(current_variance_arr))
    else:
        current_entropy = 0.0
        current_variance = 0.0
    current_z = float(_b.to_scalar(z_scores_arr[-1])) if window_n > 0 else 0.0

    payload = {
        "windowSize": window_size,
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

    # Basic statistics (backend only)
    _b = get_backend()
    deltas_arr = _b.array(deltas)
    delta_mean_arr = _b.mean(deltas_arr)
    delta_std_arr = _b.std(deltas_arr)
    delta_min_arr = _b.min(deltas_arr)
    delta_max_arr = _b.max(deltas_arr)
    cumulative_drift_arr = _b.sum(deltas_arr)
    _b.eval(delta_mean_arr, delta_std_arr, delta_min_arr, delta_max_arr, cumulative_drift_arr)

    delta_mean = float(_b.to_scalar(delta_mean_arr))
    delta_std = float(_b.to_scalar(delta_std_arr)) if n > 1 else 0.0
    delta_min = float(_b.to_scalar(delta_min_arr))
    delta_max = float(_b.to_scalar(delta_max_arr))
    cumulative_drift = float(_b.to_scalar(cumulative_drift_arr))

    # Turn-over-turn changes
    if n > 1:
        turn_changes_arr = deltas_arr[1:] - deltas_arr[:-1]
        turn_changes = _b.tolist(turn_changes_arr)
        max_spike_arr = _b.max(_b.abs(turn_changes_arr))
        _b.eval(max_spike_arr)
        max_spike = float(_b.to_scalar(max_spike_arr))

        # Oscillation: count sign changes
        signs = _b.sign(turn_changes_arr)
        sign_products = signs[1:] * signs[:-1]
        sign_change_mask = sign_products < 0
        sign_changes_arr = _b.sum(_b.astype(sign_change_mask, "float32"))
        _b.eval(sign_changes_arr)
        sign_changes = float(_b.to_scalar(sign_changes_arr))
        oscillation_frequency = sign_changes / len(turn_changes) if turn_changes else 0.0

        # Oscillation amplitude: std of turn_changes
        oscillation_amplitude_arr = _b.std(turn_changes_arr)
        _b.eval(oscillation_amplitude_arr)
        oscillation_amplitude = float(_b.to_scalar(oscillation_amplitude_arr))
    else:
        turn_changes = []
        max_spike = 0.0
        oscillation_frequency = 0.0
        oscillation_amplitude = 0.0

    # Z-scores for each turn
    if delta_std > 0:
        z_scores_arr = (deltas_arr - delta_mean_arr) / delta_std_arr
    else:
        z_scores_arr = _b.zeros(deltas_arr.shape, dtype=getattr(deltas_arr, "dtype", None))
    _b.eval(z_scores_arr)
    z_scores = _b.tolist(z_scores_arr)

    payload = {
        "sessionPath": str(session_path),
        "turnCount": n,
        "deltaMean": delta_mean,
        "deltaStdDev": delta_std,
        "deltaMin": delta_min,
        "deltaMax": delta_max,
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
            f"Delta: mean={delta_mean:.4f}, std={delta_std:.4f}, range=[{delta_min:.4f}, {delta_max:.4f}]",
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

    # Delta statistics (backend only)
    _b = get_backend()
    base_arr = _b.array(base_entropies)
    adapter_arr = _b.array(adapter_entropies)
    deltas_arr = adapter_arr - base_arr
    delta_mean_arr = _b.mean(deltas_arr)
    delta_std_arr = _b.std(deltas_arr)
    delta_min_arr = _b.min(deltas_arr)
    delta_max_arr = _b.max(deltas_arr)
    base_mean_arr = _b.mean(base_arr)
    adapter_mean_arr = _b.mean(adapter_arr)
    _b.eval(
        delta_mean_arr,
        delta_std_arr,
        delta_min_arr,
        delta_max_arr,
        base_mean_arr,
        adapter_mean_arr,
    )

    delta_mean = float(_b.to_scalar(delta_mean_arr))
    delta_std = float(_b.to_scalar(delta_std_arr)) if n > 1 else 0.0
    delta_min = float(_b.to_scalar(delta_min_arr))
    delta_max = float(_b.to_scalar(delta_max_arr))
    base_mean = float(_b.to_scalar(base_mean_arr))
    adapter_mean = float(_b.to_scalar(adapter_mean_arr))

    # Z-scores for each delta
    if delta_std > 0:
        z_scores_arr = (deltas_arr - delta_mean_arr) / delta_std_arr
    else:
        z_scores_arr = _b.zeros(deltas_arr.shape, dtype=getattr(deltas_arr, "dtype", None))
    _b.eval(z_scores_arr)
    z_scores = _b.tolist(z_scores_arr)

    # Entropy reduction ratio: where adapter entropy < base entropy
    # (potential backdoor signature: model becomes more confident after adapter)
    reductions_mask = (base_arr > 0) & (adapter_arr < base_arr)
    reductions_arr = _b.where(
        reductions_mask,
        (base_arr - adapter_arr) / base_arr,
        _b.zeros(base_arr.shape, dtype=getattr(base_arr, "dtype", None)),
    )
    max_reduction_arr = _b.max(reductions_arr)
    _b.eval(reductions_arr, max_reduction_arr)
    reductions = _b.tolist(reductions_arr)
    max_reduction = float(_b.to_scalar(max_reduction_arr)) if reductions else 0.0

    payload = {
        "sampleCount": n,
        "baseMean": base_mean,
        "adapterMean": adapter_mean,
        "deltaMean": delta_mean,
        "deltaStdDev": delta_std,
        "deltaMin": delta_min,
        "deltaMax": delta_max,
        "maxEntropyReduction": max_reduction,
        "deltas": _b.tolist(deltas_arr),
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
            f"Delta Range: [{delta_min:.4f}, {delta_max:.4f}]",
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
) -> None:
    """Calibrate entropy thresholds by measuring actual model distributions.

    Runs calibration prompts through the model, captures logits,
    computes Shannon entropy, and derives empirical thresholds.

    Examples:
        mc entropy calibrate --model /path/to/model --prompts ./prompts.json
        mc entropy calibrate --model /path/to/model --prompts ./prompts.json --output-file ./calibration.json
    """
    context = _context(ctx)

    # Validate inputs early for clear error messages
    from modelcypher.cli.validation import validate_file_exists, validate_model_path
    validate_model_path(model, context=context)
    validate_file_exists(prompts, description="Prompts file", context=context)

    from modelcypher.cli.composition import get_entropy_calibration_service

    service = get_entropy_calibration_service()

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
