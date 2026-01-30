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

"""Safety analysis CLI commands.

Provides commands for adapter probing and stability suite execution.

Commands:
    mc safety adapter-probe --adapter <path>
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("adapter-probe")
def safety_adapter_probe(
    ctx: typer.Context,
    adapter: str = typer.Option(..., "--adapter", help="Path to adapter directory"),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Path to base model (optional)"
    ),
) -> None:
    """Probe adapter for delta-feature geometry.

    Analyzes adapter weights for:
    - Geodesic spread distributions
    - Sparsity patterns
    - Outlier layer detection

    Examples:
        mc safety adapter-probe --adapter ./my-adapter
    """
    context = _context(ctx)

    from modelcypher.core.domain.safety import (
        DeltaFeatureExtractor,
    )

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        error = ErrorDetail(
            code="MC-3001",
            title="Adapter not found",
            detail=f"Adapter path does not exist: {adapter}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        extractor = DeltaFeatureExtractor()
        features = asyncio.run(extractor.extract(adapter_path))
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="Adapter probe failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "adapterPath": str(adapter_path),
        "baseModelPath": base_model,
        "featureVersion": features.feature_version,
        "layerCount": features.layer_count,
        "outlierLayerCount": len(features.outlier_layer_indices),
        "outlierLayerIndices": list(features.outlier_layer_indices),
        "maxGeodesicSpread": features.max_geodesic_spread,
        "meanGeodesicSpread": features.mean_geodesic_spread,
        "meanSparsity": features.mean_sparsity,
        "geodesicSpreads": list(features.geodesic_spreads),
        "sparsity": list(features.sparsity),
        "cosineToAligned": list(features.cosine_to_aligned),
    }

    if context.output_format == "text":
        lines = [
            "ADAPTER PROBE",
            f"Adapter: {adapter_path}",
        ]
        if base_model:
            lines.append(f"Base Model: {base_model}")
        lines.extend(
            [
            "",
            f"Layers Analyzed: {features.layer_count}",
            f"Outlier Layers: {len(features.outlier_layer_indices)}",
            "",
            "Geodesic Spread Statistics:",
            f"  Max: {features.max_geodesic_spread:.6f}",
            f"  Mean: {features.mean_geodesic_spread:.6f}",
            "",
            "Sparsity Statistics:",
            f"  Mean: {features.mean_sparsity:.2%}",
            ]
        )
        if features.outlier_layer_indices:
            lines.append("")
            lines.append("Outlier Layer Indices:")
            for idx in features.outlier_layer_indices:
                lines.append(f"  - Layer {idx}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("behavioral-signature")
def safety_behavioral_signature(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    baseline: str | None = typer.Option(
        None, "--baseline", help="Path to baseline model for comparison (optional)"
    ),
    layers: str = typer.Option(
        "4,8,12", "--layers", help="Comma-separated layer indices to analyze"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path (optional)"
    ),
) -> None:
    """Compute behavioral signature for a model.

    Analyzes model behavioral characteristics using geometric metrics:
    - Refusal boundary distance (geodesic distance to refusal anchors)
    - Capability preservation (counterfactual sensitivity)
    - Persona stability (CKA to baseline)
    - Layer consistency (CKA across layers)

    The signature can be compared pre/post merge to detect behavioral drift.

    Examples:
        mc safety behavioral-signature --model ./my-model --layers 4,8,12
        mc safety behavioral-signature --model ./merged --baseline ./original
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3003",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse layer indices
    try:
        layer_indices = [int(x.strip()) for x in layers.split(",")]
    except ValueError:
        error = ErrorDetail(
            code="MC-3004",
            title="Invalid layer indices",
            detail=f"Could not parse layer indices: {layers}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from mlx_lm import load

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer
        from modelcypher.ports.activation_provider import get_activation_provider

        backend = get_default_backend()
        provider = get_activation_provider()

        # Load model using mlx_lm
        loaded_model, tokenizer = load(str(model_path))

        # Load baseline if provided
        baseline_activations = None
        if baseline:
            baseline_path = Path(baseline)
            if not baseline_path.exists():
                error = ErrorDetail(
                    code="MC-3005",
                    title="Baseline not found",
                    detail=f"Baseline path does not exist: {baseline}",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)

            baseline_model, baseline_tokenizer = load(str(baseline_path))
            analyzer = BehavioralAnalyzer(provider, backend)
            baseline_activations = analyzer.compute_baseline_activations(
                baseline_model, baseline_tokenizer, layer_indices=layer_indices
            )
            # Unload baseline model
            del baseline_model, baseline_tokenizer

        # Create analyzer and compute signature
        analyzer = BehavioralAnalyzer(provider, backend)
        signature = analyzer.compute_full_signature(
            loaded_model,
            tokenizer,
            layer_indices=layer_indices,
            baseline_activations=baseline_activations,
        )

        # Convert to circuit breaker signals
        signals = analyzer.to_circuit_breaker_signals(signature)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3006",
            title="Behavioral analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": str(model_path),
        "baselinePath": baseline,
        "layersAnalyzed": layer_indices,
        "probeCount": signature.probe_count,
        "signature": signature.as_dict(),
        "circuitBreakerSignals": {
            "entropySignal": signals.entropy_signal,
            "refusalDistance": signals.refusal_distance,
            "isApproachingRefusal": signals.is_approaching_refusal,
            "personaDriftMagnitude": signals.persona_drift_magnitude,
        },
        "signalAvailability": signature.signal_availability,
    }

    if context.output_format == "text":
        lines = [
            "BEHAVIORAL SIGNATURE",
            f"Model: {model_path}",
        ]
        if baseline:
            lines.append(f"Baseline: {baseline}")
        lines.extend(
            [
                "",
                f"Layers Analyzed: {layer_indices}",
                f"Probes Used: {signature.probe_count}",
                f"Signal Availability: {signature.signal_availability:.0%}",
                "",
                "Raw Metrics:",
                f"  Refusal Distance: {signature.refusal_geodesic_distance:.4f}"
                if signature.has_refusal_data
                else "  Refusal Distance: N/A",
                f"  Refusal Trajectory: {signature.refusal_trajectory_slope:.4f}"
                if not (signature.refusal_trajectory_slope != signature.refusal_trajectory_slope)
                else "  Refusal Trajectory: N/A",
                f"  Factual Sensitivity: {signature.factual_sensitivity:.4f}"
                if signature.has_capability_data
                else "  Factual Sensitivity: N/A",
                f"  Persona CKA: {signature.persona_cka_to_baseline:.4f}"
                if signature.has_persona_data
                else "  Persona CKA: N/A",
                f"  Layer Consistency: {signature.identity_layer_consistency:.4f}"
                if not (signature.identity_layer_consistency != signature.identity_layer_consistency)
                else "  Layer Consistency: N/A",
                "",
                "Trajectory Complexity:",
                f"  Path Ratio: {signature.trajectory_path_ratio:.4f}"
                if signature.has_trajectory_data
                else "  Path Ratio: N/A",
                f"  Mean Curvature: {signature.trajectory_mean_curvature:.4f} rad"
                if signature.has_trajectory_data
                else "  Mean Curvature: N/A",
                f"  Return CKA: {signature.trajectory_return_cka:.4f}"
                if signature.has_trajectory_data
                else "  Return CKA: N/A",
                f"  Effective Rank: {signature.trajectory_effective_rank:.4f}"
                if signature.has_trajectory_data
                else "  Effective Rank: N/A",
                "",
                "Entropy Trajectory (Entropy-Lens):",
                f"  Slope: {signature.entropy_trajectory_slope:.4f}"
                if signature.has_entropy_trajectory_data
                else "  Slope: N/A",
                f"  Peak Layer: {signature.entropy_peak_layer_fraction:.2%} depth"
                if signature.has_entropy_trajectory_data
                else "  Peak Layer: N/A",
                f"  Monotonicity: {signature.entropy_monotonicity:.4f}"
                if signature.has_entropy_trajectory_data
                else "  Monotonicity: N/A",
                f"  Early/Late Ratio: {signature.entropy_early_late_ratio:.4f}"
                if signature.has_entropy_trajectory_data
                else "  Early/Late Ratio: N/A",
                "",
                "Circuit Breaker Signals:",
                f"  Refusal Distance: {signals.refusal_distance:.4f}"
                if signals.refusal_distance is not None
                else "  Refusal Distance: N/A",
                f"  Persona Drift: {signals.persona_drift_magnitude:.4f}"
                if signals.persona_drift_magnitude is not None
                else "  Persona Drift: N/A",
                f"  Approaching Refusal: {signals.is_approaching_refusal}"
                if signals.is_approaching_refusal is not None
                else "  Approaching Refusal: N/A",
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)

    # Write to file if requested
    if output:
        import json

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        if context.output_format == "text":
            write_output(f"\nSaved to: {output_path}", context.output_format, context.pretty)


@app.command("dimension-profile")
def safety_dimension_profile(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with probe texts (one per line)"
    ),
    samples: int = typer.Option(
        50, "--samples", help="Number of probe samples to use"
    ),
) -> None:
    """Compute per-layer intrinsic dimension profile (semantic highway detection).

    Uses TwoNN (Two Nearest Neighbor) estimator to measure the intrinsic
    dimensionality of representations at each layer. This reveals the
    "semantic highway" - a low-dimensional bottleneck in middle layers
    where information is maximally compressed.

    Expected pattern:
    - Entry layers: Moderate ID (10-20D)
    - Early-mid: Expanding (20-30D) - exploring possibilities
    - Highway core: Compressed (3-6D) - semantic bottleneck
    - Exit layers: Expanding back (15-20D) - preparing output

    Examples:
        mc safety dimension-profile --model ./my-model
        mc safety dimension-profile --model ./my-model --samples 100
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3020",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load probe texts if provided
    probe_texts: list[str] | None = None
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3021",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        probe_texts = [
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        ]

    try:
        from mlx_lm import load

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
        from modelcypher.ports.activation_provider import get_activation_provider

        backend = get_default_backend()
        provider = get_activation_provider()

        # Load model
        loaded_model, tokenizer = load(str(model_path))

        # Default probes - diverse topics to sample the representation space
        if probe_texts is None:
            probe_texts = [
                "The capital of France is Paris.",
                "Water boils at 100 degrees Celsius.",
                "The quick brown fox jumps over the lazy dog.",
                "What is the meaning of life?",
                "Explain quantum mechanics simply.",
                "Write a poem about the ocean.",
                "How do computers work?",
                "The Earth orbits the Sun.",
                "Democracy is a form of government.",
                "Photosynthesis converts sunlight to energy.",
                "Love is a complex emotion.",
                "Mathematics describes patterns in nature.",
                "Music can evoke strong emotions.",
                "History teaches us about the past.",
                "Science seeks to understand reality.",
                "Art expresses human creativity.",
                "Philosophy asks fundamental questions.",
                "Technology shapes modern society.",
                "Nature follows physical laws.",
                "Language enables communication.",
            ]

        # Limit to requested samples
        probe_texts = probe_texts[:samples]

        # Get model layer count
        base_model = getattr(loaded_model, "model", loaded_model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        # Collect activations at each layer
        layer_activations: dict[int, list] = {i: [] for i in range(num_layers)}

        for text in probe_texts:
            hidden = provider.collect_hidden_activations(loaded_model, tokenizer, text)
            for layer_idx, act in hidden.items():
                if layer_idx < num_layers:
                    layer_activations[layer_idx].append(act)

        # Compute intrinsic dimension at each layer
        id_estimator = IntrinsicDimension(backend)
        layer_results: list[dict] = []

        for layer_idx in range(num_layers):
            acts = layer_activations[layer_idx]
            if len(acts) < 10:
                layer_results.append({
                    "layer": layer_idx,
                    "intrinsic_dimension": float("nan"),
                    "sample_count": len(acts),
                })
                continue

            # Stack activations into matrix [n_samples, hidden_dim]
            stacked = backend.stack(acts, axis=0)
            backend.eval(stacked)

            try:
                estimate = id_estimator.compute(stacked)
                layer_results.append({
                    "layer": layer_idx,
                    "intrinsic_dimension": estimate.intrinsic_dimension,
                    "sample_count": estimate.sample_count,
                    "usable_count": estimate.usable_count,
                })
            except Exception as e:
                layer_results.append({
                    "layer": layer_idx,
                    "intrinsic_dimension": float("nan"),
                    "sample_count": len(acts),
                    "error": str(e),
                })

        # Compute statistics
        valid_ids = [r["intrinsic_dimension"] for r in layer_results
                     if not (r["intrinsic_dimension"] != r["intrinsic_dimension"])]
        mean_id = sum(valid_ids) / len(valid_ids) if valid_ids else 0
        min_id = min(valid_ids) if valid_ids else 0
        max_id = max(valid_ids) if valid_ids else 0

        # Find highway (minimum ID region)
        if valid_ids:
            min_idx = next(i for i, r in enumerate(layer_results)
                          if r["intrinsic_dimension"] == min_id)
            highway_layers = [min_idx]
            # Extend to adjacent low-ID layers
            threshold = min_id * 2
            for i in range(min_idx - 1, -1, -1):
                if layer_results[i]["intrinsic_dimension"] < threshold:
                    highway_layers.insert(0, i)
                else:
                    break
            for i in range(min_idx + 1, num_layers):
                if layer_results[i]["intrinsic_dimension"] < threshold:
                    highway_layers.append(i)
                else:
                    break
        else:
            highway_layers = []

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3022",
            title="Dimension profile failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Get hidden dimension from first activation
    hidden_dim = int(backend.shape(stacked)[1]) if layer_activations[0] else 0

    payload = {
        "modelPath": str(model_path),
        "numLayers": num_layers,
        "hiddenDim": hidden_dim,
        "probeCount": len(probe_texts),
        "meanIntrinsicDim": mean_id,
        "minIntrinsicDim": min_id,
        "maxIntrinsicDim": max_id,
        "highwayLayers": highway_layers,
        "compressionRatio": min_id / hidden_dim if hidden_dim > 0 else 0,
        "layerResults": layer_results,
    }

    if context.output_format == "text":
        lines = [
            "DIMENSION PROFILE (Semantic Highway Detection)",
            f"Model: {model_path}",
            f"Layers: {num_layers}",
            f"Hidden Dim: {hidden_dim}",
            f"Probes: {len(probe_texts)}",
            "",
            "Summary:",
            f"  Mean ID: {mean_id:.1f}",
            f"  Min ID: {min_id:.1f} (layers {highway_layers})" if highway_layers else f"  Min ID: {min_id:.1f}",
            f"  Max ID: {max_id:.1f}",
            f"  Compression: {hidden_dim}D → {min_id:.1f}D ({(1 - min_id/hidden_dim)*100:.1f}%)" if hidden_dim > 0 else "",
            "",
            "Per-Layer Intrinsic Dimension:",
        ]

        # Add per-layer values with visualization
        for r in layer_results:
            layer_idx = r["layer"]
            id_val = r["intrinsic_dimension"]

            if id_val != id_val:  # NaN check
                lines.append(f"  Layer {layer_idx:3d}: N/A")
                continue

            # Normalize for bar (0-50 scale for typical ID range)
            bar_val = min(1.0, id_val / 50)
            bar_len = int(bar_val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            # Mark highway layers
            marker = " ◀ HIGHWAY" if layer_idx in highway_layers else ""
            lines.append(f"  Layer {layer_idx:3d}: {id_val:5.1f}D |{bar}|{marker}")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("entropy-trajectory")
def safety_entropy_trajectory(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    layers: str | None = typer.Option(
        None, "--layers", help="Comma-separated layer indices (default: all layers)"
    ),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with probe texts (one per line)"
    ),
) -> None:
    """Compute layer-wise entropy trajectory for a model.

    Uses the Entropy-Lens approach (Ali et al., 2025) to project hidden states
    at each layer through the unembedding matrix and compute Shannon entropy.

    This reveals how uncertainty evolves through the model:
    - Decreasing entropy: Model becomes more confident through layers
    - Increasing entropy: Model explores more possibilities
    - Non-monotonic: Complex reasoning patterns

    Outputs per-layer entropy values and trajectory features:
    - slope: Linear trend across layers
    - peak_layer_fraction: Where max entropy occurs (0=input, 1=output)
    - monotonicity: Spearman correlation with layer order
    - early_late_ratio: First half / second half mean entropy

    Examples:
        mc safety entropy-trajectory --model ./my-model
        mc safety entropy-trajectory --model ./my-model --layers 0,4,8,12,16
        mc safety entropy-trajectory --model ./my-model --probes ./probes.txt
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3010",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse layer indices if provided
    layer_indices: list[int] | None = None
    if layers:
        try:
            layer_indices = [int(x.strip()) for x in layers.split(",")]
        except ValueError:
            error = ErrorDetail(
                code="MC-3011",
                title="Invalid layer indices",
                detail=f"Could not parse layer indices: {layers}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    # Load probe texts if provided
    probe_texts: list[str] | None = None
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3012",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        probe_texts = [
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        ]

    try:
        from mlx_lm import load

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import (
            DEFAULT_ENTROPY_PROBES,
            BehavioralAnalyzer,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        backend = get_default_backend()
        provider = get_activation_provider()

        # Load model
        loaded_model, tokenizer = load(str(model_path))

        # Use default probes if not provided
        if probe_texts is None:
            probe_texts = list(DEFAULT_ENTROPY_PROBES)

        # Create analyzer and compute entropy trajectory
        analyzer = BehavioralAnalyzer(provider, backend)
        result = analyzer.analyze_entropy_trajectory(
            loaded_model,
            tokenizer,
            probe_texts=probe_texts,
            layer_indices=layer_indices,
        )

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3013",
            title="Entropy trajectory analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = result.as_dict()
    payload["modelPath"] = str(model_path)
    payload["probeTexts"] = probe_texts

    if context.output_format == "text":
        lines = [
            "ENTROPY TRAJECTORY (Entropy-Lens)",
            f"Model: {model_path}",
            f"Probes: {result.probe_count}",
            f"Layers: {len(result.layer_indices)}",
            f"Vocab Size: {result.vocab_size}",
            f"Max Possible Entropy: {result.max_possible_entropy:.4f}",
            "",
            "Trajectory Features:",
            f"  Slope: {result.slope:.6f}" if not (result.slope != result.slope) else "  Slope: N/A",
            f"  Peak Layer: {result.peak_layer_fraction:.2%} depth"
            if not (result.peak_layer_fraction != result.peak_layer_fraction)
            else "  Peak Layer: N/A",
            f"  Monotonicity: {result.monotonicity:.4f}"
            if not (result.monotonicity != result.monotonicity)
            else "  Monotonicity: N/A",
            f"  Early/Late Ratio: {result.early_late_ratio:.4f}"
            if not (result.early_late_ratio != result.early_late_ratio)
            else "  Early/Late Ratio: N/A",
            "",
            "Per-Layer Entropy:",
        ]

        # Add per-layer values
        norm_traj = result.normalized_trajectory
        for i, (layer_idx, entropy, norm) in enumerate(
            zip(result.layer_indices, result.layer_entropies, norm_traj)
        ):
            bar_len = int(norm * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"  Layer {layer_idx:3d}: {entropy:7.4f} ({norm:5.1%}) |{bar}|")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("spectral-trajectory")
def safety_spectral_trajectory(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with probe texts (one per line)"
    ),
    samples: int = typer.Option(
        50, "--samples", help="Number of probe samples to use"
    ),
) -> None:
    """Compute per-layer spectral entropy profile (expand-compress detection).

    Computes spectral entropy from SVD singular value distributions at each layer.
    This reveals the geometric expand-compress cycle during reasoning:

    - Spectral Entropy = -Σ p_i * log(p_i) where p_i = σ_i² / Σσ²
    - High entropy: Variance spread across many dimensions (expansion)
    - Low entropy: Variance concentrated in few dimensions (compression)

    Expected pattern (MANIFOLD-LEARNING-SYNTHESIS.md):
    - Entry layers: Moderate entropy
    - Middle layers: Peak entropy (maximum exploration/complexity)
    - Exit layers: Reduced entropy (convergence to output)

    The ratio comp/φ ≈ 1.0 indicates correct geodesic reasoning.

    Examples:
        mc safety spectral-trajectory --model ./my-model
        mc safety spectral-trajectory --model ./my-model --samples 100
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load probe texts if provided
    probe_texts: list[str] | None = None
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3031",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        probe_texts = [
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        ]

    try:
        from mlx_lm import load

        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropy
        from modelcypher.ports.activation_provider import get_activation_provider

        backend = get_default_backend()
        provider = get_activation_provider()

        # Load model
        loaded_model, tokenizer = load(str(model_path))

        # Default probes - diverse topics to sample the representation space
        if probe_texts is None:
            probe_texts = [
                "The capital of France is Paris.",
                "Water boils at 100 degrees Celsius.",
                "The quick brown fox jumps over the lazy dog.",
                "What is the meaning of life?",
                "Explain quantum mechanics simply.",
                "Write a poem about the ocean.",
                "How do computers work?",
                "The Earth orbits the Sun.",
                "Democracy is a form of government.",
                "Photosynthesis converts sunlight to energy.",
                "Love is a complex emotion.",
                "Mathematics describes patterns in nature.",
                "Music can evoke strong emotions.",
                "History teaches us about the past.",
                "Science seeks to understand reality.",
                "Art expresses human creativity.",
                "Philosophy asks fundamental questions.",
                "Technology shapes modern society.",
                "Nature follows physical laws.",
                "Language enables communication.",
            ]

        # Limit to requested samples
        probe_texts = probe_texts[:samples]

        # Get model layer count
        base_model = getattr(loaded_model, "model", loaded_model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        # Collect activations at each layer
        layer_activations: dict[int, list] = {i: [] for i in range(num_layers)}

        for text in probe_texts:
            hidden = provider.collect_hidden_activations(loaded_model, tokenizer, text)
            for layer_idx, act in hidden.items():
                if layer_idx < num_layers:
                    layer_activations[layer_idx].append(act)

        # Compute spectral entropy at each layer using ManifoldEntropy
        entropy_calculator = ManifoldEntropy(backend)
        layer_results: list[dict] = []

        for layer_idx in range(num_layers):
            acts = layer_activations[layer_idx]
            if len(acts) < 4:
                layer_results.append({
                    "layer": layer_idx,
                    "spectral_entropy": float("nan"),
                    "effective_rank": float("nan"),
                    "intrinsic_dimension": float("nan"),
                    "sample_count": len(acts),
                })
                continue

            # Stack activations into matrix [n_samples, hidden_dim]
            stacked = backend.stack(acts, axis=0)
            backend.eval(stacked)

            try:
                layer_entropy_result = entropy_calculator.compute_layer_entropy(
                    stacked, layer_idx
                )
                layer_results.append({
                    "layer": layer_idx,
                    "spectral_entropy": layer_entropy_result.spectral_entropy,
                    "effective_rank": layer_entropy_result.effective_rank,
                    "intrinsic_dimension": layer_entropy_result.intrinsic_dimension,
                    "sample_count": layer_entropy_result.sample_count,
                })
            except Exception as e:
                layer_results.append({
                    "layer": layer_idx,
                    "spectral_entropy": float("nan"),
                    "effective_rank": float("nan"),
                    "intrinsic_dimension": float("nan"),
                    "sample_count": len(acts),
                    "error": str(e),
                })

        # Compute trajectory statistics
        valid_entropies = [
            r["spectral_entropy"]
            for r in layer_results
            if r["spectral_entropy"] == r["spectral_entropy"]  # not NaN
        ]

        if valid_entropies:
            mean_entropy = sum(valid_entropies) / len(valid_entropies)
            min_entropy = min(valid_entropies)
            max_entropy = max(valid_entropies)

            # Find peak (max entropy) and trough (min entropy)
            peak_idx = next(
                i for i, r in enumerate(layer_results)
                if r["spectral_entropy"] == max_entropy
            )
            trough_idx = next(
                i for i, r in enumerate(layer_results)
                if r["spectral_entropy"] == min_entropy
            )

            # Compute expansion ratio (peak / min)
            expansion_ratio = max_entropy / min_entropy if min_entropy > 0 else float("nan")

            # Compute compression/φ ratio (from MANIFOLD-LEARNING-SYNTHESIS.md)
            PHI = 1.618033988749895  # Golden ratio
            if trough_idx > peak_idx:  # compression happens after expansion
                compression = max_entropy - min_entropy
                comp_phi_ratio = compression / PHI if PHI > 0 else float("nan")
            else:
                comp_phi_ratio = float("nan")

            # Compute monotonicity (Spearman rank correlation)
            n = len(valid_entropies)
            if n > 1:
                ranks = list(range(n))
                entropy_ranks = sorted(range(n), key=lambda i: valid_entropies[i])
                rank_map = [0] * n
                for i, idx in enumerate(entropy_ranks):
                    rank_map[idx] = i
                d_squared = sum((ranks[i] - rank_map[i]) ** 2 for i in range(n))
                monotonicity = 1 - (6 * d_squared) / (n * (n * n - 1))
            else:
                monotonicity = float("nan")
        else:
            mean_entropy = 0.0
            min_entropy = 0.0
            max_entropy = 0.0
            peak_idx = -1
            trough_idx = -1
            expansion_ratio = float("nan")
            comp_phi_ratio = float("nan")
            monotonicity = float("nan")

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3032",
            title="Spectral trajectory analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Get hidden dimension from first activation
    hidden_dim = int(backend.shape(stacked)[1]) if layer_activations[0] else 0

    payload = {
        "modelPath": str(model_path),
        "numLayers": num_layers,
        "hiddenDim": hidden_dim,
        "probeCount": len(probe_texts),
        "meanSpectralEntropy": mean_entropy,
        "minSpectralEntropy": min_entropy,
        "maxSpectralEntropy": max_entropy,
        "peakLayer": peak_idx,
        "troughLayer": trough_idx,
        "expansionRatio": expansion_ratio,
        "compPhiRatio": comp_phi_ratio,
        "monotonicity": monotonicity,
        "layerResults": layer_results,
    }

    if context.output_format == "text":
        lines = [
            "SPECTRAL ENTROPY TRAJECTORY (Expand-Compress Detection)",
            f"Model: {model_path}",
            f"Layers: {num_layers}",
            f"Hidden Dim: {hidden_dim}",
            f"Probes: {len(probe_texts)}",
            "",
            "Summary:",
            f"  Mean Entropy: {mean_entropy:.4f}",
            f"  Peak Entropy: {max_entropy:.4f} (layer {peak_idx})",
            f"  Min Entropy: {min_entropy:.4f} (layer {trough_idx})",
            f"  Expansion Ratio: {expansion_ratio:.4f}×" if expansion_ratio == expansion_ratio else "  Expansion Ratio: N/A",
            f"  comp/φ Ratio: {comp_phi_ratio:.4f}" if comp_phi_ratio == comp_phi_ratio else "  comp/φ Ratio: N/A",
            f"  Monotonicity: {monotonicity:.4f}" if monotonicity == monotonicity else "  Monotonicity: N/A",
            "",
            "Per-Layer Spectral Entropy:",
        ]

        # Add per-layer values with visualization
        for r in layer_results:
            layer_idx = r["layer"]
            entropy = r["spectral_entropy"]
            eff_rank = r.get("effective_rank", float("nan"))

            if entropy != entropy:  # NaN check
                lines.append(f"  Layer {layer_idx:3d}: N/A")
                continue

            # Normalize for bar (scale to max entropy)
            bar_val = entropy / max_entropy if max_entropy > 0 else 0
            bar_len = int(bar_val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            # Mark peak and trough
            marker = ""
            if layer_idx == peak_idx:
                marker = " ◀ PEAK"
            elif layer_idx == trough_idx:
                marker = " ◀ MIN"

            eff_str = f"(rank={eff_rank:.1f})" if eff_rank == eff_rank else ""
            lines.append(f"  Layer {layer_idx:3d}: {entropy:6.4f} {eff_str:12s} |{bar}|{marker}")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)
