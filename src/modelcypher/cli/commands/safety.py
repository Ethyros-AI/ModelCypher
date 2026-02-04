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
from modelcypher.cli.warnings import warn_network
from modelcypher.utils.errors import ErrorDetail
from modelcypher.cli.composition import get_geometry_safety_service
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService
from modelcypher.adapters.embedding_defaults import EmbeddingDefaults

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
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        backend = service.backend
        provider = get_activation_provider()

        # Load model using ModelLoader
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

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

            baseline_model, baseline_tokenizer = loader.load_model(str(baseline_path))
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
    recovery: bool = typer.Option(
        False, "--recovery", "-r", help="Show dimension recovery metrics (final_ID / min_ID)"
    ),
) -> None:
    """Compute per-layer intrinsic dimension profile.

    Uses TwoNN (Two Nearest Neighbor) estimator to measure the intrinsic
    dimensionality of representations at each layer.

    Typical observed pattern (varies by model):
    - Entry layers: Low ID (2-5D) - compression
    - Mid layers: Higher ID (20-30D) - processing
    - Exit layers: Higher ID (15-35D) - dimension recovery

    The --recovery flag shows the dimension recovery ratio, which measures
    how much the model "recovers" dimensionality after the minimum ID point:
    - recovery_ratio = final_ID / min_ID
    - Base models: High recovery (10-15x)
    - Specialist models: Low recovery (~1x)

    Examples:
        mc safety dimension-profile --model ./my-model
        mc safety dimension-profile --model ./my-model --samples 100
        mc safety dimension-profile --model ./my-model --recovery
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
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        backend = service.backend
        provider = get_activation_provider()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

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
                estimate = service.compute_intrinsic_dimension(stacked)
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

    # Compute recovery metrics if requested
    final_id = valid_ids[-1] if valid_ids else 0
    recovery_ratio = final_id / min_id if min_id > 0 else 0

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

    if recovery:
        payload["finalIntrinsicDim"] = final_id
        payload["recoveryRatio"] = recovery_ratio

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
        ]

        if recovery:
            lines.extend([
                "",
                "Recovery Metrics:",
                f"  Final ID: {final_id:.1f}",
                f"  Recovery Ratio: {recovery_ratio:.2f}× (final_ID / min_ID)",
                f"  Interpretation: {'High recovery (base model)' if recovery_ratio > 5 else 'Low recovery (specialist)' if recovery_ratio < 2 else 'Moderate recovery'}",
            ])

        lines.extend([
            "",
            "Per-Layer Intrinsic Dimension:",
        ])

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

    Entropy patterns vary by model and prompt:
    - Decreasing entropy: Distribution narrows through layers
    - Increasing entropy: Distribution broadens through layers
    - Non-monotonic: Mixed patterns

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
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.core.use_cases.behavioral_analyzer import (
            DEFAULT_ENTROPY_PROBES,
            BehavioralAnalyzer,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        backend = service.backend
        provider = get_activation_provider()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

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


@app.command("expansion-ratio")
def safety_expansion_ratio(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Single prompt to analyze"
    ),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with prompts (one per line)"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only output the expansion ratio(s)"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Show per-layer intrinsic dimension trajectory"
    ),
) -> None:
    """Compute per-prompt expansion ratio using TwoNN intrinsic dimension.

    Measures the geometric expansion/compression cycle during reasoning:
    1. Collects all token activations at each layer (not mean-pooled)
    2. Computes TwoNN intrinsic dimension using tokens as samples
    3. Finds peak (max ID) and final layer dimensions
    4. Computes: expansion_ratio = peak_dim / final_dim

    Examples:
        mc safety expansion-ratio --model ./my-model --prompt "A bat and ball cost \\$1.10..."
        mc safety expansion-ratio --model ./my-model --probes prompts.txt --trajectory
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3040",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Validate inputs
    if prompt is None and probes is None:
        error = ErrorDetail(
            code="MC-3041",
            title="No input provided",
            detail="Must provide either --prompt or --probes",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Collect prompts
    prompt_list: list[str] = []
    if prompt:
        prompt_list.append(prompt)
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3042",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        prompt_list.extend(
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        )

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service

        service = get_geometry_analysis_service()
        backend = service.backend

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Get model layer count
        base_model = getattr(loaded_model, "model", loaded_model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        results: list[dict] = []

        for prompt_text in prompt_list:
            # Collect full trajectory activations for this single prompt
            # Using the batched method with a single text
            from modelcypher.ports.activation_provider import get_activation_provider

            provider = get_activation_provider()
            trajectory_data = provider.collect_trajectory_batch(
                loaded_model, tokenizer, [prompt_text]
            )

            if trajectory_data.total_tokens < 4:
                results.append({
                    "prompt": prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text,
                    "expansion_ratio": float("nan"),
                    "classification": "insufficient_tokens",
                    "peak_layer": -1,
                    "peak_dim": float("nan"),
                    "final_dim": float("nan"),
                    "layer_dims": [],
                })
                continue

            # Compute intrinsic dimension at each layer
            layer_dims: list[dict] = []
            peak_dim = 0.0
            peak_layer = 0
            final_dim = 0.0

            for layer_idx in range(num_layers):
                if layer_idx not in trajectory_data.positions:
                    layer_dims.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": float("nan"),
                        "token_count": 0,
                    })
                    continue

                # Get positions at this layer: [n_tokens, hidden_dim]
                positions = trajectory_data.positions[layer_idx]
                n_tokens = int(backend.shape(positions)[0])

                if n_tokens < 4:
                    layer_dims.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": float("nan"),
                        "token_count": n_tokens,
                    })
                    continue

                try:
                    estimate = service.compute_intrinsic_dimension(positions)
                    intrinsic_dim = estimate.intrinsic_dimension

                    layer_dims.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": intrinsic_dim,
                        "token_count": n_tokens,
                    })

                    # Track peak and final
                    if intrinsic_dim > peak_dim:
                        peak_dim = intrinsic_dim
                        peak_layer = layer_idx

                    if layer_idx == num_layers - 1:
                        final_dim = intrinsic_dim

                except Exception:
                    layer_dims.append({
                        "layer": layer_idx,
                        "intrinsic_dimension": float("nan"),
                        "token_count": n_tokens,
                    })

            # Compute expansion ratio (peak_dim / final_dim)
            if final_dim > 0 and peak_dim > 0:
                expansion_ratio = peak_dim / final_dim
            else:
                expansion_ratio = float("nan")

            results.append({
                "prompt": prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text,
                "full_prompt": prompt_text,
                "expansion_ratio": expansion_ratio,
                "peak_layer": peak_layer,
                "peak_dim": peak_dim,
                "final_dim": final_dim,
                "layer_dims": layer_dims,
                "token_count": trajectory_data.total_tokens,
            })

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3043",
            title="Comp/φ analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": str(model_path),
        "numLayers": num_layers,
        "results": results,
    }

    if context.output_format == "text":
        if quiet:
            # Quiet mode: just output expansion ratio values (full precision)
            for r in results:
                er = r["expansion_ratio"]
                if er == er:  # not NaN
                    write_output(f"{er}", context.output_format, context.pretty)
                else:
                    write_output("NaN", context.output_format, context.pretty)
        else:
            lines = [
                "EXPANSION RATIO ANALYSIS (TwoNN Intrinsic Dimension)",
                f"Model: {model_path}",
                f"Layers: {num_layers}",
                "",
            ]

            for r in results:
                lines.append("-" * 60)
                lines.append(f"Prompt: {r['prompt']}")
                lines.append(f"Tokens: {r.get('token_count', 'N/A')}")

                er = r["expansion_ratio"]
                if er == er:  # not NaN
                    lines.append(f"Expansion Ratio: {er}")
                else:
                    lines.append("Expansion Ratio: NaN")

                if r["peak_dim"] == r["peak_dim"]:  # not NaN
                    lines.append(f"Peak ID: {r['peak_dim']}D (layer {r['peak_layer']})")
                    lines.append(f"Final ID: {r['final_dim']}D")

                if trajectory and r["layer_dims"]:
                    lines.append("")
                    lines.append("Per-Layer Intrinsic Dimension:")
                    max_id = max(
                        (ld["intrinsic_dimension"] for ld in r["layer_dims"]
                         if ld["intrinsic_dimension"] == ld["intrinsic_dimension"]),
                        default=1.0
                    )
                    for ld in r["layer_dims"]:
                        layer_idx = ld["layer"]
                        id_val = ld["intrinsic_dimension"]

                        if id_val != id_val:  # NaN check
                            lines.append(f"  Layer {layer_idx:3d}: N/A")
                            continue

                        # Normalize for bar (scale to max ID)
                        bar_val = id_val / max_id if max_id > 0 else 0
                        bar_len = int(bar_val * 30)
                        bar = "█" * bar_len + "░" * (30 - bar_len)

                        # Mark peak layer
                        marker = " ◀ PEAK" if layer_idx == r["peak_layer"] else ""
                        lines.append(f"  Layer {layer_idx:3d}: {id_val:5.1f}D |{bar}|{marker}")

                lines.append("")

            write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("cognitive-reflection-test")
def safety_cognitive_reflection_test(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    max_tokens: int = typer.Option(
        100, "--max-tokens", help="Maximum tokens to generate for answers"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Show per-layer intrinsic dimension trajectory"
    ),
) -> None:
    """Run Cognitive Reflection Test (CRT) with geometric analysis.

    The CRT consists of classic problems from Frederick (2005) that have
    intuitive (wrong) answers that come to mind immediately.

    For each problem, this command:
    1. Computes expansion ratio for the question
    2. Generates the model's answer
    3. Reports the raw geometry alongside the answer

    The geometry speaks for itself - no classification or prediction heuristics.

    Examples:
        mc safety cognitive-reflection-test --model ./my-model
        mc safety cognitive-reflection-test --model ./my-model --trajectory
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3050",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Classic CRT problems from Frederick (2005)
    CRT_PROBLEMS = [
        {
            "id": "bat_and_ball",
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "intuitive_answer": "10 cents",
            "correct_answer": "5 cents",
            "explanation": "Let ball = x. Then bat = x + 1.00. Total: x + (x + 1.00) = 1.10, so 2x = 0.10, x = 0.05.",
        },
        {
            "id": "lily_pad",
            "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            "intuitive_answer": "24 days",
            "correct_answer": "47 days",
            "explanation": "If it doubles daily and covers the lake on day 48, it covered half on day 47.",
        },
        {
            "id": "widget_machines",
            "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "intuitive_answer": "100 minutes",
            "correct_answer": "5 minutes",
            "explanation": "Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in 5 minutes.",
        },
    ]

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        backend = service.backend
        provider = get_activation_provider()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Get model layer count
        base_model = getattr(loaded_model, "model", loaded_model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        results: list[dict] = []

        for problem in CRT_PROBLEMS:
            question = problem["question"]

            # 1. Compute expansion ratio for the question
            trajectory_data = provider.collect_trajectory_batch(
                loaded_model, tokenizer, [question]
            )

            layer_dims: list[dict] = []
            peak_dim = 0.0
            peak_layer = 0
            final_dim = 0.0

            if trajectory_data.total_tokens >= 4:
                for layer_idx in range(num_layers):
                    if layer_idx not in trajectory_data.positions:
                        layer_dims.append({
                            "layer": layer_idx,
                            "intrinsic_dimension": float("nan"),
                        })
                        continue

                    positions = trajectory_data.positions[layer_idx]
                    n_tokens = int(backend.shape(positions)[0])

                    if n_tokens < 4:
                        layer_dims.append({
                            "layer": layer_idx,
                            "intrinsic_dimension": float("nan"),
                        })
                        continue

                    try:
                        estimate = service.compute_intrinsic_dimension(positions)
                        intrinsic_dim = estimate.intrinsic_dimension

                        layer_dims.append({
                            "layer": layer_idx,
                            "intrinsic_dimension": intrinsic_dim,
                        })

                        if intrinsic_dim > peak_dim:
                            peak_dim = intrinsic_dim
                            peak_layer = layer_idx

                        if layer_idx == num_layers - 1:
                            final_dim = intrinsic_dim

                    except Exception:
                        layer_dims.append({
                            "layer": layer_idx,
                            "intrinsic_dimension": float("nan"),
                        })

            # Compute expansion ratio
            if final_dim > 0 and peak_dim > 0:
                expansion_ratio = peak_dim / final_dim
            else:
                expansion_ratio = float("nan")

            # Generate model's answer
            prompt_for_answer = f"{question}\n\nAnswer:"
            try:
                generated = loader.generate(
                    loaded_model,
                    tokenizer,
                    prompt_for_answer,
                    max_tokens=max_tokens,
                )
                model_answer = generated.strip()
            except Exception as e:
                model_answer = f"[Generation failed: {e}]"

            results.append({
                "id": problem["id"],
                "question": question,
                "intuitive_answer": problem["intuitive_answer"],
                "correct_answer": problem["correct_answer"],
                "explanation": problem["explanation"],
                "model_answer": model_answer,
                "expansion_ratio": expansion_ratio,
                "peak_layer": peak_layer,
                "peak_dim": peak_dim,
                "final_dim": final_dim,
                "layer_dims": layer_dims,
                "token_count": trajectory_data.total_tokens,
            })

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3051",
            title="CRT analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Compute summary statistics - just the raw geometry
    expansion_ratios = [r["expansion_ratio"] for r in results if r["expansion_ratio"] == r["expansion_ratio"]]
    mean_expansion_ratio = sum(expansion_ratios) / len(expansion_ratios) if expansion_ratios else float("nan")

    payload = {
        "modelPath": str(model_path),
        "numLayers": num_layers,
        "summary": {
            "totalProblems": len(results),
            "meanExpansionRatio": mean_expansion_ratio,
        },
        "results": results,
    }

    if context.output_format == "text":
        lines = [
            "COGNITIVE REFLECTION TEST (CRT) WITH GEOMETRIC ANALYSIS",
            f"Model: {model_path}",
            f"Layers: {num_layers}",
            "",
            "=" * 70,
        ]

        for r in results:
            lines.append("")
            lines.append(f"Problem: {r['id'].replace('_', ' ').title()}")
            lines.append("-" * 60)
            lines.append(f"Q: {r['question']}")
            lines.append("")
            lines.append(f"Intuitive (wrong): {r['intuitive_answer']}")
            lines.append(f"Correct: {r['correct_answer']}")
            lines.append("")
            lines.append(f"Model's answer: {r['model_answer'][:200]}...")
            lines.append("")

            er = r["expansion_ratio"]
            if er == er:
                lines.append(f"Expansion Ratio: {er}")
                lines.append(f"Peak ID: {r['peak_dim']}D (layer {r['peak_layer']})")
                lines.append(f"Final ID: {r['final_dim']}D")
            else:
                lines.append("Expansion Ratio: NaN")

            if trajectory and r["layer_dims"]:
                lines.append("")
                lines.append("ID Trajectory:")
                max_id = max(
                    (ld["intrinsic_dimension"] for ld in r["layer_dims"]
                     if ld["intrinsic_dimension"] == ld["intrinsic_dimension"]),
                    default=1.0
                )
                for ld in r["layer_dims"]:
                    layer_idx = ld["layer"]
                    id_val = ld["intrinsic_dimension"]
                    if id_val != id_val:
                        continue
                    bar_val = id_val / max_id if max_id > 0 else 0
                    bar_len = int(bar_val * 20)
                    bar = "█" * bar_len
                    marker = " ◀" if layer_idx == r["peak_layer"] else ""
                    lines.append(f"  L{layer_idx:2d}: {id_val}D |{bar}{marker}")

            lines.append("")
            lines.append("=" * 70)

        # Summary - just the raw numbers
        lines.extend([
            "",
            "SUMMARY",
            "-" * 30,
            f"Total problems: {len(results)}",
            f"Mean Expansion Ratio: {mean_expansion_ratio}",
        ])

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("reasoning-flow")
def safety_reasoning_flow(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Single prompt to analyze"
    ),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with prompts (one per line)"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only output summary metrics"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Show per-layer flow metrics"
    ),
    tokens: bool = typer.Option(
        False, "--tokens", "-T", help="Show per-token curvature (Zhou et al. methodology)"
    ),
) -> None:
    """Compute reasoning flow geometry (Zhou et al., ICLR 2026).

    Implements the geometric framework from "The Geometry of Reasoning: Flowing
    Logics in Representation Space" for analyzing LLM reasoning trajectories.

    Key insight: LLM reasoning forms smooth flows in embedding space. Logical
    statements act as local controllers governing the velocity of these flows.

    Computes three orders of geometric features:
    - Order-0 (Positions): Embeddings cluster by surface-level semantics
    - Order-1 (Velocities): Trajectories with same logic structure align
    - Order-2 (Menger Curvature): Logic signal intensifies beyond semantics

    For each prompt, reports:
    - Arc length: Total path length in embedding space
    - Mean velocity: Average step size through layers
    - Mean curvature: Average bending of trajectory (Menger κ)
    - Max curvature: Peak curvature (sharpest turn)
    - Smoothness: 1/(1+mean_curvature) - higher = smoother flow
    - Directness: straight_line_dist/arc_length - higher = more direct

    Use --tokens (-T) for token-level curvature (WHERE in the prompt the reasoning bends).
    Use --trajectory (-t) for layer-level curvature (architectural property).

    Reference: Zhou et al. (2025) arXiv:2510.09782
    GitHub: https://github.com/MasterZhou1/Reasoning-Flow

    Examples:
        mc safety reasoning-flow --model ./my-model --prompt "What is 2+2?" -T
        mc safety reasoning-flow --model ./my-model --probes prompts.txt -t
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3060",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Validate inputs
    if prompt is None and probes is None:
        error = ErrorDetail(
            code="MC-3061",
            title="No input provided",
            detail="Must provide either --prompt or --probes",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Collect prompts
    prompt_list: list[str] = []
    if prompt:
        prompt_list.append(prompt)
    if probes:
        probes_path = Path(probes)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-3062",
                title="Probes file not found",
                detail=f"Probes file does not exist: {probes}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        prompt_list.extend(
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        )

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        provider = get_activation_provider()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Get model layer count
        base_model = getattr(loaded_model, "model", loaded_model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError("Could not find model layers")
        num_layers = len(layers)

        results: list[dict] = []

        for prompt_text in prompt_list:
            # Collect full trajectory activations
            trajectory_data = provider.collect_trajectory_batch(
                loaded_model, tokenizer, [prompt_text]
            )

            if trajectory_data.total_tokens < 3:
                results.append({
                    "prompt": prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text,
                    "error": "insufficient_tokens",
                    "token_count": trajectory_data.total_tokens,
                })
                continue

            # Analyze flow at each layer - positions are already backend arrays
            layer_profiles = service.analyze_reasoning_flow(trajectory_data.positions)

            # Aggregate metrics across layers
            arc_lengths = [p.metrics.arc_length for p in layer_profiles]
            mean_velocities = [p.metrics.mean_velocity_norm for p in layer_profiles]
            mean_curvatures = [p.metrics.mean_curvature for p in layer_profiles]
            max_curvatures = [p.metrics.max_curvature for p in layer_profiles]
            smoothnesses = [p.metrics.smoothness for p in layer_profiles]
            directnesses = [p.metrics.directness for p in layer_profiles]

            # Overall flow characteristics
            overall_arc = sum(arc_lengths)
            overall_mean_velocity = sum(mean_velocities) / len(mean_velocities) if mean_velocities else 0.0
            overall_mean_curvature = sum(mean_curvatures) / len(mean_curvatures) if mean_curvatures else 0.0
            overall_max_curvature = max(max_curvatures) if max_curvatures else 0.0
            overall_smoothness = sum(smoothnesses) / len(smoothnesses) if smoothnesses else 0.0
            overall_directness = sum(directnesses) / len(directnesses) if directnesses else 0.0

            # Peak curvature layer (sharpest turn)
            if max_curvatures:
                peak_curv_layer = max_curvatures.index(max(max_curvatures))
                peak_curv_value = max_curvatures[peak_curv_layer]
            else:
                peak_curv_layer = -1
                peak_curv_value = 0.0

            # Build layer detail
            layer_detail = []
            for p in layer_profiles:
                layer_detail.append({
                    "layer": p.layer_idx,
                    "arc_length": p.metrics.arc_length,
                    "mean_velocity": p.metrics.mean_velocity_norm,
                    "velocity_variance": p.metrics.velocity_variance,
                    "mean_curvature": p.metrics.mean_curvature,
                    "max_curvature": p.metrics.max_curvature,
                    "curvature_integral": p.metrics.curvature_integral,
                    "smoothness": p.metrics.smoothness,
                    "directness": p.metrics.directness,
                })

            # Token-level curvature analysis (Zhou et al. methodology)
            token_profile = service.analyze_token_curvature(trajectory_data.positions)

            # Get token strings for display
            token_ids = tokenizer.encode(prompt_text)
            token_strings = [tokenizer.decode([tid]) for tid in token_ids]

            token_detail = []
            for i, curv in enumerate(token_profile.token_curvatures):
                tok_idx = token_profile.token_indices[i]
                tok_str = token_strings[tok_idx] if tok_idx < len(token_strings) else "?"
                token_detail.append({
                    "token_idx": tok_idx,
                    "token": tok_str,
                    "curvature": curv,
                    "is_peak": tok_idx == token_profile.peak_token_idx,
                })

            results.append({
                "prompt": prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text,
                "full_prompt": prompt_text,
                "token_count": trajectory_data.total_tokens,
                "overall": {
                    "total_arc_length": overall_arc,
                    "mean_velocity": overall_mean_velocity,
                    "mean_curvature": overall_mean_curvature,
                    "max_curvature": overall_max_curvature,
                    "peak_curvature_layer": peak_curv_layer,
                    "smoothness": overall_smoothness,
                    "directness": overall_directness,
                },
                "layers": layer_detail,
                "token_curvature": {
                    "peak_token_idx": token_profile.peak_token_idx,
                    "peak_token": token_strings[token_profile.peak_token_idx] if token_profile.peak_token_idx < len(token_strings) else "?",
                    "peak_curvature": token_profile.peak_curvature,
                    "mean_curvature": token_profile.mean_curvature,
                    "std_curvature": token_profile.std_curvature,
                    "tokens": token_detail,
                },
            })

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3063",
            title="Reasoning flow analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": str(model_path),
        "numLayers": num_layers,
        "reference": "Zhou et al. (2025) arXiv:2510.09782",
        "results": results,
    }

    if context.output_format == "text":
        if quiet:
            # Quiet mode: just output key metrics
            for r in results:
                if "error" in r:
                    write_output(f"{r['prompt']}: {r['error']}", context.output_format, context.pretty)
                else:
                    o = r["overall"]
                    write_output(
                        f"κ_mean={o['mean_curvature']:.4f} κ_max={o['max_curvature']:.4f} "
                        f"smooth={o['smoothness']:.4f} direct={o['directness']:.4f}",
                        context.output_format, context.pretty
                    )
        else:
            lines = [
                "REASONING FLOW GEOMETRY (Zhou et al., ICLR 2026)",
                f"Model: {model_path}",
                f"Layers: {num_layers}",
                "",
                "Reference: arXiv:2510.09782",
                "GitHub: github.com/MasterZhou1/Reasoning-Flow",
                "",
            ]

            for r in results:
                lines.append("=" * 70)
                lines.append(f"Prompt: {r['prompt']}")
                lines.append(f"Tokens: {r.get('token_count', 'N/A')}")

                if "error" in r:
                    lines.append(f"Error: {r['error']}")
                    continue

                o = r["overall"]
                lines.append("")
                lines.append("Overall Flow Metrics:")
                lines.append(f"  Total arc length:    {o['total_arc_length']}")
                lines.append(f"  Mean velocity:       {o['mean_velocity']}")
                lines.append(f"  Mean curvature (κ):  {o['mean_curvature']}")
                lines.append(f"  Max curvature (κ):   {o['max_curvature']} (layer {o['peak_curvature_layer']})")
                lines.append(f"  Smoothness:          {o['smoothness']}")
                lines.append(f"  Directness:          {o['directness']}")

                if trajectory and r.get("layers"):
                    lines.append("")
                    lines.append("Per-Layer Curvature Profile:")
                    max_curv = max((ld["max_curvature"] for ld in r["layers"]), default=1.0)
                    for ld in r["layers"]:
                        layer_idx = ld["layer"]
                        curv = ld["max_curvature"]
                        bar_val = curv / max_curv if max_curv > 0 else 0
                        bar_len = int(bar_val * 20)
                        bar = "█" * bar_len
                        marker = " ◀ peak" if layer_idx == o["peak_curvature_layer"] else ""
                        lines.append(f"  L{layer_idx:2d}: κ={curv:.4f} |{bar}{marker}")

                if tokens and r.get("token_curvature"):
                    tc = r["token_curvature"]
                    lines.append("")
                    lines.append("Per-Token Curvature (Zhou et al. methodology):")
                    lines.append(f"  Peak token: [{tc['peak_token_idx']}] '{tc['peak_token']}' (κ={tc['peak_curvature']:.4f})")
                    lines.append(f"  Mean: {tc['mean_curvature']:.4f}, Std: {tc['std_curvature']:.4f}")
                    lines.append("")
                    if tc.get("tokens"):
                        max_tok_curv = max((t["curvature"] for t in tc["tokens"]), default=1.0)
                        for t in tc["tokens"]:
                            curv = t["curvature"]
                            bar_val = curv / max_tok_curv if max_tok_curv > 0 else 0
                            bar_len = int(bar_val * 30)
                            bar = "█" * bar_len
                            marker = " ◀ PEAK" if t["is_peak"] else ""
                            tok_display = repr(t["token"])[:12].ljust(12)
                            lines.append(f"  [{t['token_idx']:2d}] {tok_display} κ={curv:.3f} |{bar}{marker}")

                lines.append("")

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
    """Compute per-layer spectral entropy profile.

    Computes spectral entropy from SVD singular value distributions at each layer.
    Tracks how variance is distributed across dimensions:

    - Spectral Entropy = -Σ p_i * log(p_i) where p_i = σ_i² / Σσ²
    - High entropy: Variance spread across many dimensions (expansion)
    - Low entropy: Variance concentrated in few dimensions (compression)

    Expected pattern (MANIFOLD-LEARNING-SYNTHESIS.md):
    - Entry layers: Moderate entropy
    - Middle layers: Peak entropy (maximum exploration/complexity)
    - Exit layers: Reduced entropy (convergence to output)

    Expansion ratio ≈ 1.0 indicates flat trajectory (peak ≈ final).

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
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service
        from modelcypher.ports.activation_provider import get_activation_provider

        service = get_geometry_analysis_service()
        backend = service.backend
        provider = get_activation_provider()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

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

        # Compute spectral entropy at each layer using GeometryAnalysisService
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
                layer_entropy_result = service.compute_layer_entropy(
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
            # Entropy expansion ratio = peak_entropy / min_entropy
            if min_entropy > 0:
                entropy_expansion_ratio = max_entropy / min_entropy
            else:
                entropy_expansion_ratio = float("nan")

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
            entropy_expansion_ratio = float("nan")
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
        "entropyExpansionRatio": entropy_expansion_ratio,
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
            f"  Mean Entropy: {mean_entropy}",
            f"  Peak Entropy: {max_entropy} (layer {peak_idx})",
            f"  Min Entropy: {min_entropy} (layer {trough_idx})",
            f"  Expansion Ratio: {expansion_ratio}×" if expansion_ratio == expansion_ratio else "  Expansion Ratio: NaN",
            f"  Entropy Expansion Ratio: {entropy_expansion_ratio}" if entropy_expansion_ratio == entropy_expansion_ratio else "  Entropy Expansion Ratio: NaN",
            f"  Monotonicity: {monotonicity}" if monotonicity == monotonicity else "  Monotonicity: NaN",
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


@app.command("jacobian-trace")
def safety_jacobian_trace(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt to analyze"),
    top_k: int = typer.Option(
        20, "--top-k", help="Number of top singular values to display per layer"
    ),
    num_probes: int = typer.Option(
        64, "--num-probes", help="Number of random probes for randomized SVD (higher = more accurate)"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only output summary metrics"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Show per-layer singular value trajectory"
    ),
) -> None:
    """Compute Jacobian spectrum at each layer (Mathematical Anatomy).

    Analyzes the layer-to-layer transformation Jacobian ∂h_l/∂h_{l-1}
    using randomized SVD to estimate singular value spectrum.

    What the Jacobian tells us:
    - σ_i > 1: Direction i is AMPLIFIED through the layer
    - σ_i < 1: Direction i is COMPRESSED
    - σ_i ≈ 1: Direction i is PRESERVED

    Key metrics:
    - Effective Rank: How many directions carry information
    - Condition Number: κ = σ_max/σ_min (numerical stability)
    - Spectral Gap: σ_1/σ_2 (dominance of first component)
    - Norm Amplification: σ_max (worst-case growth)

    Hypothesis (from research plan):
    - Reasoning models (DeepSeek-R1) have different Jacobian spectra than
      intuitive models (LFM2)
    - The 1356× norm amplification in reasoning may come from Jacobian structure
    - Bottleneck layers (low effective rank) are information compression points

    Technical note: Uses randomized range finder (Halko et al. 2011) to avoid
    materializing the full [hidden_dim × hidden_dim] Jacobian matrix.

    Examples:
        mc safety jacobian-trace --model ./my-model --prompt "What is 2+2?"
        mc safety jacobian-trace --model ./my-model -p "A bat and ball cost \\$1.10..." --trajectory
    """
    context = _context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3070",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_geometry_analysis_service

        service = get_geometry_analysis_service()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Compute Jacobian trace
        result = service.trace_jacobian_spectrum(
            model=loaded_model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_path=str(model_path),
            num_probes=num_probes,
        )

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3071",
            title="Jacobian trace analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = result.as_dict()

    if context.output_format == "text":
        if quiet:
            # Quiet mode: just summary metrics
            lines = [
                f"mean_eff_rank={result.mean_effective_rank:.2f}",
                f"max_amplification={result.max_norm_amplification:.4f}",
                f"bottleneck_layer={result.bottleneck_layer}",
                f"expansion_layer={result.expansion_layer}",
            ]
            write_output(" ".join(lines), context.output_format, context.pretty)
        else:
            lines = [
                "JACOBIAN SPECTRUM ANALYSIS (Mathematical Anatomy)",
                f"Model: {model_path}",
                f"Layers: {result.total_layers}",
                f"Probes: {num_probes}",
                f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}",
                "",
                "Summary:",
                f"  Mean Effective Rank: {result.mean_effective_rank:.2f}",
                f"  Mean Condition Number: {result.mean_condition_number:.2f}",
                f"  Max Norm Amplification: {result.max_norm_amplification:.4f}",
                f"  Cumulative Amplification: {result.cumulative_amplification:.2e}",
                f"  Bottleneck Layer: {result.bottleneck_layer} (lowest effective rank)",
                f"  Expansion Layer: {result.expansion_layer} (highest amplification)",
                "",
            ]

            if trajectory:
                lines.append("Per-Layer Jacobian Spectrum:")
                lines.append("")

                # Find max amplification for normalization
                max_amp = max(p.norm_amplification for p in result.profiles) or 1.0

                for p in result.profiles:
                    # Singular value bar visualization
                    amp_ratio = p.norm_amplification / max_amp if max_amp > 0 else 0
                    bar_len = int(amp_ratio * 30)
                    bar = "█" * bar_len + "░" * (30 - bar_len)

                    # Markers
                    markers = []
                    if p.layer_idx == result.bottleneck_layer:
                        markers.append("BOTTLENECK")
                    if p.layer_idx == result.expansion_layer:
                        markers.append("EXPANSION")
                    marker_str = f" ◀ {', '.join(markers)}" if markers else ""

                    lines.append(f"Layer {p.layer_idx:3d}:")
                    lines.append(f"  σ_max={p.norm_amplification:.4f} |{bar}|{marker_str}")
                    lines.append(f"  eff_rank={p.effective_rank_shannon:.1f}  κ={p.condition_number:.1f}  gap={p.spectral_gap:.2f}")

                    # Top singular values (truncated)
                    sv_display = [f"{s:.3f}" for s in p.top_k_singular_values[:min(top_k, 8)]]
                    if len(p.top_k_singular_values) > 8:
                        sv_display.append("...")
                    lines.append(f"  top_σ: [{', '.join(sv_display)}]")
                    lines.append("")

            write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("circuit-breaker")
def safety_circuit_breaker(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Evaluate circuit breaker state.

    Examples:
        mc safety circuit-breaker --job abc123
    """
    context = _context(ctx)
    service = get_geometry_safety_service()
    state, _signals = service.evaluate_circuit_breaker(
        job_id=job_id,
    )

    output = {
        "severity": state.severity,
        "dominantSource": state.dominant_source.value if state.dominant_source else None,
    }

    if context.output_format == "text":
        lines = [
            "CIRCUIT BREAKER EVALUATION",
            f"Severity: {output['severity']:.3f}",
        ]
        if output["dominantSource"] is not None:
            lines.append(f"Dominant Source: {output['dominantSource']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("persona")
def safety_persona(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Analyze persona drift for a training job.

    Examples:
        mc safety persona --job abc123
    """
    context = _context(ctx)
    service = get_geometry_safety_service()
    drift_info = service.persona_drift(job_id)
    if drift_info is None:
        raise typer.BadParameter(f"Job '{job_id}' not found or has no persona drift metrics.")

    output = {
        "jobId": job_id,
        "overallDriftMagnitude": drift_info.overall_drift_magnitude,
        "driftingTraits": drift_info.drifting_traits,
        "refusalDistance": drift_info.refusal_distance,
        "isApproachingRefusal": drift_info.is_approaching_refusal,
    }

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT ANALYSIS",
            f"Job: {output['jobId']}",
            f"Drift Magnitude: {output['overallDriftMagnitude']:.4f}",
        ]
        if output["driftingTraits"]:
            lines.append(f"Drifting Traits: {', '.join(output['driftingTraits'])}")
        if output["refusalDistance"] is not None:
            approaching = "YES" if output.get("isApproachingRefusal") else "NO"
            lines.append(
                f"Refusal Distance: {output['refusalDistance']:.4f} (Approaching: {approaching})"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("jailbreak-test")
def safety_jailbreak_test(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompts: str | None = typer.Option(
        None, "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    prompt: list[str] | None = typer.Option(None, "--prompt", help="Individual prompt(s) to test"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter to apply"),
) -> None:
    """Execute jailbreak entropy analysis to test model safety boundaries.

    Examples:
        mc safety jailbreak-test --model ./model --prompts ./prompts.json
        mc safety jailbreak-test --model ./model --prompt "test prompt"
    """
    context = _context(ctx)

    # Collect prompts from file or individual --prompt flags
    prompt_list: list[str] = []
    if prompts:
        # Will be loaded from file by the service
        prompt_input: list[str] | str = prompts
    elif prompt:
        prompt_list = list(prompt)
        prompt_input = prompt_list
    else:
        raise typer.BadParameter("Provide either --prompts file or --prompt values")

    service = get_geometry_safety_service()
    result = service.jailbreak_test(
        model_path=model,
        prompts=prompt_input,
        adapter_path=adapter,
    )

    output = {
        "modelPath": result.model_path,
        "adapterPath": result.adapter_path,
        "promptsTested": result.prompts_tested,
        "vulnerabilitiesFound": result.vulnerabilities_found,
        "meanThresholdExceedance": result.mean_threshold_exceedance,
        "processingTime": result.processing_time,
        "vulnerabilityDetails": [
            {
                "prompt": v.prompt[:100] + "..." if len(v.prompt) > 100 else v.prompt,
                "vulnerabilityType": v.vulnerability_type,
                "baselineEntropy": v.baseline_entropy,
                "attackEntropy": v.attack_entropy,
                "deltaH": v.delta_h,
                "thresholdExceedance": v.threshold_exceedance,
                "attackVector": v.attack_vector,
            }
            for v in result.vulnerability_details
        ],
    }

    if context.output_format == "text":
        lines = [
            "JAILBREAK TEST RESULTS",
            f"Model: {result.model_path}",
        ]
        if result.adapter_path:
            lines.append(f"Adapter: {result.adapter_path}")
        lines.append(f"Prompts Tested: {result.prompts_tested}")
        lines.append(f"Vulnerabilities Found: {result.vulnerabilities_found}")
        lines.append(f"Mean Threshold Exceedance: {result.mean_threshold_exceedance:.2f}")
        lines.append(f"Processing Time: {result.processing_time:.2f}s")

        if result.vulnerability_details:
            lines.append("")
            lines.append("VULNERABILITY DETAILS:")
            for i, v in enumerate(
                result.vulnerability_details[:10], 1
            ):  # Limit to 10 in text output
                lines.append(
                    f"  {i}. {v.vulnerability_type} via {v.attack_vector}"
                )
                lines.append(f"     Prompt: {v.prompt[:60]}...")
                lines.append(f"     Delta H: {v.delta_h:.3f}, Threshold Exceedance: {v.threshold_exceedance:.2f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("probe-redteam")
def safety_probe_redteam(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Scan adapter metadata for threat indicators (static analysis).

    Examples:
        mc safety probe-redteam --name my-adapter
        mc safety probe-redteam --name my-adapter --tag skill1 --tag skill2
    """
    context = _context(ctx)
    source, _ = EmbeddingDefaults.resolved_source()
    if source == "http":
        warn_network(
            context,
            f"Embedding provider uses HTTP endpoint from {EmbeddingDefaults.EMBEDDING_API_URL_ENV}.",
        )
    service = SafetyProbeService(embedder=EmbeddingDefaults.make_default_embedder())

    indicators = service.scan_adapter_metadata(
        name=name,
        description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.threat_indicators_payload(indicators)

    if context.output_format == "text":
        lines = [
            "RED TEAM STATIC ANALYSIS",
            f"Adapter: {name}",
            f"Threat Indicators: {payload['count']}",
            f"Max Mean Distance: {payload['maxMeanDistance']:.4f}",
        ]
        if indicators:
            lines.append("")
            lines.append("DETECTED OUTLIERS:")
            for ind in indicators:
                lines.append(
                    f"  [{ind.mean_distance:.4f}] {ind.field}: {ind.text}"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-behavioral")
def safety_probe_behavioral(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Run behavioral probes (requires inference hook for full analysis).

    Examples:
        mc safety probe-behavioral --name my-adapter
    """

    context = _context(ctx)
    source, _ = EmbeddingDefaults.resolved_source()
    if source == "http":
        warn_network(
            context,
            f"Embedding provider uses HTTP endpoint from {EmbeddingDefaults.EMBEDDING_API_URL_ENV}.",
        )
    service = SafetyProbeService(embedder=EmbeddingDefaults.make_default_embedder())

    result = service.run_behavioral_probes(
        adapter_name=name,
        adapter_description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.composite_result_payload(result)

    if context.output_format == "text":
        lines = [
            "BEHAVIORAL PROBE RESULTS",
            f"Adapter: {name}",
            f"Any Findings: {payload['anyFindings']}",
            f"Probes Run: {payload['probeCount']}",
        ]
        if payload["aggregateFindingCounts"]:
            counts_str = ", ".join(
                f"{k}: {v}" for k, v in payload["aggregateFindingCounts"].items()
            )
            lines.append(f"Aggregate Finding Counts: {counts_str}")
        if payload["anyFindings"]:
            lines.append("")
            lines.append("PROBES WITH FINDINGS:")
            for r in result.probe_results:
                if r.has_findings:
                    counts = r.finding_counts or {}
                    counts_str = ", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "none"
                    lines.append(f"  {r.probe_name}: {r.details} (counts: {counts_str})")
        if payload["allFindings"]:
            lines.append("")
            lines.append("FINDINGS:")
            for finding in payload["allFindings"][:10]:
                lines.append(f"  - {finding}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# BENCHMARK COMMANDS
# =============================================================================


@app.command("benchmark")
def run_benchmark(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    suite: str = typer.Option(
        "quick", "--suite", "-s", help="Benchmark suite (quick, reasoning, factual, comprehensive)"
    ),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory for results"),
) -> None:
    """Run benchmark suite with geometric metrics.

    Suites:
        quick: gsm8k, arc_easy, boolq
        reasoning: gsm8k, arc_challenge, hellaswag
        factual: mmlu, arc_easy, boolq
        comprehensive: All of the above

    Examples:
        mc analyze benchmark /path/to/model --suite quick
        mc analyze benchmark /path/to/model --suite comprehensive -o ./results
    """
    from modelcypher.core.use_cases.benchmark_service import BenchmarkService

    context = _context(ctx)
    service = BenchmarkService()

    typer.echo(f"Running benchmark suite: {suite}")

    # This is a simplified interface - full benchmark requires model loading
    payload = {
        "suite": suite,
        "model": model,
        "status": "benchmark_service_available",
        "note": "Full benchmark requires model loading and inference. Use BenchmarkService directly for detailed results.",
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# LORA DIAGNOSTIC COMMANDS
# =============================================================================


@app.command("lora-svd")
def lora_svd_diagnostic(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to LoRA adapter"),
    base_model: str = typer.Option(..., "--base", "-b", help="Path to base model"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Show top-k layers by change"),
) -> None:
    """Analyze LoRA adapter with SVD decomposition.

    Shows rank changes, null space components, and subspace overlap per layer.
    Useful for understanding what a LoRA adapter is actually doing geometrically.

    Examples:
        mc analyze lora-svd ./my-adapter --base /path/to/base
        mc analyze lora-svd ./my-adapter --base /path/to/base --top-k 10
    """
    from modelcypher.core.use_cases.lora_diagnostic_service import (
        LayerSVDReport,
        run_diagnostic,
    )

    context = _context(ctx)

    typer.echo(f"Analyzing LoRA adapter: {adapter_path}")

    report = run_diagnostic(model_path=base_model, adapter_path=adapter_path)

    # Sort by frobenius delta (relative change)
    sorted_reports = sorted(
        report.layer_svd, key=lambda r: abs(r.frobenius_delta), reverse=True
    )

    if context.output_format == "text":
        lines = [
            "LORA SVD DIAGNOSTIC",
            f"Adapter: {adapter_path}",
            f"Base model: {base_model}",
            f"Layers with LoRA: {report.layers_with_lora}",
            f"Total params modified: {report.total_params_modified}",
            "",
            f"TOP {top_k} LAYERS BY FROBENIUS CHANGE:",
        ]
        for r in sorted_reports[:top_k]:
            lines.append(
                f"  Layer {r.layer_idx} ({r.weight_name}): "
                f"rank {r.rank_before}→{r.rank_after} (Δ{r.rank_delta:+d}), "
                f"frob_delta={r.frobenius_delta:.4f}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "adapter_path": adapter_path,
        "base_model": base_model,
        "layers_with_lora": report.layers_with_lora,
        "total_params_modified": report.total_params_modified,
        "avg_null_space_activation": report.avg_null_space_activation,
        "avg_subspace_overlap": report.avg_subspace_overlap,
        "peak_change_layer": report.peak_change_layer,
        "top_layers": [
            {
                "layer_idx": r.layer_idx,
                "weight_name": r.weight_name,
                "rank_before": r.rank_before,
                "rank_after": r.rank_after,
                "rank_delta": r.rank_delta,
                "frobenius_delta": r.frobenius_delta,
            }
            for r in sorted_reports[:top_k]
        ],
    }
    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# SPARSE REGION COMMANDS
# =============================================================================


@app.command("sparse-region")
def sparse_region_analysis(
    ctx: typer.Context,
    list_domains: bool = typer.Option(False, "--list-domains", "-l", help="List available sparse region domains"),
    list_pairs: bool = typer.Option(False, "--list-pairs", "-p", help="List contrastive pairs for refusal detection"),
) -> None:
    """Explore sparse activation regions and refusal directions.

    Sparse regions in activation space can correspond to specific behaviors
    like refusal or domain-specific knowledge.

    Examples:
        mc analyze sparse-region --list-domains
        mc analyze sparse-region --list-pairs
    """
    from modelcypher.core.use_cases.geometry_sparse_service import (
        GeometrySparseService,
    )

    context = _context(ctx)
    service = GeometrySparseService()

    if list_pairs:
        pairs = service.get_contrastive_pairs()
        payload = GeometrySparseService.contrastive_pairs_payload(pairs)
    else:
        # Default to listing domains
        domains = service.list_domains()
        payload = GeometrySparseService.domains_payload(domains)

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# KNOWLEDGE ANALYSIS COMMANDS
# =============================================================================


@app.command("knowledge-type")
def knowledge_type_analysis(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    statement: str = typer.Option(..., "--statement", "-s", help="Statement to analyze"),
    counterfactual: str = typer.Option(..., "--counterfactual", "-c", help="Counterfactual version"),
    layer: int = typer.Option(..., "--layer", "-l", help="Layer index to analyze"),
) -> None:
    """Analyze whether a statement is factual knowledge or opinion.

    Uses counterfactual sensitivity to distinguish facts from opinions:
    - Facts: high sensitivity (~0.2+), representation changes when violated
    - Opinions: low sensitivity (~0.06), similar representation regardless

    Examples:
        mc analyze knowledge-type /path/to/model \\
            --statement "The capital of France is Paris" \\
            --counterfactual "The capital of France is Madrid" \\
            --layer 12
    """
    from modelcypher.cli.composition import get_activation_provider, get_backend
    from modelcypher.core.use_cases.knowledge_analyzer import KnowledgeAnalyzer

    context = _context(ctx)

    typer.echo(f"Analyzing knowledge type at layer {layer}")

    analyzer = KnowledgeAnalyzer(
        activation_provider=get_activation_provider(),
        backend=get_backend(),
    )

    # Note: Full analysis requires loading the model
    payload = {
        "model": model,
        "statement": statement,
        "counterfactual": counterfactual,
        "layer": layer,
        "status": "knowledge_analyzer_available",
        "note": "Full analysis requires model loading. Use KnowledgeAnalyzer.analyze_statement() directly.",
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# CURRICULUM PROFILING COMMANDS
# =============================================================================


@app.command("curriculum-profile")
def curriculum_profile(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    problems_file: str = typer.Option(..., "--problems", "-p", help="JSON file with problems to profile"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output CSV file"),
) -> None:
    """Profile training problems by geometric difficulty.

    Measures difficulty using geometric signals:
    - CKA similarity to reference
    - Activation barrier height
    - Fisher Information
    - Trajectory curvature
    - Local density
    - Intrinsic dimension

    Examples:
        mc analyze curriculum-profile /path/to/model --problems problems.json
        mc analyze curriculum-profile /path/to/model --problems problems.json -o difficulty.csv
    """
    from modelcypher.core.use_cases.curriculum_profiler import CurriculumProfiler

    context = _context(ctx)

    typer.echo(f"Profiling curriculum difficulty")

    payload = {
        "model": model,
        "problems_file": problems_file,
        "output_file": output_file,
        "status": "curriculum_profiler_available",
        "metrics": [
            "cka_similarity",
            "activation_barrier",
            "fisher_information",
            "trajectory_curvature",
            "local_density",
            "intrinsic_dimension",
        ],
        "note": "Full profiling requires model loading. Use CurriculumProfiler directly for detailed results.",
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# ENTROPY MONITOR COMMANDS
# =============================================================================


@app.command("uncertainty-modes")
def uncertainty_modes(
    ctx: typer.Context,
) -> None:
    """List available uncertainty response modes.

    Modes determine how the model responds when uncertainty is detected:
    - Butler: No exploration, answer with available knowledge or decline
    - Autonomous: Research gaps automatically (query memory, search)
    - Human-in-loop: Pause at uncertainty, ask for guidance

    Examples:
        mc analyze uncertainty-modes
    """
    from modelcypher.core.use_cases.entropy_monitor import (
        UncertaintyAction,
        UncertaintyMode,
    )

    context = _context(ctx)

    payload = {
        "modes": [
            {
                "name": mode.value,
                "description": {
                    "butler": "No exploration. Answer with available knowledge or decline.",
                    "autonomous": "Research gaps automatically. Query memory, search, augment context.",
                    "human_in_loop": "Pause at uncertainty. Ask user for guidance before proceeding.",
                }[mode.value],
            }
            for mode in UncertaintyMode
        ],
        "actions": [
            {
                "name": action.value,
                "description": {
                    "proceed": "Continue generation - uncertainty is acceptable.",
                    "abstain": "Stop generation - uncertainty too high.",
                    "retrieve": "Pause and retrieve - query memory/search before continuing.",
                    "ask": "Pause and ask - request user guidance.",
                    "warn": "Continue with warning - hallucination risk detected.",
                }[action.value],
            }
            for action in UncertaintyAction
        ],
    }

    if context.output_format == "text":
        lines = [
            "UNCERTAINTY RESPONSE MODES",
            "",
            "MODES (user-configurable):",
        ]
        for m in payload["modes"]:
            lines.append(f"  {m['name']}: {m['description']}")
        lines.append("")
        lines.append("ACTIONS (system-determined):")
        for a in payload["actions"]:
            lines.append(f"  {a['name']}: {a['description']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# ENTROPY PROBE COMMANDS
# =============================================================================


@app.command("entropy-pattern")
def entropy_pattern_analysis(
    ctx: typer.Context,
    samples_file: str = typer.Option(..., "--samples", "-s", help="JSON file with (entropy, variance) samples"),
    detect_distress: bool = typer.Option(False, "--detect-distress", "-d", help="Detect distress patterns"),
) -> None:
    """Analyze entropy/variance samples for patterns.

    Detects trends, anomalies, and potential distress signals in
    entropy time series data.

    Examples:
        mc analyze entropy-pattern --samples samples.json
        mc analyze entropy-pattern --samples samples.json --detect-distress
    """
    import json
    from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

    context = _context(ctx)
    service = EntropyProbeService()

    # Load samples from file
    try:
        with open(samples_file) as f:
            data = json.load(f)
        samples = [(s["entropy"], s["variance"]) for s in data["samples"]]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        payload = {"error": f"Failed to load samples: {e}"}
        write_output(payload, context.output_format, context.pretty)
        return

    pattern = service.analyze_pattern(samples)

    if detect_distress:
        distress = service.detect_distress(samples)
        payload = {
            "pattern": {
                "trend": pattern.trend.value if hasattr(pattern.trend, "value") else str(pattern.trend),
                "sample_count": len(samples),
            },
            "distress_detected": distress is not None,
            "distress": {
                "type": distress.distress_type if distress else None,
                "confidence": distress.confidence if distress else None,
            } if distress else None,
        }
    else:
        payload = {
            "pattern": {
                "trend": pattern.trend.value if hasattr(pattern.trend, "value") else str(pattern.trend),
                "sample_count": len(samples),
            },
        }

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy-baseline-verify")
def entropy_baseline_verify(
    ctx: typer.Context,
    baseline_file: str = typer.Option(..., "--baseline", "-b", help="Path to baseline JSON"),
    deltas_file: str = typer.Option(..., "--deltas", "-d", help="JSON file with observed delta values"),
    adapter_path: str | None = typer.Option(None, "--adapter", "-a", help="Path to adapter (for reporting)"),
) -> None:
    """Verify observed entropy deltas against declared baseline.

    Compares observed delta values against a previously computed baseline
    to detect unexpected entropy shifts.

    Examples:
        mc analyze entropy-baseline-verify --baseline baseline.json --deltas observed.json
    """
    import json
    from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

    context = _context(ctx)
    service = EntropyProbeService()

    # Load deltas
    try:
        with open(deltas_file) as f:
            data = json.load(f)
        deltas = data.get("deltas", data.get("values", []))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        payload = {"error": f"Failed to load deltas: {e}"}
        write_output(payload, context.output_format, context.pretty)
        return

    result = service.verify_baseline(
        baseline_path=baseline_file,
        observed_deltas=deltas,
        adapter_path=adapter_path,
    )

    payload = {
        "baseline_file": baseline_file,
        "deltas_file": deltas_file,
        "adapter_path": adapter_path,
        "verified": result.verified,
        "comparison": {
            "mean_difference": result.comparison.mean_difference,
            "std_difference": result.comparison.std_difference,
        } if hasattr(result, "comparison") else None,
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# CONCEPT RESPONSE MATRIX COMMANDS
# =============================================================================


@app.command("crm-build")
def crm_build(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for CRM"),
    adapter: str | None = typer.Option(None, "--adapter", "-a", help="Optional adapter path"),
) -> None:
    """Build Concept Response Matrix for a model.

    Computes activations for semantic anchors (primes, gates, emotions)
    and stores them for cross-architecture comparison.

    Examples:
        mc analyze crm-build /path/to/model --output ./crm/model1
        mc analyze crm-build /path/to/model --output ./crm/model1 --adapter ./adapter
    """
    from modelcypher.core.use_cases.concept_response_matrix_service import (
        ConceptResponseMatrixService,
    )

    context = _context(ctx)

    typer.echo(f"Building CRM for: {model}")

    # Note: Full build requires HiddenStateEngine
    payload = {
        "model": model,
        "output": output,
        "adapter": adapter,
        "status": "crm_service_available",
        "anchors": ["semantic_primes", "computational_gates", "sequence_invariants", "emotions"],
        "note": "Full CRM build requires HiddenStateEngine. Use ConceptResponseMatrixService.build() directly.",
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("crm-compare")
def crm_compare(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Path to source CRM"),
    target: str = typer.Argument(..., help="Path to target CRM"),
) -> None:
    """Compare two Concept Response Matrices.

    Computes CKA alignment between source and target CRMs to measure
    cross-architecture semantic similarity.

    Examples:
        mc analyze crm-compare ./crm/model1 ./crm/model2
    """
    from modelcypher.core.use_cases.concept_response_matrix_service import (
        ConceptResponseMatrixService,
    )

    context = _context(ctx)

    typer.echo(f"Comparing CRMs: {source} vs {target}")

    # Note: Full comparison requires loading CRM data
    payload = {
        "source": source,
        "target": target,
        "status": "crm_service_available",
        "metrics": ["mean_cka", "alignment_precision", "layer_correspondence"],
        "note": "Full comparison requires CRM data. Use ConceptResponseMatrixService.compare() directly.",
    }

    write_output(payload, context.output_format, context.pretty)


# =============================================================================
# BILM PROBE COMMANDS
# =============================================================================


@app.command("bilm-probe-info")
def bilm_probe_info(
    ctx: typer.Context,
) -> None:
    """Show information about BiLM probe training.

    BiLM probes use bidirectional language model representations for
    token-level domain classification (e.g., detecting specific content types).

    Examples:
        mc analyze bilm-probe-info
    """
    context = _context(ctx)

    payload = {
        "description": "Bidirectional LM Probe for token-level domain classification",
        "inputs": {
            "forward_positive": "Forward LM activations for positive samples [n_pos, hidden_dim]",
            "backward_positive": "Backward LM activations for positive samples [n_pos, hidden_dim]",
            "forward_negative": "Forward LM activations for negative samples [n_neg, hidden_dim]",
            "backward_negative": "Backward LM activations for negative samples [n_neg, hidden_dim]",
        },
        "outputs": {
            "train_accuracy": "Training accuracy",
            "train_f1": "Training F1 score",
            "val_accuracy": "Validation accuracy (if val_split > 0)",
            "val_f1": "Validation F1 score",
        },
        "hyperparameters": {
            "val_split": "Fraction for validation (default: 0.1)",
            "learning_rate": "Learning rate (default: 0.01)",
            "max_iterations": "Max training iterations (default: 1000)",
        },
        "usage": "Use BiLMProbeService.train() with collected activations from forward and backward LM passes.",
    }

    if context.output_format == "text":
        lines = [
            "BILM PROBE TRAINING",
            "",
            payload["description"],
            "",
            "REQUIRED INPUTS:",
        ]
        for name, desc in payload["inputs"].items():
            lines.append(f"  {name}: {desc}")
        lines.append("")
        lines.append("OUTPUTS:")
        for name, desc in payload["outputs"].items():
            lines.append(f"  {name}: {desc}")
        lines.append("")
        lines.append("HYPERPARAMETERS:")
        for name, desc in payload["hyperparameters"].items():
            lines.append(f"  {name}: {desc}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
