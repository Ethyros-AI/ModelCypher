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
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.use_cases.behavioral_analyzer import BehavioralAnalyzer
        from modelcypher.core.use_cases.model_service import ModelService
        from modelcypher.ports.activation_provider import get_activation_provider

        backend = get_default_backend()
        provider = get_activation_provider()
        model_service = ModelService()

        # Load model
        loaded_model, tokenizer = model_service.load_model(str(model_path))

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

            baseline_model, baseline_tokenizer = model_service.load_model(str(baseline_path))
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
