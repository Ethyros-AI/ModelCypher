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

"""Behavioral analysis safety CLI commands.

Provides commands for behavioral analysis of models and adapters:
- adapter-probe: Probe adapter for delta-feature geometry
- behavioral-signature: Compute behavioral signature for a model
- cognitive-reflection-test: Run CRT with geometric analysis
"""

from __future__ import annotations

import asyncio

from modelcypher.cli.exit_codes import EXIT_RUNTIME

from ._common import (
    ErrorDetail,
    Path,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


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
    context = get_context(ctx)

    from modelcypher.core.domain.safety import (
        DeltaFeatureExtractor,
    )

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        error = ErrorDetail(
            code="MC-3001",
            title="Adapter not found",
            detail=f"Adapter path does not exist: {adapter}",
            hint="Ensure the adapter path points to a valid adapter directory.",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3003",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
            hint="Provide comma-separated integers (e.g., --layers 4,8,12).",
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
                    hint="Ensure the baseline model path points to a valid directory with config.json.",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3050",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
                    bar = "#" * bar_len
                    marker = " <" if layer_idx == r["peak_layer"] else ""
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
