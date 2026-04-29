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

"""Geometric analysis CLI commands.

Provides commands for geometric analysis of model representations:
- chain-profile: Unified causal chain diagnostic (entropy → curvature → ID → phase)
- concept-volume: Multi-concept Riemannian volume analysis
- dimension-profile: Per-layer intrinsic dimension profiling
- entropy-trajectory: Layer-wise entropy trajectory analysis
- expansion-ratio: TwoNN intrinsic dimension expansion ratio
- reasoning-flow: Zhou et al. reasoning flow geometry
- spectral-trajectory: Spectral entropy profile
- jacobian-trace: Jacobian spectrum analysis
"""

from __future__ import annotations

from enum import Enum

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


class VerificationDepthMode(str, Enum):
    """Grouping mode for verification-depth profiling."""

    CUMULATIVE = "cumulative"
    EXACT = "exact"


@app.command("dimension-profile")
def safety_dimension_profile(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    adapter: str | None = typer.Option(
        None, "--adapter", help="Optional adapter directory to load before profiling"
    ),
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
        mc analyze dimension-profile --model ./my-model
        mc analyze dimension-profile --model ./my-model --samples 100
        mc analyze dimension-profile --model ./my-model --recovery
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3020",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    adapter_path = Path(adapter) if adapter else None
    if adapter_path is not None and not adapter_path.exists():
        error = ErrorDetail(
            code="MC-3023",
            title="Adapter not found",
            detail=f"Adapter path does not exist: {adapter}",
            hint="Provide a valid adapter directory or omit --adapter.",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
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
        loaded_model, tokenizer = loader.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path is not None else None,
        )

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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    # Get hidden dimension from first activation
    hidden_dim = int(backend.shape(stacked)[1]) if layer_activations[0] else 0

    # Compute recovery metrics if requested
    final_id = valid_ids[-1] if valid_ids else 0
    recovery_ratio = final_id / min_id if min_id > 0 else 0

    payload = {
        "modelPath": str(model_path),
        "adapterPath": str(adapter_path) if adapter_path is not None else None,
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
            f"Adapter: {adapter_path}" if adapter_path is not None else "",
            f"Layers: {num_layers}",
            f"Hidden Dim: {hidden_dim}",
            f"Probes: {len(probe_texts)}",
            "",
            "Summary:",
            f"  Mean ID: {mean_id:.1f}",
            f"  Min ID: {min_id:.1f} (layers {highway_layers})" if highway_layers else f"  Min ID: {min_id:.1f}",
            f"  Max ID: {max_id:.1f}",
            f"  Compression: {hidden_dim}D -> {min_id:.1f}D ({(1 - min_id/hidden_dim)*100:.1f}%)" if hidden_dim > 0 else "",
        ]

        if recovery:
            lines.extend([
                "",
                "Recovery Metrics:",
                f"  Final ID: {final_id:.1f}",
                f"  Recovery Ratio: {recovery_ratio:.2f}x (final_ID / min_ID)",
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
            bar = "#" * bar_len + "." * (40 - bar_len)

            # Mark highway layers
            marker = " < HIGHWAY" if layer_idx in highway_layers else ""
            lines.append(f"  Layer {layer_idx:3d}: {id_val:5.1f}D |{bar}|{marker}")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("verification-depth-profile")
def safety_verification_depth_profile(
    ctx: typer.Context,
    model: str | None = typer.Option(None, "--model", help="Path to model directory"),
    levels: str | None = typer.Option(
        None, "--levels", help="Comma-separated verification-depth levels (e.g., 0,1,2,3,4)"
    ),
    mode: VerificationDepthMode = typer.Option(
        VerificationDepthMode.CUMULATIVE,
        "--mode",
        case_sensitive=False,
        help="Grouping mode: cumulative (depth <= level) or exact (depth == level)",
    ),
    max_probes_per_level: int | None = typer.Option(
        None,
        "--max-probes-per-level",
        help="Optional cap on number of probes sampled per verification-depth level",
    ),
    batch_size: int = typer.Option(
        20,
        "--batch-size",
        help="Probe batch size for trajectory collection",
    ),
    probes: str | None = typer.Option(
        None,
        "--probes",
        help="Optional probe JSON file override",
    ),
) -> None:
    """Profile manifold observability across verification-depth levels.

    Analyze-only command: no merge behavior changes, no merge-stage gating.

    Reports raw per-layer measurements for each requested verification-depth level:
    - activation_rank (positions-only)
    - trajectory_rank (positions + velocities)
    - intrinsic_dimension (TwoNN on trajectory samples)
    - condition_number
    - d+1 probe gap and coverage ratios

    Examples:
        mc analyze verification-depth-profile --model ./my-model
        mc analyze verification-depth-profile --model ./my-model --levels 0,1,2,3,4
        mc analyze verification-depth-profile --model ./my-model --mode exact --max-probes-per-level 200
        mc analyze verification-depth-profile --model ./my-model --probes ./data/probes/deep_reasoning.json
    """
    context = get_context(ctx)

    if model is None or not model.strip():
        error = ErrorDetail(
            code="MC-3072",
            title="Missing model path",
            detail="Provide --model with a valid model directory path.",
            hint="Use --model to specify a path to a model directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3072",
            title="Missing model path",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    parsed_levels: list[int] | None = None
    if levels is not None:
        try:
            level_tokens = [token.strip() for token in levels.split(",") if token.strip()]
            if not level_tokens:
                raise ValueError("No levels provided")
            parsed_levels = [int(token) for token in level_tokens]
            if any(level < 0 for level in parsed_levels):
                raise ValueError("Levels must be non-negative integers")
        except ValueError:
            error = ErrorDetail(
                code="MC-3073",
                title="Invalid levels",
                detail=f"Could not parse levels: {levels}",
                hint="Provide comma-separated non-negative integers (e.g., --levels 0,1,2,3,4).",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    probe_inventory = []
    try:
        from modelcypher.core.domain.atlas.probe_loader import (
            load_all_probes,
            load_probes_from_file,
        )

        if probes is not None:
            probes_path = Path(probes)
            if not probes_path.exists():
                error = ErrorDetail(
                    code="MC-3074",
                    title="Probes file not found",
                    detail=f"Probes file does not exist: {probes}",
                    hint="Provide a valid path to a probes JSON file.",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)
            probe_inventory = list(load_probes_from_file(probes_path))
        else:
            probe_inventory = list(load_all_probes())
    except typer.Exit:
        raise
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3076",
            title="Verification-depth profile failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    probes_with_depth = [probe for probe in probe_inventory if probe.verification_depth is not None]
    if not probes_with_depth:
        error = ErrorDetail(
            code="MC-3075",
            title="No verification-depth metadata",
            detail="No probes contain verification-depth metadata (verification_depth or verification_depth_default).",
            hint="Use a probes file that includes verification_depth fields, or use --probes to specify one.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_verification_depth_profile_service

        service = get_verification_depth_profile_service()

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        result = service.profile(
            model=loaded_model,
            tokenizer=tokenizer,
            probes=probe_inventory,
            levels=parsed_levels,
            mode=mode.value,
            max_probes_per_level=max_probes_per_level,
            batch_size=batch_size,
        )

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3076",
            title="Verification-depth profile failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    payload = result.to_dict()
    payload["modelPath"] = str(model_path)
    payload["probesPath"] = probes

    if context.output_format == "text":
        lines = [
            "VERIFICATION-DEPTH PROFILE",
            f"Model: {model_path}",
            f"Mode: {result.mode}",
            f"Levels: {', '.join(str(level) for level in result.levels)}",
            f"Total probes: {result.total_probes}",
            f"Probes with verification depth: {result.probes_with_verification_depth}",
            f"Batch size: {result.batch_size}",
            f"Max probes per level: {result.max_probes_per_level if result.max_probes_per_level is not None else 'none'}",
            "",
            "Plateau Summary:",
            f"  Rank plateau level: {result.rank_plateau_level}",
            f"  ID plateau level: {result.id_plateau_level}",
            f"  Plateau disagreement: {result.plateau_disagreement}",
            "",
        ]

        for level_profile in result.level_profiles:
            lines.extend(
                [
                    f"Level {level_profile.level} ({level_profile.mode}):",
                    f"  probe_count={level_profile.probe_count}",
                    f"  canonical_trajectory_rank={level_profile.canonical_trajectory_rank}",
                    f"  canonical_intrinsic_dimension={level_profile.canonical_intrinsic_dimension}",
                ]
            )
            for layer in level_profile.layer_profiles:
                lines.append(
                    "  "
                    f"L{layer.layer_idx:03d} "
                    f"activation_rank={layer.activation_rank} "
                    f"trajectory_rank={layer.trajectory_rank} "
                    f"intrinsic_dimension={layer.intrinsic_dimension:.6f} "
                    f"condition_number={layer.condition_number:.6f} "
                    f"d_plus_1_gap={layer.d_plus_1_gap} "
                    f"coverage_probe={layer.coverage_ratio_probe:.6f} "
                    f"coverage_trajectory={layer.coverage_ratio_trajectory:.6f}"
                )
            lines.append("")

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
        mc analyze entropy-trajectory --model ./my-model
        mc analyze entropy-trajectory --model ./my-model --layers 0,4,8,12,16
        mc analyze entropy-trajectory --model ./my-model --probes ./probes.txt
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3010",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
                hint="Provide comma-separated integers (e.g., --layers 0,4,8,12,16).",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
            bar = "#" * bar_len + "." * (40 - bar_len)
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
        mc analyze expansion-ratio --model ./my-model --prompt "A bat and ball cost \\$1.10..."
        mc analyze expansion-ratio --model ./my-model --probes prompts.txt --trajectory
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3040",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
            hint="Use --prompt for a single prompt or --probes for a file of prompts.",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
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
            title="Expansion ratio analysis failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
                        bar = "#" * bar_len + "." * (30 - bar_len)

                        # Mark peak layer
                        marker = " < PEAK" if layer_idx == r["peak_layer"] else ""
                        lines.append(f"  Layer {layer_idx:3d}: {id_val:5.1f}D |{bar}|{marker}")

                lines.append("")

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
    - Mean curvature: Average bending of trajectory (Menger kappa)
    - Max curvature: Peak curvature (sharpest turn)
    - Smoothness: 1/(1+mean_curvature) - higher = smoother flow
    - Directness: straight_line_dist/arc_length - higher = more direct

    Use --tokens (-T) for token-level curvature (WHERE in the prompt the reasoning bends).
    Use --trajectory (-t) for layer-level curvature (architectural property).

    Reference: Zhou et al. (2025) arXiv:2510.09782
    GitHub: https://github.com/MasterZhou1/Reasoning-Flow

    Examples:
        mc analyze reasoning-flow --model ./my-model --prompt "What is 2+2?" -T
        mc analyze reasoning-flow --model ./my-model --probes prompts.txt -t
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3060",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
            hint="Use --prompt for a single prompt or --probes for a file of prompts.",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
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
            else:
                peak_curv_layer = -1

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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
                        f"kappa_mean={o['mean_curvature']:.4f} kappa_max={o['max_curvature']:.4f} "
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
                lines.append(f"  Mean curvature (kappa):  {o['mean_curvature']}")
                lines.append(f"  Max curvature (kappa):   {o['max_curvature']} (layer {o['peak_curvature_layer']})")
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
                        bar = "#" * bar_len
                        marker = " < peak" if layer_idx == o["peak_curvature_layer"] else ""
                        lines.append(f"  L{layer_idx:2d}: kappa={curv:.4f} |{bar}{marker}")

                if tokens and r.get("token_curvature"):
                    tc = r["token_curvature"]
                    lines.append("")
                    lines.append("Per-Token Curvature (Zhou et al. methodology):")
                    lines.append(f"  Peak token: [{tc['peak_token_idx']}] '{tc['peak_token']}' (kappa={tc['peak_curvature']:.4f})")
                    lines.append(f"  Mean: {tc['mean_curvature']:.4f}, Std: {tc['std_curvature']:.4f}")
                    lines.append("")
                    if tc.get("tokens"):
                        max_tok_curv = max((t["curvature"] for t in tc["tokens"]), default=1.0)
                        for t in tc["tokens"]:
                            curv = t["curvature"]
                            bar_val = curv / max_tok_curv if max_tok_curv > 0 else 0
                            bar_len = int(bar_val * 30)
                            bar = "#" * bar_len
                            marker = " < PEAK" if t["is_peak"] else ""
                            tok_display = repr(t["token"])[:12].ljust(12)
                            lines.append(f"  [{t['token_idx']:2d}] {tok_display} kappa={curv:.3f} |{bar}{marker}")

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

    - Spectral Entropy = -sum p_i * log(p_i) where p_i = sigma_i^2 / sum(sigma^2)
    - High entropy: Variance spread across many dimensions (expansion)
    - Low entropy: Variance concentrated in few dimensions (compression)

    Expected pattern (MANIFOLD-LEARNING-SYNTHESIS.md):
    - Entry layers: Moderate entropy
    - Middle layers: Peak entropy (maximum exploration/complexity)
    - Exit layers: Reduced entropy (convergence to output)

    Expansion ratio ~ 1.0 indicates flat trajectory (peak ~ final).

    Examples:
        mc analyze spectral-trajectory --model ./my-model
        mc analyze spectral-trajectory --model ./my-model --samples 100
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
            f"  Expansion Ratio: {expansion_ratio}x" if expansion_ratio == expansion_ratio else "  Expansion Ratio: NaN",
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
            bar = "#" * bar_len + "." * (40 - bar_len)

            # Mark peak and trough
            marker = ""
            if layer_idx == peak_idx:
                marker = " < PEAK"
            elif layer_idx == trough_idx:
                marker = " < MIN"

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

    Analyzes the layer-to-layer transformation Jacobian dh_l/dh_{l-1}
    using randomized SVD to estimate singular value spectrum.

    What the Jacobian tells us:
    - sigma_i > 1: Direction i is AMPLIFIED through the layer
    - sigma_i < 1: Direction i is COMPRESSED
    - sigma_i ~ 1: Direction i is PRESERVED

    Key metrics:
    - Effective Rank: How many directions carry information
    - Condition Number: kappa = sigma_max/sigma_min (numerical stability)
    - Spectral Gap: sigma_1/sigma_2 (dominance of first component)
    - Norm Amplification: sigma_max (worst-case growth)

    Hypothesis (from research plan):
    - Reasoning models (DeepSeek-R1) have different Jacobian spectra than
      intuitive models (LFM2)
    - The 1356x norm amplification in reasoning may come from Jacobian structure
    - Bottleneck layers (low effective rank) are information compression points

    Technical note: Uses randomized range finder (Halko et al. 2011) to avoid
    materializing the full [hidden_dim x hidden_dim] Jacobian matrix.

    Examples:
        mc analyze jacobian-trace --model ./my-model --prompt "What is 2+2?"
        mc analyze jacobian-trace --model ./my-model -p "A bat and ball cost \\$1.10..." --trajectory
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3070",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

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
                    bar = "#" * bar_len + "." * (30 - bar_len)

                    # Markers
                    markers = []
                    if p.layer_idx == result.bottleneck_layer:
                        markers.append("BOTTLENECK")
                    if p.layer_idx == result.expansion_layer:
                        markers.append("EXPANSION")
                    marker_str = f" < {', '.join(markers)}" if markers else ""

                    lines.append(f"Layer {p.layer_idx:3d}:")
                    lines.append(f"  sigma_max={p.norm_amplification:.4f} |{bar}|{marker_str}")
                    lines.append(f"  eff_rank={p.effective_rank_shannon:.1f}  kappa={p.condition_number:.1f}  gap={p.spectral_gap:.2f}")

                    # Top singular values (truncated)
                    sv_display = [f"{s:.3f}" for s in p.top_k_singular_values[:min(top_k, 8)]]
                    if len(p.top_k_singular_values) > 8:
                        sv_display.append("...")
                    lines.append(f"  top_sigma: [{', '.join(sv_display)}]")
                    lines.append("")

            write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("concept-volume")
def concept_volume_analysis(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    concepts: str = typer.Option(
        ..., "--concepts", help="Path to concepts JSON file"
    ),
    layer: int = typer.Option(..., "--layer", help="Layer index to analyze"),
) -> None:
    """Analyze concept volumes in activation space using Riemannian density estimation.

    For each named concept (defined by prompts in a JSON file), collects hidden
    activations at the specified layer, estimates a Riemannian ConceptVolume, and
    computes pairwise geometric relations (overlap, distance, curvature divergence).

    Concepts file format (JSON):
        {
          "capital_cities": ["Paris is the capital of France", "Tokyo is ..."],
          "math_operations": ["2+2=4", "The integral of x is x^2/2"]
        }

    Output includes per-concept volume statistics and a pairwise relation table.

    Examples:
        mc analyze concept-volume --model ./my-model --concepts concepts.json --layer 12
    """
    import json as json_mod

    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3080",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    concepts_path = Path(concepts)
    if not concepts_path.exists():
        error = ErrorDetail(
            code="MC-3081",
            title="Concepts file not found",
            detail=f"Concepts file does not exist: {concepts}",
            hint="Provide a valid path to a concepts JSON file.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        concept_data = json_mod.loads(concepts_path.read_text())
    except (json_mod.JSONDecodeError, OSError) as exc:
        error = ErrorDetail(
            code="MC-3082",
            title="Invalid concepts file",
            detail=f"Could not parse concepts JSON: {exc}",
            hint="Ensure the concepts file is valid JSON mapping concept names to prompt lists.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not isinstance(concept_data, dict) or not concept_data:
        error = ErrorDetail(
            code="MC-3083",
            title="Invalid concepts format",
            detail="Concepts file must be a non-empty JSON object mapping concept names to prompt lists.",
            hint='Expected format: {"concept_name": ["prompt1", "prompt2"], ...}.',
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_concept_volume_service

        service = get_concept_volume_service()

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        result = service.analyze_concept_volumes(
            loaded_model, tokenizer, concept_data, layer,
        )

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3084",
            title="Concept volume analysis failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    payload = result.to_dict()
    payload["modelPath"] = str(model_path)
    payload["conceptsPath"] = str(concepts_path)

    if context.output_format == "text":
        lines = [
            "CONCEPT VOLUME ANALYSIS (Riemannian Density)",
            f"Model: {model_path}",
            f"Layer: {layer}",
            f"Concepts: {len(result.concept_stats)}",
            "",
        ]

        if result.concept_stats:
            lines.append("Per-Concept Volumes:")
            for s in result.concept_stats:
                lines.append(f"  {s.concept_id}:")
                lines.append(f"    Samples: {s.num_samples}")
                lines.append(f"    Influence: {s.influence_type}")
                lines.append(f"    Geodesic Radius: {s.geodesic_radius:.6f}")
                lines.append(f"    Effective Radius: {s.effective_radius:.6f}")
                lines.append(f"    Volume: {s.volume:.6e}")
                lines.append(f"    Dimension: {s.dimension}")
                if s.mean_sectional_curvature is not None:
                    lines.append(f"    Mean Curvature: {s.mean_sectional_curvature:.6f}")
            lines.append("")

        if result.pairwise_relations:
            lines.append("Pairwise Relations:")
            for r in result.pairwise_relations:
                lines.append(f"  {r.concept_a} <-> {r.concept_b}:")
                lines.append(f"    Overlap:      {r.overlap_coefficient:.4f}")
                lines.append(f"    Jaccard:      {r.jaccard_index:.4f}")
                lines.append(f"    Bhattacharyya: {r.bhattacharyya_coefficient:.4f}")
                lines.append(f"    Centroid Dist: {r.centroid_distance:.6f}")
                lines.append(f"    Curvature Div: {r.curvature_divergence:.6f}")
                lines.append(f"    Subspace Align: {r.subspace_alignment:.4f}")
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


# Default probes for chain-profile — 6 categories × 10, covering diverse
# semantic domains to sample the representation space uniformly.
_CHAIN_PROFILE_PROBES: tuple[str, ...] = (
    # Factual recall
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at standard pressure.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "DNA carries genetic information in living organisms.",
    "The periodic table organizes elements by atomic number.",
    "Gravity causes objects to fall toward Earth.",
    "The human body contains 206 bones.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Earth revolves around the Sun in approximately 365 days.",
    "Sound travels faster in water than in air.",
    # Logical reasoning
    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins are warm-blooded.",
    "A number that is divisible by 6 must also be divisible by 2 and 3.",
    "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
    "All squares are rectangles, but not all rectangles are squares.",
    "If A implies B, and B implies C, then A implies C.",
    "The contrapositive of 'if P then Q' is 'if not Q then not P'.",
    "A set with n elements has 2^n subsets.",
    "If no fish can fly and all birds can fly, then no fish are birds.",
    "The negation of 'all cats are black' is 'there exists a cat that is not black'.",
    "If the sum of two numbers is even, they are either both even or both odd.",
    # Creative / abstract
    "Imagine a world where colors have sounds and music has texture.",
    "The meaning of life is a question that has puzzled philosophers for millennia.",
    "Art transcends language barriers and speaks to the human condition.",
    "A poem about autumn leaves falling like memories fading.",
    "The boundary between dreaming and waking is thinner than we think.",
    "Music is mathematics made audible, patterns made beautiful.",
    "Time flows like a river, but rivers can be dammed.",
    "The stars are not just lights — they are furnaces forging the elements.",
    "Consciousness might be the universe experiencing itself.",
    "Language shapes thought as much as thought shapes language.",
    # Technical / mathematical
    "The eigenvalues of a symmetric matrix are always real.",
    "Gradient descent minimizes a function by following the negative gradient.",
    "The Fourier transform decomposes a signal into its frequency components.",
    "A neural network with one hidden layer can approximate any continuous function.",
    "The determinant of a matrix is zero if and only if it is singular.",
    "Convolution in the spatial domain equals multiplication in the frequency domain.",
    "The chain rule computes derivatives of composed functions.",
    "Singular value decomposition factors a matrix into rotation and scaling.",
    "Entropy measures the uncertainty in a probability distribution.",
    "The central limit theorem states that sample means converge to normal distribution.",
    # Conversational / social
    "Hello, how are you doing today?",
    "Could you explain this concept in simpler terms?",
    "I disagree with that assessment. Here is why.",
    "Thank you for your help. I appreciate it.",
    "What would you recommend for someone just starting out?",
    "That is an interesting perspective I had not considered before.",
    "Can you summarize the main points of this discussion?",
    "I think we should approach this problem differently.",
    "Let me know if you need any clarification on my question.",
    "What are the potential risks we should be aware of?",
    # Ambiguous / adversarial
    "The bank was steep and the river was deep.",
    "She saw the man with the telescope.",
    "Flying planes can be dangerous.",
    "Time flies like an arrow; fruit flies like a banana.",
    "The old man the boats.",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "I never said she stole my money.",
    "The chicken is ready to eat.",
    "Visiting relatives can be boring.",
    "The horse raced past the barn fell.",
)


@app.command("chain-profile")
def safety_chain_profile(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    probes: str | None = typer.Option(
        None, "--probes", help="Path to file with probe texts (one per line)"
    ),
    samples: int = typer.Option(
        60, "--samples", help="Number of probe samples to use"
    ),
) -> None:
    """Compute unified causal chain profile for a model.

    Measures the validated causal chain at every layer:

        Entropy -> Curvature (angular) -> Cumulative curvature -> ID -> Phase

    Per-layer measurements:
    - Entropy: Shannon entropy via unembedding projection (Entropy-Lens)
    - Curvature: Angular change in radians (arccos of cosine similarity)
    - Attn/MLP decomposition: Fraction of curvature from attention vs MLP
    - Intrinsic dimension: TwoNN estimator (Facco et al., 2017)
    - Phase: highway / processing / exit (data-derived boundaries)

    Cross-link correlations:
    - Entropy <-> Curvature: Spearman r (validated range: 0.4-0.6)
    - Cumulative curvature <-> ID: Spearman r (family-dependent)
    - Mean attention fraction: ~0.37 (universal across architectures)

    For LFM2 hybrid models, non-attention layers (ShortConv/SSM) get total
    curvature only -- attn/MLP decomposition is not available.

    Examples:
        mc analyze chain-profile --model ./my-model
        mc analyze chain-profile --model ./my-model --samples 100
        mc analyze chain-profile --model ./my-model --probes ./probes.txt
    """
    import math

    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3030",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
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
                hint="Provide a valid path to a probes text file (one probe per line).",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        probe_texts = [
            line.strip() for line in probes_path.read_text().splitlines() if line.strip()
        ]

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.cli.composition import get_chain_analysis_service

        service = get_chain_analysis_service()

        # Load model
        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        # Use default probes if not provided
        if probe_texts is None:
            probe_texts = list(_CHAIN_PROFILE_PROBES)

        # Limit to requested samples
        probe_texts = probe_texts[:samples]

        # Compute chain profile
        profile = service.analyze_chain(loaded_model, tokenizer, probe_texts)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3032",
            title="Chain profile analysis failed",
            detail=str(exc),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    payload = profile.as_dict()

    if context.output_format == "text":
        lines = [
            "CAUSAL CHAIN PROFILE",
            f"Model: {model_path}",
            f"Layers: {profile.num_layers}",
            f"Hidden Dim: {profile.hidden_dim}",
            f"Probes: {profile.probe_count}",
            "",
            "Per-Layer Chain:",
            f"  {'Layer':>5} | {'Entropy':>8} | {'Curv (rad)':>10} | {'Attn frac':>9} | {'ID (TwoNN)':>10} | Phase",
            f"  {'-----':>5}-+-{'--------':>8}-+-{'----------':>10}-+-{'---------':>9}-+-{'----------':>10}-+------",
        ]

        for m in profile.layers:
            entropy_str = f"{m.entropy:8.4f}" if m.entropy > 0 else "     N/A"
            curv_str = f"{m.total_curvature:10.4f}"
            attn_str = f"{m.attn_fraction:9.2%}" if m.attn_fraction is not None else "      N/A"
            id_str = f"{m.intrinsic_dimension:10.2f}" if not math.isnan(m.intrinsic_dimension) else "       N/A"
            phase_str = m.phase.value

            lines.append(
                f"  {m.layer_idx:5d} | {entropy_str} | {curv_str} | {attn_str} | {id_str} | {phase_str}"
            )

        lines.extend([
            "",
            "Cross-Link Correlations:",
            f"  Entropy -> Curvature:       Spearman r = {profile.correlations.entropy_to_curvature:.3f}"
            if not math.isnan(profile.correlations.entropy_to_curvature)
            else "  Entropy -> Curvature:       N/A (insufficient data)",
            f"  Cumulative curv -> ID:      Spearman r = {profile.correlations.cumulative_curvature_to_id:.3f}"
            if not math.isnan(profile.correlations.cumulative_curvature_to_id)
            else "  Cumulative curv -> ID:      N/A (insufficient data)",
        ])

        if profile.correlations.mean_attn_fraction is not None:
            lines.append(
                f"  Mean attention fraction:   {profile.correlations.mean_attn_fraction:.3f}"
            )
        else:
            lines.append("  Mean attention fraction:   N/A (no decomposition available)")

        # Phase summary
        phase_counts: dict[str, int] = {}
        for m in profile.layers:
            phase_counts[m.phase.value] = phase_counts.get(m.phase.value, 0) + 1

        lines.extend([
            "",
            "Phase Summary:",
        ])
        for phase_name in ("highway", "processing", "exit"):
            count = phase_counts.get(phase_name, 0)
            if count > 0:
                layer_idxs = [
                    m.layer_idx for m in profile.layers if m.phase.value == phase_name
                ]
                lines.append(f"  {phase_name:>10}: {count} layers ({layer_idxs})")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)


@app.command("attention-collapse")
def safety_attention_collapse(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str = typer.Option(
        "The capital of France is",
        "--prompt",
        help="Text input for attention analysis",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Model dtype for rank-1 threshold (float32, float16, bfloat16)",
    ),
) -> None:
    """Detect attention head collapse via SVD analysis.

    Computes per-head singular value decomposition of attention weight matrices
    to identify rank-1 collapsed heads (gradient-dead) and measure effective rank.

    Metrics per head:
    - Rank-1 ratio: sigma_2 / sigma_1 (collapsed when < sqrt(eps_dtype), IEEE 754)
    - Effective rank: exp(Shannon entropy of normalized sigma^2)  (Roy & Vetterli 2007)
    - Gradient suppression: sigma_2 / sqrt(2T)  (Theorem H.1, Sanyal et al. TMLR 2025)

    Thresholds:
    - bfloat16 (7-bit mantissa): sqrt(2^-7) = 0.0884
    - float16  (10-bit mantissa): sqrt(2^-10) = 0.0312
    - float32  (23-bit mantissa): sqrt(2^-23) = 3.45e-4

    Only softmax attention layers are analyzed. Conv layers (LFM2) and
    linear attention layers (Qwen3.5 GatedDeltaNet) are skipped.

    Examples:
        mc analyze attention-collapse --model ./my-model
        mc analyze attention-collapse --model ./my-model --prompt "Hello world"
        mc analyze attention-collapse --model ./my-model --dtype float32
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3060",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    valid_dtypes = ("float32", "float16", "bfloat16")
    if dtype not in valid_dtypes:
        error = ErrorDetail(
            code="MC-3061",
            title="Invalid dtype",
            detail=f"dtype must be one of {valid_dtypes}, got '{dtype}'",
            hint="Use --dtype float32, --dtype float16, or --dtype bfloat16.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    try:
        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.core.domain.geometry.attention_collapse import (
            _DTYPE_SQRT_EPS,
            compute_attention_collapse,
            compute_collapse_profile,
            summarize_layer_collapse,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        provider = get_activation_provider()
        sqrt_eps = _DTYPE_SQRT_EPS[dtype]

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        attn_matrices = provider.collect_attention_matrices(
            loaded_model, tokenizer, prompt
        )

        layer_results = []
        all_head_results: dict[int, list] = {}

        for layer_idx in sorted(attn_matrices.keys()):
            head_matrices = attn_matrices[layer_idx]
            head_results = []
            for head_mat in head_matrices:
                mat_list = provider._backend.tolist(head_mat)
                result = compute_attention_collapse(mat_list, dtype)
                head_results.append(result)
            all_head_results[layer_idx] = head_results
            layer_summary = summarize_layer_collapse(
                head_results, layer_idx=layer_idx
            )
            layer_results.append(layer_summary)

        profile = compute_collapse_profile(layer_results)

        # Build output payload
        payload = {
            "model": str(model_path),
            "prompt": prompt,
            "dtype": dtype,
            "profile": profile.to_dict(),
            "layers": [lr.to_dict() for lr in layer_results],
        }

        if context.output_format == "text":
            lines = [
                f"Attention Collapse Analysis: {model_path.name}",
                f"Prompt: {prompt!r}",
                f"Dtype: {dtype}  |  Rank-1 threshold: sqrt(eps) = {sqrt_eps:.6f}",
                "",
                f"{'Layer':>5} | {'Collapsed':>9} | {'MaxEffRank':>10} | {'MeanGradSupp':>12}",
                f"{'-----':>5}-+-{'---------':>9}-+-{'----------':>10}-+-{'------------':>12}",
            ]
            for lr in layer_results:
                n_heads = len(all_head_results[lr.layer_idx])
                lines.append(
                    f"{lr.layer_idx:5d} | "
                    f"{lr.collapsed_head_count}/{n_heads:>3}     | "
                    f"{lr.max_effective_rank:10.2f} | "
                    f"{lr.mean_gradient_suppression:12.4f}"
                )
            lines.extend([
                "",
                f"Total layers: {profile.total_layers}  |  "
                f"Layers with any collapse: {profile.collapsed_layer_count}",
            ])
            if profile.collapse_onset_layer is not None:
                lines.append(
                    f"Collapse onset: layer {profile.collapse_onset_layer}"
                )
            else:
                lines.append("Collapse onset: none detected")

            write_output("\n".join(lines), context.output_format, context.pretty)
        else:
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-3062",
            title="Attention collapse analysis failed",
            detail=str(e),
            hint="Check model path and ensure model is loadable.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)


@app.command("attention-sink")
def safety_attention_sink(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompt: str = typer.Option(
        "The capital of France is",
        "--prompt",
        help="Text input for attention analysis",
    ),
) -> None:
    """Analyze attention sink positions and value-weighted active sinks.

    Computes per-head sink scores measuring how much attention each token
    position receives from causal successors, then weights by value-vector
    norms to identify geometrically impactful sinks.

    Metrics per token:
    - Sink score: s_i = (1/(T-i)) * sum_{u>=i} A_{u,i}  (Binkowski et al. 2026)
    - Active sink: sink_score * ||V_i||_2  (geometric impact weighting)

    No tuned thresholds. All measurements are continuous.

    Only softmax attention layers are analyzed. Conv layers (LFM2) and
    linear attention layers (Qwen3.5 GatedDeltaNet) are skipped.

    Examples:
        mc analyze attention-sink --model ./my-model
        mc analyze attention-sink --model ./my-model --prompt "Hello world"
    """
    context = get_context(ctx)

    model_path = Path(model)
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-3063",
            title="Model not found",
            detail=f"Model path does not exist: {model}",
            hint="Ensure the model path points to a valid directory with config.json.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)

    try:
        import math

        from modelcypher.adapters.model_loader import ModelLoader
        from modelcypher.core.domain.geometry.attention_sink import (
            compute_active_sinks,
            compute_sink_scores,
            summarize_layer_sinks,
        )
        from modelcypher.ports.activation_provider import get_activation_provider

        provider = get_activation_provider()

        loader = ModelLoader()
        loaded_model, tokenizer = loader.load_model(str(model_path))

        attn_matrices, value_vectors = provider.collect_attention_matrices_with_values(
            loaded_model, tokenizer, prompt
        )

        layer_results = []
        for layer_idx in sorted(attn_matrices.keys()):
            head_matrices = attn_matrices[layer_idx]
            head_values = value_vectors[layer_idx]

            head_sinks = []
            active_sinks = []

            for head_idx, (head_mat, head_v) in enumerate(
                zip(head_matrices, head_values)
            ):
                mat_list = provider._backend.tolist(head_mat)
                head_sink = compute_sink_scores(mat_list, head_idx=head_idx)
                head_sinks.append(head_sink)

                v_list = provider._backend.tolist(head_v)
                v_norms = [
                    math.sqrt(sum(x * x for x in row)) for row in v_list
                ]
                active = compute_active_sinks(head_sink, v_norms)
                active_sinks.append(active)

            layer_summary = summarize_layer_sinks(
                head_sinks, active_results=active_sinks, layer_idx=layer_idx
            )
            layer_results.append(layer_summary)

        payload = {
            "model": str(model_path),
            "prompt": prompt,
            "layers": [lr.to_dict() for lr in layer_results],
        }

        if context.output_format == "text":
            lines = [
                f"Attention Sink Analysis: {model_path.name}",
                f"Prompt: {prompt!r}",
                "",
                f"{'Layer':>5} | {'DomSinkPos':>10} | {'MeanMaxSink':>11} | {'Heads':>5}",
                f"{'-----':>5}-+-{'----------':>10}-+-{'-----------':>11}-+-{'-----':>5}",
            ]
            for lr in layer_results:
                n_heads = len(lr.head_results)
                lines.append(
                    f"{lr.layer_idx:5d} | "
                    f"{lr.dominant_sink_position:10d} | "
                    f"{lr.mean_max_sink_score:11.4f} | "
                    f"{n_heads:5d}"
                )

            # Summary: how many layers have BOS/early token as dominant
            early_count = sum(
                1 for lr in layer_results if lr.dominant_sink_position <= 1
            )
            total = len(layer_results)
            lines.extend([
                "",
                f"Layers with BOS/early dominant sink: {early_count}/{total}",
            ])

            # Show heads where active differs from raw
            reweighted_count = 0
            for lr in layer_results:
                if lr.active_results:
                    for hs, act in zip(lr.head_results, lr.active_results):
                        if hs.max_sink_position != act.max_active_position:
                            reweighted_count += 1
            total_heads = sum(len(lr.head_results) for lr in layer_results)
            lines.append(
                f"Heads where V-norm reweighting changes max sink: "
                f"{reweighted_count}/{total_heads}"
            )

            write_output("\n".join(lines), context.output_format, context.pretty)
        else:
            write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-3064",
            title="Attention sink analysis failed",
            detail=str(e),
            hint="Check model path and ensure model is loadable.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=EXIT_RUNTIME)
