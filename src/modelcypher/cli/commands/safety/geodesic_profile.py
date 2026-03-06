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

"""Geodesic layer profile CLI command.

Experimental. Measures per-layer geodesic deviation profile.

    mc safety geodesic-profile --model PATH --prompt TEXT
    mc safety geodesic-profile --model PATH --prompts FILE
"""

from __future__ import annotations

import json

from ._common import (
    ErrorDetail,
    Path,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


@app.command("geodesic-profile")
def geodesic_profile(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", help="Path to model directory"
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", "-p", help="Single prompt to profile"
    ),
    prompts: str | None = typer.Option(
        None, "--prompts", help="JSON file with categorized prompts"
    ),
    adapter: str | None = typer.Option(
        None, "--adapter", help="Path to adapter directory. Compares baseline vs adapted."
    ),
) -> None:
    """Profile geodesic deviation across all layers.

    Experimental. Collects trajectory once, then measures geodesic deviation
    at every layer to show WHERE deviation emerges in the layer stack.

    Single prompt mode: per-layer deviation with bar chart visualization.
    Batch mode (--prompts): per-category per-layer averages with divergence onset.
    Comparison mode (--adapter): per-layer delta between baseline and adapted model.

    Example:
        mc safety geodesic-profile --model /path/to/model --prompt "What is 2+2? Let me think..."
        mc safety geodesic-profile --model /path/to/model --prompts data/probes/geodesic_comparison.json
        mc safety geodesic-profile --model /path/to/model --adapter /path/to/adapter --prompt "..."
    """
    context = get_context(ctx)
    model_path = Path(model)

    if not model_path.exists():
        error = ErrorDetail(
            code="MC-8003",
            title="Model not found",
            detail=f"Path does not exist: {model_path}",
            hint="Provide a valid path to a model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if adapter is not None and not Path(adapter).exists():
        error = ErrorDetail(
            code="MC-8006",
            title="Adapter not found",
            detail=f"Path does not exist: {adapter}",
            hint="Provide a valid path to an adapter directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if prompt is None and prompts is None:
        error = ErrorDetail(
            code="MC-8005",
            title="No input provided",
            detail="Provide --prompt or --prompts",
            hint="Use --prompt for single text or --prompts for categorized JSON file",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    from modelcypher.cli.composition import (
        get_activation_provider,
        get_backend,
    )
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.core.use_cases.geodesic_trajectory_service import (
        GeodesicTrajectoryService,
    )

    backend = get_backend()
    loader = ModelLoader(backend)
    service = GeodesicTrajectoryService(
        backend=backend,
        activation_provider=get_activation_provider(),
    )

    if adapter is not None:
        # Comparison mode: load baseline and adapted models
        baseline_obj, tokenizer_obj = loader.load_model(str(model_path))
        adapted_obj, _ = loader.load_model(str(model_path), adapter_path=adapter)

        if prompts is not None:
            _run_comparison_batch(
                context, service, baseline_obj, adapted_obj, tokenizer_obj,
                model_path, adapter, prompts,
            )
        else:
            assert prompt is not None
            _run_comparison_single(
                context, service, baseline_obj, adapted_obj, tokenizer_obj,
                model_path, adapter, prompt,
            )
    else:
        # Standard mode: single model
        model_obj, tokenizer_obj = loader.load_model(str(model_path))

        if prompts is not None:
            _run_batch(context, service, model_obj, tokenizer_obj, model_path, prompts)
        else:
            assert prompt is not None
            _run_single(context, service, model_obj, tokenizer_obj, model_path, prompt)


def _run_single(context, service, model_obj, tokenizer_obj, model_path, prompt):
    """Run single-prompt layer profile."""
    result = service.measure_layer_profile(
        model=model_obj,
        tokenizer=tokenizer_obj,
        text=prompt,
    )

    if context.output_format == "text":
        lines = [
            "GEODESIC LAYER PROFILE",
            f"Model: {model_path.name}",
            f"Tokens: {result.token_count}",
            f"Layers: {result.num_layers}",
            "",
        ]

        if result.peak_deviation_layer is not None:
            lines.append(f"Peak deviation: layer {result.peak_deviation_layer}")
        if result.inflection_layer is not None:
            lines.append(
                f"Onset (Donoho-Johnstone universal threshold): layer {result.inflection_layer}"
            )
        lines.append("")

        # Find max deviation for normalization
        max_dev = max(
            (lp.mean_deviation for lp in result.layer_profiles), default=1.0
        )
        if max_dev == 0:
            max_dev = 1.0

        lines.append(
            f"{'Layer':>5}  {'mean_dev':>8}  {'path_ratio':>10}  {'ID':>5}  "
        )

        for lp in result.layer_profiles:
            bar_val = min(1.0, lp.mean_deviation / max_dev)
            bar_len = int(bar_val * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)

            marker = ""
            if lp.layer == result.peak_deviation_layer:
                marker = " < PEAK"
            elif lp.layer == result.inflection_layer:
                marker = " < ONSET"

            lines.append(
                f"  {lp.layer:3d}  {lp.mean_deviation:8.4f}  "
                f"{lp.path_length_ratio:10.2f}  {lp.intrinsic_dimension:5.1f}  "
                f"|{bar}|{marker}"
            )

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(result.to_dict(), context.output_format, context.pretty)


def _run_batch(context, service, model_obj, tokenizer_obj, model_path, prompts_path_str):
    """Run batch layer profile across categorized prompts."""
    prompts_path = Path(prompts_path_str)

    if not prompts_path.exists():
        error = ErrorDetail(
            code="MC-8004",
            title="Prompts file not found",
            detail=f"Path does not exist: {prompts_path}",
            hint="Provide a JSON file with categorized prompts",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    with open(prompts_path) as f:
        categorized_prompts = json.load(f)

    result = service.measure_layer_profile_batch(
        model=model_obj,
        tokenizer=tokenizer_obj,
        categorized_prompts=categorized_prompts,
        model_path=str(model_path),
    )

    if context.output_format == "text":
        total_prompts = sum(cp.prompt_count for cp in result.category_profiles)
        lines = [
            "GEODESIC LAYER PROFILE (Batch)",
            f"Model: {model_path.name}",
            f"Categories: {len(result.category_profiles)}, Prompts: {total_prompts}",
            f"Layers: {result.num_layers}",
            "",
        ]

        if result.divergence_onset_layer is not None:
            lines.append(
                f"Divergence onset: layer {result.divergence_onset_layer} "
                f"(inter-category spread exceeds MAD-relative threshold)"
            )
            lines.append("")

        for cp in result.category_profiles:
            lines.append(f"Category: {cp.category} (n={cp.prompt_count})")

            max_dev = max(
                (lp.mean_deviation for lp in cp.layer_profiles), default=1.0
            )
            if max_dev == 0:
                max_dev = 1.0

            for lp in cp.layer_profiles:
                bar_val = min(1.0, lp.mean_deviation / max_dev)
                bar_len = int(bar_val * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)

                marker = ""
                if lp.layer == cp.peak_deviation_layer:
                    marker = " < PEAK"
                elif lp.layer == cp.inflection_layer:
                    marker = " < ONSET"

                lines.append(
                    f"  {lp.layer:3d}  {lp.mean_deviation:8.4f}  |{bar}|{marker}"
                )
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(result.to_dict(), context.output_format, context.pretty)


def _run_comparison_single(
    context, service, baseline_obj, adapted_obj, tokenizer_obj,
    model_path, adapter_path, prompt,
):
    """Run single-prompt comparison between baseline and adapted model."""
    result = service.compare_layer_profiles(
        baseline_model=baseline_obj,
        adapted_model=adapted_obj,
        tokenizer=tokenizer_obj,
        text=prompt,
        model_path=str(model_path),
        adapter_path=adapter_path,
    )

    if context.output_format == "text":
        lines = [
            "GEODESIC LAYER COMPARISON",
            f"Model: {model_path.name}",
            f"Adapter: {Path(adapter_path).name}",
            f"Tokens: {result.token_count}",
            f"Layers: {result.num_layers}",
            "",
            f"Max divergence: layer {result.max_divergence_layer} "
            f"(|delta| = {result.max_divergence_magnitude:.4f})",
            "",
            f"{'Layer':>5}  {'base_dev':>8}  {'adapt_dev':>9}  {'delta':>8}  ",
        ]

        max_abs_delta = max((abs(d) for d in result.delta_deviations), default=1.0)
        if max_abs_delta == 0:
            max_abs_delta = 1.0

        adapted_by_layer = {lp.layer: lp for lp in result.adapted_profiles}

        for i, lp in enumerate(result.baseline_profiles):
            adapted_lp = adapted_by_layer.get(lp.layer)
            adapt_dev = adapted_lp.mean_deviation if adapted_lp else 0.0
            delta = result.delta_deviations[i] if i < len(result.delta_deviations) else 0.0

            bar_val = min(1.0, abs(delta) / max_abs_delta)
            bar_len = int(bar_val * 20)
            if delta >= 0:
                bar = "+" * bar_len + "." * (20 - bar_len)
            else:
                bar = "-" * bar_len + "." * (20 - bar_len)

            marker = ""
            if lp.layer == result.max_divergence_layer:
                marker = " < MAX DIVERGENCE"

            lines.append(
                f"  {lp.layer:3d}  {lp.mean_deviation:8.4f}  {adapt_dev:9.4f}  "
                f"{delta:+8.4f}  |{bar}|{marker}"
            )

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(result.to_dict(), context.output_format, context.pretty)


def _run_comparison_batch(
    context, service, baseline_obj, adapted_obj, tokenizer_obj,
    model_path, adapter_path, prompts_path_str,
):
    """Run batch comparison between baseline and adapted model."""
    prompts_path = Path(prompts_path_str)

    if not prompts_path.exists():
        error = ErrorDetail(
            code="MC-8004",
            title="Prompts file not found",
            detail=f"Path does not exist: {prompts_path}",
            hint="Provide a JSON file with categorized prompts",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    with open(prompts_path) as f:
        categorized_prompts = json.load(f)

    result = service.compare_layer_profiles_batch(
        baseline_model=baseline_obj,
        adapted_model=adapted_obj,
        tokenizer=tokenizer_obj,
        categorized_prompts=categorized_prompts,
        model_path=str(model_path),
        adapter_path=adapter_path,
    )

    if context.output_format == "text":
        lines = [
            "GEODESIC LAYER COMPARISON (Batch)",
            f"Model: {model_path.name}",
            f"Adapter: {Path(adapter_path).name}",
            f"Layers: {result.num_layers}",
            "",
            f"Max divergence: layer {result.max_divergence_layer} "
            f"(|delta| = {result.max_divergence_magnitude:.4f})",
            "",
            "Per-layer mean delta (adapted - baseline):",
        ]

        max_abs = max((abs(d) for _, d in result.mean_delta_by_layer), default=1.0)
        if max_abs == 0:
            max_abs = 1.0

        for layer_idx, delta in result.mean_delta_by_layer:
            bar_val = min(1.0, abs(delta) / max_abs)
            bar_len = int(bar_val * 20)
            if delta >= 0:
                bar = "+" * bar_len + "." * (20 - bar_len)
            else:
                bar = "-" * bar_len + "." * (20 - bar_len)

            marker = ""
            if layer_idx == result.max_divergence_layer:
                marker = " < MAX"

            lines.append(f"  {layer_idx:3d}  {delta:+8.4f}  |{bar}|{marker}")

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(result.to_dict(), context.output_format, context.pretty)
