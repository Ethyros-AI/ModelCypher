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

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.output import write_error, write_output

from .common import cleanup_memory, get_context, load_model_and_provider


def _select_probes(probe_count: int | None, probes: list):
    if probe_count is None:
        return probes
    if probe_count <= 0:
        raise ValueError("probe-count must be positive.")
    if probe_count >= len(probes):
        return probes
    step = max(1, len(probes) // probe_count)
    return probes[::step][:probe_count]


def register(app: typer.Typer) -> None:
    @app.command("manifold-evidence")
    def manifold_evidence(
        ctx: typer.Context,
        model: Path = typer.Argument(..., help="Path to model directory"),
        layer: int | None = typer.Option(
            None, "--layer", help="Layer index (defaults to middle layer)"
        ),
        layers: str | None = typer.Option(
            None, "--layers", help="Comma-separated list of layer indices"
        ),
        all_layers: bool = typer.Option(
            False, "--all-layers", help="Run evidence across all layers"
        ),
        probe_count: int | None = typer.Option(
            None, "--probe-count", help="Optional cap on number of probes"
        ),
        positive_geometry: bool = typer.Option(
            False,
            "--positive-geometry",
            help="Include positive-geometry signatures in the output",
        ),
        positive_geometry_max_minors: int | None = typer.Option(
            None,
            "--positive-geometry-max-minors",
            help="Optional cap on positive-geometry minors evaluated",
        ),
        positive_geometry_rank_source: str = typer.Option(
            "svd",
            "--positive-geometry-rank-source",
            help="Positive-geometry rank selection (svd|spectral-gap|fixed)",
        ),
        positive_geometry_rank: int | None = typer.Option(
            None,
            "--positive-geometry-rank",
            help="Override positive-geometry subspace rank (used with --positive-geometry-rank-source fixed)",
        ),
        positive_geometry_selection: str = typer.Option(
            "lexicographic",
            "--positive-geometry-selection",
            help="Positive-geometry minor selection (lexicographic only)",
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save evidence JSON"
        ),
    ) -> None:
        """Compute manifold evidence metrics from atlas probe activations."""
        context = get_context(ctx)

        if positive_geometry_selection != "lexicographic":
            write_error(
                "positive-geometry-selection must be lexicographic.",
                context.output_format,
            )
            raise typer.Exit(code=1)
        if positive_geometry_max_minors is not None and positive_geometry_max_minors <= 0:
            write_error(
                "positive-geometry-max-minors must be positive.",
                context.output_format,
            )
            raise typer.Exit(code=1)
        if positive_geometry_rank_source not in {"svd", "spectral-gap", "fixed"}:
            write_error(
                "positive-geometry-rank-source must be one of: svd, spectral-gap, fixed.",
                context.output_format,
            )
            raise typer.Exit(code=1)
        if positive_geometry_rank_source != "fixed" and positive_geometry_rank is not None:
            write_error(
                "--positive-geometry-rank is only valid when --positive-geometry-rank-source fixed.",
                context.output_format,
            )
            raise typer.Exit(code=1)
        if positive_geometry_rank_source == "fixed" and positive_geometry_rank is None:
            write_error(
                "--positive-geometry-rank is required when --positive-geometry-rank-source fixed.",
                context.output_format,
            )
            raise typer.Exit(code=1)

        try:
            from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.domain.geometry.manifold_evidence import (
                compute_manifold_evidence,
            )
            from modelcypher.core.domain.geometry.positive_geometry import (
                compute_positive_grassmann_signature,
            )
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.cli.commands.geometry.atlas import AtlasActivationCache
            from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone

            probes = UnifiedAtlasInventory.all_probes()
            if not probes:
                raise ValueError("No atlas probes available.")

            if all_layers and layers:
                raise ValueError("Use either --all-layers or --layers, not both.")
            if layer is not None and (all_layers or layers):
                raise ValueError("Use --layer for a single layer, or --layers/--all-layers.")

            selected = _select_probes(probe_count, probes)
            prompts = []
            for probe in selected:
                if probe.support_texts:
                    prompts.append(probe.support_texts[0])
                else:
                    prompts.append(probe.name)

            if all_layers or layers:
                model_obj, tokenizer = load_model_for_training(str(model))
                resolved = resolve_model_backbone(
                    model_obj, getattr(model_obj, "model_type", None)
                )
                if not resolved:
                    raise ValueError("Could not resolve model architecture.")
                embed_tokens, layers_module, norm = resolved
                num_layers = len(layers_module)
                backend = get_backend()

                if layers:
                    layer_indices = [
                        int(value.strip())
                        for value in layers.split(",")
                        if value.strip()
                    ]
                else:
                    layer_indices = list(range(num_layers))
                if not layer_indices:
                    raise ValueError("No layers specified.")
                for layer_idx in layer_indices:
                    if layer_idx < 0 or layer_idx >= num_layers:
                        raise ValueError(
                            f"Layer {layer_idx} out of range for model with {num_layers} layers."
                        )

                chunk_size = len(layer_indices)
                chunks = [
                    layer_indices[i : i + chunk_size]
                    for i in range(0, len(layer_indices), chunk_size)
                ]

                provider = AtlasActivationCache(
                    tokenizer,
                    embed_tokens,
                    layers_module,
                    norm,
                    backend,
                    frechet_k_neighbors=None,
                    frechet_max_k_neighbors=None,
                    progress_callback=None,
                )

                layer_reports = []
                for chunk in chunks:
                    provider.preload_layers(prompts, chunk)
                    for layer_idx in chunk:
                        activations = provider.get_activations(prompts, layer_idx)
                        arr = backend.array(activations)
                        backend.eval(arr)
                        report = compute_manifold_evidence(arr, backend=backend)
                        positive_signature = None
                        if positive_geometry:
                            positive_signature = compute_positive_grassmann_signature(
                                arr,
                                backend=backend,
                                max_minors=positive_geometry_max_minors,
                                selection=positive_geometry_selection,
                                rank_source=positive_geometry_rank_source,
                                rank_override=positive_geometry_rank,
                            )
                        layer_reports.append(
                            {
                                "layer": layer_idx,
                                "activationCount": len(activations),
                                "evidence": asdict(report),
                                "positiveGeometry": (
                                    positive_signature.to_dict()
                                    if positive_signature is not None
                                    else None
                                ),
                            }
                        )
                    provider.clear_layers(chunk)

                cleanup_memory()

                payload = {
                    "_schema": "mc.geometry.research.manifold_evidence_sweep.v1",
                    "modelPath": str(model),
                    "probeCount": len(selected),
                    "layers": sorted(layer_indices),
                    "layerReports": layer_reports,
                    "positiveGeometryConfig": {
                        "enabled": positive_geometry,
                        "selection": positive_geometry_selection,
                        "maxMinors": positive_geometry_max_minors,
                        "rankSource": positive_geometry_rank_source,
                        "rankOverride": positive_geometry_rank,
                    },
                }
                if output:
                    from modelcypher.utils.json import dump_json

                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(dump_json(payload, pretty=True))

                write_output(payload, context.output_format, context.pretty)
                return

            model_obj, _tokenizer, backend, provider, num_layers = load_model_and_provider(
                str(model)
            )
            if layer is None:
                layer_idx = max(0, num_layers // 2)
            else:
                if layer < 0 or layer >= num_layers:
                    raise ValueError(
                        f"Layer {layer} out of range for model with {num_layers} layers."
                    )
                layer_idx = layer

            activations = provider.get_activations(prompts, layer_idx)
            if len(activations) != len(prompts):
                raise ValueError(
                    "Activation collection returned mismatched sample counts. "
                    "Reduce probe count or inspect probes."
                )

            arr = backend.array(activations)
            backend.eval(arr)

            report = compute_manifold_evidence(arr, backend=backend)
            positive_signature = None
            if positive_geometry:
                positive_signature = compute_positive_grassmann_signature(
                    arr,
                    backend=backend,
                    max_minors=positive_geometry_max_minors,
                    selection=positive_geometry_selection,
                    rank_source=positive_geometry_rank_source,
                    rank_override=positive_geometry_rank,
                )
            cleanup_memory()

            payload = {
                "_schema": "mc.geometry.research.manifold_evidence.v1",
                "modelPath": str(model),
                "layer": layer_idx,
                "probeCount": len(selected),
                "activationCount": len(activations),
                "evidence": asdict(report),
                "positiveGeometry": (
                    positive_signature.to_dict() if positive_signature is not None else None
                ),
                "positiveGeometryConfig": {
                    "enabled": positive_geometry,
                    "selection": positive_geometry_selection,
                    "maxMinors": positive_geometry_max_minors,
                    "rankSource": positive_geometry_rank_source,
                    "rankOverride": positive_geometry_rank,
                },
            }

            if output:
                from modelcypher.utils.json import dump_json

                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(dump_json(payload, pretty=True))

            if context.output_format == "text":
                lines = [
                    "",
                    "=" * 60,
                    "MANIFOLD EVIDENCE REPORT",
                    "=" * 60,
                    f"Model: {model}",
                    f"Layer: {layer_idx}",
                    f"Samples: {report.sample_count}",
                    f"Intrinsic dimension: {report.intrinsic_dimension}",
                    f"Effective rank (Renyi): {report.effective_rank.renyi_effective_rank:.4f}",
                    f"Effective rank (Shannon): {report.effective_rank.shannon_effective_rank:.4f}",
                    f"Support ratio (Renyi): {report.support_diagnostics.renyi_support_ratio:.4f}",
                    f"Support ratio (Shannon): {report.support_diagnostics.shannon_support_ratio:.4f}",
                    f"Null ratio (Renyi): {report.support_diagnostics.renyi_null_ratio:.4f}",
                    f"Null ratio (Shannon): {report.support_diagnostics.shannon_null_ratio:.4f}",
                ]
                if report.support_diagnostics.renyi_id_gap is not None:
                    lines.append(
                        f"ID gap (Renyi): {report.support_diagnostics.renyi_id_gap:.4f}"
                    )
                if report.support_diagnostics.shannon_id_gap is not None:
                    lines.append(
                        f"ID gap (Shannon): {report.support_diagnostics.shannon_id_gap:.4f}"
                    )
                if report.tangent_rank is not None:
                    lines.append(
                        f"Tangent rank (Renyi): {report.tangent_rank.renyi_effective_rank:.4f}"
                    )
                    lines.append(
                        f"Tangent rank (Shannon): {report.tangent_rank.shannon_effective_rank:.4f}"
                    )
                if report.curvature is not None:
                    lines.append("")
                    lines.append("Curvature:")
                    lines.append(f"  Mean sectional: {report.curvature.mean_sectional:.6f}")
                    lines.append(f"  Min sectional: {report.curvature.min_sectional:.6f}")
                    lines.append(f"  Max sectional: {report.curvature.max_sectional:.6f}")
                    lines.append(f"  Dominant sign: {report.curvature.dominant_sign}")
                if positive_signature is not None:
                    lines.append("")
                    lines.append("Positive geometry:")
                    lines.append(f"  Rank: {positive_signature.subspace_rank}")
                    lines.append(f"  Positive fraction: {positive_signature.positive_fraction:.6f}")
                    lines.append(f"  Negative fraction: {positive_signature.negative_fraction:.6f}")
                    lines.append(f"  Zero fraction: {positive_signature.zero_fraction:.6f}")
                    lines.append(f"  Sign entropy: {positive_signature.sign_entropy:.6f}")
                    lines.append(f"  Plucker norm: {positive_signature.plucker_norm:.6f}")
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Manifold evidence failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
