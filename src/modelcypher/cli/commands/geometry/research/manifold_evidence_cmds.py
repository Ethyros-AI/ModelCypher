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
        batch_size: int = typer.Option(
            8, "--batch-size", help="Batch size for activation collection"
        ),
        pooling: str = typer.Option(
            "mean", "--pooling", help="Token pooling: mean or frechet"
        ),
        layer_chunk_size: int | None = typer.Option(
            None, "--layer-chunk-size", help="Layers per activation pass"
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save evidence JSON"
        ),
    ) -> None:
        """Compute manifold evidence metrics from atlas probe activations."""
        context = get_context(ctx)

        try:
            from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
            from modelcypher.core.domain.geometry.manifold_evidence import (
                compute_manifold_evidence,
            )
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.cli.commands.geometry.atlas import AtlasActivationCache
            from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone
            from modelcypher.core.domain._backend import get_default_backend

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

            pool_mode = pooling.strip().lower()
            if pool_mode not in {"mean", "frechet"}:
                raise ValueError("Pooling must be 'mean' or 'frechet'.")
            if batch_size < 1:
                raise ValueError("batch-size must be >= 1.")

            if all_layers or layers:
                model_obj, tokenizer = load_model_for_training(str(model))
                resolved = resolve_model_backbone(
                    model_obj, getattr(model_obj, "model_type", None)
                )
                if not resolved:
                    raise ValueError("Could not resolve model architecture.")
                embed_tokens, layers_module, norm = resolved
                num_layers = len(layers_module)
                backend = get_default_backend()

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

                chunk_size = (
                    len(layer_indices)
                    if layer_chunk_size is None or layer_chunk_size <= 0
                    else min(layer_chunk_size, len(layer_indices))
                )
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
                    pooling=pool_mode,
                    batch_size=batch_size,
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
                        layer_reports.append(
                            {
                                "layer": layer_idx,
                                "activationCount": len(activations),
                                "evidence": asdict(report),
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
            cleanup_memory()

            payload = {
                "_schema": "mc.geometry.research.manifold_evidence.v1",
                "modelPath": str(model),
                "layer": layer_idx,
                "probeCount": len(selected),
                "activationCount": len(activations),
                "evidence": asdict(report),
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
                ]
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
                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)

        except Exception as exc:
            write_error(f"Manifold evidence failed: {exc}", context.output_format)
            raise typer.Exit(1) from exc
