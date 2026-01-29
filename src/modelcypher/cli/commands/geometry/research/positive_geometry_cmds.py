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

import json
import logging
from pathlib import Path
from random import Random
from typing import Iterable

import typer

from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.domains import resolve_domains
from modelcypher.core.domain.geometry.positive_geometry import (
    compute_positive_grassmann_signature,
)

from .common import cleanup_memory, get_context

logger = logging.getLogger(__name__)


def _select_probes(probe_count: int | None, probes: list):
    if probe_count is None:
        return probes
    if probe_count <= 0:
        raise ValueError("probe-count must be positive.")
    if probe_count >= len(probes):
        return probes
    step = max(1, len(probes) // probe_count)
    return probes[::step][:probe_count]


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _hash_probe_order(probes: Iterable) -> str:
    import hashlib

    ids = ",".join([getattr(p, "probe_id", str(p)) for p in probes])
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def _resolve_layers(
    model_obj,
    layers_option: str | None,
    layer_option: int | None,
    all_layers: bool,
):
    from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone

    resolved = resolve_model_backbone(model_obj, getattr(model_obj, "model_type", None))
    if not resolved:
        raise ValueError("Could not resolve model architecture.")
    embed_tokens, layers_module, norm = resolved
    num_layers = len(layers_module)

    if all_layers and layers_option:
        raise ValueError("Use either --all-layers or --layers, not both.")
    if layer_option is not None and (all_layers or layers_option):
        raise ValueError("Use --layer for a single layer, or --layers/--all-layers.")

    if layers_option:
        layer_indices = [
            int(value.strip())
            for value in layers_option.split(",")
            if value.strip()
        ]
    elif all_layers:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [max(0, num_layers // 2) if layer_option is None else layer_option]

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} out of range for model with {num_layers} layers.")

    return embed_tokens, layers_module, norm, num_layers, layer_indices


def register(app: typer.Typer) -> None:
    @app.command("positive-geometry")
    def positive_geometry(
        ctx: typer.Context,
        model: Path = typer.Argument(..., help="Path to model directory"),
        layer: int | None = typer.Option(
            None, "--layer", help="Layer index (defaults to middle layer)"
        ),
        layers: str | None = typer.Option(
            None, "--layers", help="Comma-separated list of layer indices"
        ),
        all_layers: bool = typer.Option(
            False, "--all-layers", help="Run signature across all layers"
        ),
        probe_count: int | None = typer.Option(
            None, "--probe-count", help="Optional cap on number of probes"
        ),
        domains: str | None = typer.Option(
            None, "--domains", help="Comma-separated atlas domains to include"
        ),
        adapter: str | None = typer.Option(
            None, "--adapter", help="Optional LoRA adapter directory to load"
        ),
        max_minors: int | None = typer.Option(
            None, "--max-minors", help="Optional cap on minors evaluated"
        ),
        rank_source: str = typer.Option(
            "svd",
            "--rank-source",
            help="Rank selection (svd|spectral-gap|fixed)",
        ),
        rank: int | None = typer.Option(
            None,
            "--rank",
            help="Override subspace rank (used when --rank-source fixed)",
        ),
        shuffle_seed: int | None = typer.Option(
            None,
            "--shuffle-seed",
            help="Seed for probe order shuffling",
        ),
        shuffle_count: int = typer.Option(
            1,
            "--shuffle-count",
            help="Number of shuffled probe orders to evaluate",
        ),
        selection: str = typer.Option(
            "lexicographic",
            "--selection",
            help="Minor selection strategy (lexicographic only)",
        ),
        output: Path | None = typer.Option(
            None, "--output-file", help="Path to save signature JSON"
        ),
    ) -> None:
        """Compute positive-geometry signatures from atlas probe activations."""
        context = get_context(ctx)

        if selection != "lexicographic":
            write_error("Only lexicographic selection is supported.", context.output_format)
            raise typer.Exit(code=1)
        if max_minors is not None and max_minors <= 0:
            write_error("max-minors must be positive.", context.output_format)
            raise typer.Exit(code=1)
        if rank_source not in {"svd", "spectral-gap", "fixed"}:
            write_error("rank-source must be one of: svd, spectral-gap, fixed.", context.output_format)
            raise typer.Exit(code=1)
        if rank_source != "fixed" and rank is not None:
            write_error("--rank is only valid when --rank-source fixed.", context.output_format)
            raise typer.Exit(code=1)
        if shuffle_count <= 0:
            write_error("shuffle-count must be positive.", context.output_format)
            raise typer.Exit(code=1)

        try:
            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.cli.commands.geometry.atlas import AtlasActivationCache
            from modelcypher.core.domain._backend import get_default_backend

            if domains:
                domain_list = resolve_domains(_split_csv(domains))
                if not domain_list:
                    raise ValueError("No valid domains resolved from --domains.")
                probes = UnifiedAtlasInventory.probes_by_domain(set(domain_list))
            else:
                probes = UnifiedAtlasInventory.all_probes()
            if not probes:
                raise ValueError("No atlas probes available.")

            selected_base = _select_probes(probe_count, probes)

            model_obj, tokenizer = load_model_for_training(str(model), adapter_path=adapter)
            embed_tokens, layers_module, norm, num_layers, layer_indices = _resolve_layers(
                model_obj,
                layers,
                layer,
                all_layers,
            )

            backend = get_default_backend()
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

            sweeps = []
            do_shuffle = shuffle_seed is not None or shuffle_count > 1
            seed_value = shuffle_seed if shuffle_seed is not None else 0
            rng = Random(seed_value) if do_shuffle else None
            for sweep_idx in range(shuffle_count):
                selected = list(selected_base)
                if do_shuffle and rng is not None:
                    rng.shuffle(selected)

                prompts = []
                for probe in selected:
                    if probe.support_texts:
                        prompts.append(probe.support_texts[0])
                    else:
                        prompts.append(probe.name)

                layer_reports = []
                chunks = [layer_indices]
                for chunk in chunks:
                    provider.preload_layers(prompts, chunk)
                    for layer_idx in chunk:
                        activations = provider.get_activations(prompts, layer_idx)
                        if not activations:
                            raise ValueError(f"No activations collected for layer {layer_idx}.")
                        arr = backend.array(activations)
                        backend.eval(arr)
                        signature = compute_positive_grassmann_signature(
                            arr,
                            backend=backend,
                            max_minors=max_minors,
                            selection=selection,
                            rank_source="svd" if rank_source == "fixed" else rank_source,
                            rank_override=rank if rank_source == "fixed" else None,
                        )
                        layer_reports.append(
                            {
                                "layer": layer_idx,
                                "activationCount": len(activations),
                                "signature": signature.to_dict(),
                            }
                        )
                    provider.clear_layers(chunk)

                sweeps.append(
                    {
                        "shuffleIndex": sweep_idx,
                        "shuffleSeed": shuffle_seed,
                        "probeOrderHash": _hash_probe_order(selected),
                        "layerReports": layer_reports,
                    }
                )

            cleanup_memory()

            payload = {
                "_schema": "mc.geometry.research.positive_geometry.v1",
                "modelPath": str(model),
                "adapterPath": adapter,
                "probeCount": len(selected_base),
                "domains": _split_csv(domains) if domains else None,
                "maxMinors": max_minors,
                "selection": selection,
                "rankSource": rank_source,
                "rankOverride": rank,
                "layers": sorted(layer_indices),
                "shuffleSeed": shuffle_seed,
                "shuffleCount": shuffle_count,
                "effectiveShuffleSeed": seed_value if do_shuffle else None,
                "sweeps": sweeps,
                "totalLayers": num_layers,
            }

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload, indent=2 if context.pretty else None))

            if context.output_format == "text":
                lines = [
                    "POSITIVE GEOMETRY SIGNATURE",
                    f"Model: {model}",
                    f"Adapter: {adapter if adapter else 'none'}",
                    f"Probes: {len(selected_base)}",
                    f"Domains: {domains if domains else 'all'}",
                    f"Layers: {', '.join(str(idx) for idx in sorted(layer_indices))}",
                    f"Max minors: {max_minors if max_minors is not None else 'all'}",
                    f"Rank source: {rank_source}",
                    f"Rank override: {rank if rank is not None else 'none'}",
                    f"Shuffle count: {shuffle_count}",
                    f"Shuffle seed: {seed_value if do_shuffle else 'none'}",
                    "",
                ]
                for sweep in sweeps:
                    lines.append(f"Shuffle {sweep['shuffleIndex']} (hash={sweep['probeOrderHash']})")
                    for report in sweep["layerReports"]:
                        sig = report["signature"]
                        lines.extend(
                            [
                                f"Layer {report['layer']}",
                                f"  Rank: {sig['subspaceRank']}",
                                f"  Evaluated minors: {sig['evaluatedMinors']} / {sig['totalMinors']}",
                                f"  Positive/Negative/Zero: {sig['positiveFraction']:.6f} / {sig['negativeFraction']:.6f} / {sig['zeroFraction']:.6f}",
                                f"  Sign entropy: {sig['signEntropy']:.6f}",
                                f"  Min/Max minor: {sig['minMinor']:.6e} / {sig['maxMinor']:.6e}",
                                f"  Mean |minor|: {sig['meanAbsMinor']:.6e}",
                                "",
                            ]
                        )

                write_output("\n".join(lines), context.output_format, context.pretty)
                return

            write_output(payload, context.output_format, context.pretty)
        except Exception as exc:
            write_error(str(exc), context.output_format)
            raise typer.Exit(code=1) from exc
