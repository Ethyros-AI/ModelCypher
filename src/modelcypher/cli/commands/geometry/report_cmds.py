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
from modelcypher.core.domain.domains import AtlasDomain, resolve_domain


app = typer.Typer(no_args_is_help=True, help="Consolidated geometry reports")


def _context(ctx: typer.Context):
    return ctx.obj


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_domains_csv(value: str | None) -> list[AtlasDomain]:
    names = _split_csv(value)
    resolved: list[AtlasDomain] = []
    unknown: list[str] = []
    for name in names:
        domain = resolve_domain(name)
        if domain is None:
            unknown.append(name)
            continue
        if domain not in resolved:
            resolved.append(domain)
    if unknown:
        raise ValueError(f"Unknown domains: {', '.join(sorted(unknown))}")
    return resolved


def _select_probes(probe_count: int | None, probes: list):
    if probe_count is None:
        return probes
    if probe_count <= 0:
        raise ValueError("probe-count must be positive.")
    if probe_count >= len(probes):
        return probes
    step = max(1, len(probes) // probe_count)
    return probes[::step][:probe_count]


def _resolve_layers(
    model_obj,
    layer_option: int | None,
    layers_option: str | None,
    all_layers: bool,
):
    from modelcypher.cli.commands.geometry.helpers import resolve_model_backbone

    resolved = resolve_model_backbone(model_obj, getattr(model_obj, "model_type", None))
    if not resolved:
        raise ValueError("Could not resolve model architecture.")
    embed_tokens, layers_module, norm = resolved
    num_layers = len(layers_module)

    if layer_option is not None and (all_layers or layers_option):
        raise ValueError("Use --layer for a single layer, or --layers/--all-layers.")

    if layers_option:
        layer_indices = []
        for value in layers_option.split(","):
            value = value.strip()
            if not value:
                continue
            try:
                layer_indices.append(int(value))
            except ValueError as exc:
                raise ValueError(f"Invalid layer index: {value}") from exc
    elif all_layers:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [max(0, num_layers // 2) if layer_option is None else layer_option]

    if not layer_indices:
        raise ValueError("No layers specified.")
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer {layer_idx} out of range for model with {num_layers} layers.")

    return embed_tokens, layers_module, norm, num_layers, sorted(layer_indices)


def _prompts_from_probes(probes: list) -> list[str]:
    prompts: list[str] = []
    for probe in probes:
        if probe.support_texts:
            prompts.append(probe.support_texts[0])
        else:
            prompts.append(probe.name)
    return prompts


def _summarize_layer_report(layer_report: dict) -> dict:
    evidence = layer_report.get("manifoldEvidence", {}) or {}
    effective_rank = evidence.get("effective_rank", {}) or {}
    curvature = evidence.get("curvature", {}) or {}
    positive = layer_report.get("positiveGeometry") or {}
    return {
        "layer": layer_report.get("layer"),
        "activationCount": layer_report.get("activationCount"),
        "intrinsicDimension": evidence.get("intrinsic_dimension"),
        "effectiveRankShannon": effective_rank.get("shannon_effective_rank"),
        "spectralEntropy": effective_rank.get("spectral_entropy"),
        "frechetVariance": evidence.get("frechet_variance"),
        "curvatureMeanSectional": curvature.get("mean_sectional"),
        "positiveGeometry": {
            "rank": positive.get("subspaceRank"),
            "posFraction": positive.get("positiveFraction"),
            "signEntropy": positive.get("signEntropy"),
            "zeroFraction": positive.get("zeroFraction"),
        } if positive else None,
    }


def _summarize_domain_report(domain_report: dict) -> dict:
    layer_reports = domain_report.get("layerReports") or []
    return {
        "probeCount": domain_report.get("probeCount"),
        "layerReports": [
            {
                "layer": report.get("layer"),
                "activationCount": report.get("activationCount"),
                "positiveGeometry": {
                    "rank": (report.get("positiveGeometry") or {}).get("subspaceRank"),
                    "posFraction": (report.get("positiveGeometry") or {}).get("positiveFraction"),
                    "signEntropy": (report.get("positiveGeometry") or {}).get("signEntropy"),
                    "zeroFraction": (report.get("positiveGeometry") or {}).get("zeroFraction"),
                },
            }
            for report in layer_reports
        ],
    }


@app.command("model")
def geometry_report_model(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to model directory"),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
    layers: str | None = typer.Option(
        None, "--layers", help="Comma-separated list of layer indices"
    ),
    all_layers: bool = typer.Option(
        False, "--all-layers", help="Run report across all layers"
    ),
    probe_count: int | None = typer.Option(
        None, "--probe-count", help="Optional cap on number of probes"
    ),
    adapter: str | None = typer.Option(
        None, "--adapter", help="Optional LoRA adapter directory to load"
    ),
    base_domains: str | None = typer.Option(
        None, "--base-domains", help="Comma-separated atlas domains for base report probes"
    ),
    domain_fingerprints: bool = typer.Option(
        False, "--domain-fingerprints", help="Include per-domain positive-geometry fingerprints"
    ),
    fingerprint_domains: str | None = typer.Option(
        None,
        "--fingerprint-domains",
        help="Comma-separated atlas domains to fingerprint (defaults to all domains)",
    ),
    positive_geometry: bool = typer.Option(
        True,
        "--positive-geometry/--no-positive-geometry",
        help="Include positive-geometry signatures in the base report",
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
        None, "--output-file", help="Path to save consolidated report JSON"
    ),
) -> None:
    """Generate a consolidated geometry report for a model."""
    context = _context(ctx)

    if positive_geometry_selection != "lexicographic":
        write_error("positive-geometry-selection must be lexicographic.", context.output_format)
        raise typer.Exit(code=1)
    if positive_geometry_max_minors is not None and positive_geometry_max_minors <= 0:
        write_error("positive-geometry-max-minors must be positive.", context.output_format)
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
        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.cli.commands.geometry.atlas import AtlasActivationCache
        from modelcypher.cli.commands.geometry.research.common import cleanup_memory
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
        from modelcypher.core.domain.geometry.manifold_evidence import (
            compute_manifold_evidence,
        )
        from modelcypher.core.domain.geometry.positive_geometry import (
            compute_positive_grassmann_signature,
        )

        base_domain_list = _resolve_domains_csv(base_domains)
        if base_domains and not base_domain_list:
            raise ValueError("No valid domains resolved from --base-domains.")

        fingerprint_domain_list = _resolve_domains_csv(fingerprint_domains)
        if fingerprint_domains and not fingerprint_domain_list:
            raise ValueError("No valid domains resolved from --fingerprint-domains.")

        if fingerprint_domains and not domain_fingerprints:
            domain_fingerprints = True

        if domain_fingerprints and not fingerprint_domain_list:
            fingerprint_domain_list = list(AtlasDomain)

        model_obj, tokenizer = load_model_for_training(str(model), adapter_path=adapter)
        embed_tokens, layers_module, norm, num_layers, layer_indices = _resolve_layers(
            model_obj,
            layer,
            layers,
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

        if base_domain_list:
            probes = UnifiedAtlasInventory.probes_by_domain(set(base_domain_list))
        else:
            probes = UnifiedAtlasInventory.all_probes()
        if not probes:
            raise ValueError("No atlas probes available for base report.")
        selected_base = _select_probes(probe_count, probes)
        prompts = _prompts_from_probes(selected_base)

        provider.preload_layers(prompts, layer_indices)
        layer_reports = []
        for layer_idx in layer_indices:
            activations = provider.get_activations(prompts, layer_idx)
            arr = backend.array(activations)
            backend.eval(arr)
            manifold_report = compute_manifold_evidence(arr, backend=backend)
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
                    "manifoldEvidence": asdict(manifold_report),
                    "positiveGeometry": (
                        positive_signature.to_dict()
                        if positive_signature is not None
                        else None
                    ),
                }
            )
        provider.clear_layers(layer_indices)

        domain_reports: dict[str, object] = {}
        if domain_fingerprints:
            for domain in fingerprint_domain_list:
                domain_probes = UnifiedAtlasInventory.probes_by_domain({domain})
                if not domain_probes:
                    continue
                selected_domain = _select_probes(probe_count, domain_probes)
                domain_prompts = _prompts_from_probes(selected_domain)
                provider.preload_layers(domain_prompts, layer_indices)
                domain_layer_reports = []
                for layer_idx in layer_indices:
                    activations = provider.get_activations(domain_prompts, layer_idx)
                    arr = backend.array(activations)
                    backend.eval(arr)
                    signature = compute_positive_grassmann_signature(
                        arr,
                        backend=backend,
                        max_minors=positive_geometry_max_minors,
                        selection=positive_geometry_selection,
                        rank_source=positive_geometry_rank_source,
                        rank_override=positive_geometry_rank,
                    )
                    domain_layer_reports.append(
                        {
                            "layer": layer_idx,
                            "activationCount": len(activations),
                            "positiveGeometry": signature.to_dict(),
                        }
                    )
                provider.clear_layers(layer_indices)
                domain_reports[domain.value] = {
                    "probeCount": len(selected_domain),
                    "layerReports": domain_layer_reports,
                }

        cleanup_memory()

        summary = {
            "layers": [_summarize_layer_report(report) for report in layer_reports],
            "domainFingerprints": {
                name: _summarize_domain_report(report)
                for name, report in (domain_reports or {}).items()
            } or None,
        }

        payload = {
            "_schema": "mc.geometry.report.model.v1",
            "modelPath": str(model),
            "adapterPath": adapter,
            "probeCount": len(selected_base),
            "layers": layer_indices,
            "baseDomains": [domain.value for domain in base_domain_list] or None,
            "report": {
                "layerReports": layer_reports,
            },
            "domainFingerprints": domain_reports or None,
            "summary": summary,
            "config": {
                "positiveGeometry": positive_geometry,
                "positiveGeometrySelection": positive_geometry_selection,
                "positiveGeometryMaxMinors": positive_geometry_max_minors,
                "positiveGeometryRankSource": positive_geometry_rank_source,
                "positiveGeometryRankOverride": positive_geometry_rank,
                "domainFingerprints": domain_fingerprints,
                "fingerprintDomains": [domain.value for domain in fingerprint_domain_list]
                if domain_fingerprints
                else None,
            },
        }

        if output:
            from modelcypher.utils.json import dump_json

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(dump_json(payload, pretty=True))

        if context.output_format == "text":
            lines = [
                "GEOMETRY REPORT",
                f"Model: {model}",
                f"Adapter: {adapter or '-'}",
                f"Layers: {', '.join(str(x) for x in layer_indices)}",
                f"Probe count: {len(selected_base)}",
                f"Base domains: {', '.join(domain.value for domain in base_domain_list) if base_domain_list else '-'}",
                "",
                "Layer summary:",
            ]
            for item in summary["layers"]:
                pg = item.get("positiveGeometry") or {}
                lines.append(
                    f"  L{item['layer']} "
                    f"ID={item['intrinsicDimension']} "
                    f"ER_sh={item['effectiveRankShannon']} "
                    f"SE={item['spectralEntropy']} "
                    f"PG_rank={pg.get('rank')} "
                    f"PG_pos={pg.get('posFraction')} "
                    f"PG_ent={pg.get('signEntropy')}"
                )
            if summary["domainFingerprints"]:
                lines.append("")
                lines.append("Domain fingerprints:")
                for name, report in summary["domainFingerprints"].items():
                    lines.append(f"  {name} (probes={report.get('probeCount')})")
                    for lr in report.get("layerReports", []):
                        pg = lr.get("positiveGeometry") or {}
                        lines.append(
                            f"    L{lr.get('layer')} "
                            f"rank={pg.get('rank')} "
                            f"pos={pg.get('posFraction')} "
                            f"ent={pg.get('signEntropy')} "
                            f"zero={pg.get('zeroFraction')}"
                        )
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        write_error(f"Geometry report failed: {exc}", context.output_format)
        raise typer.Exit(1) from exc


__all__ = ["app"]
