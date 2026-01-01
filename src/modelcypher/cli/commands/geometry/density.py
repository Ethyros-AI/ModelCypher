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

"""Knowledge density CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import (
    AtlasDomain,
    AtlasSource,
    UnifiedAtlasInventory,
)
from modelcypher.core.domain.geometry.knowledge_density import (
    KnowledgeDensityAnalyzer,
    ModelDensityProfile,
)
from modelcypher.core.domain.geometry.knowledge_diff import (
    KnowledgeDiffer,
    compute_graft_mask,
)
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        frechet_k_neighbors: int | None = None,
        frechet_max_k_neighbors: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend
        self._frechet_k_neighbors = frechet_k_neighbors
        self._frechet_max_k_neighbors = frechet_max_k_neighbors

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        activations = []
        pending = []

        for text in texts:
            if not text:
                continue
            try:
                tokens = self._tokenizer.encode(text)
                if not tokens:
                    continue
                input_ids = self._backend.array([tokens])
                hidden = forward_through_backbone(
                    input_ids,
                    self._embed_tokens,
                    self._layers,
                    self._norm,
                    target_layer=layer,
                    backend=self._backend,
                )
                mean = frechet_mean(
                    hidden[0],
                    backend=self._backend,
                    k_neighbors=self._frechet_k_neighbors,
                    max_k_neighbors=self._frechet_max_k_neighbors,
                )
                self._backend.async_eval(mean)
                pending.append(mean)
                activations.append(mean)
            except Exception as exc:
                logger.debug("Activation failed for text '%s': %s", text, exc)
                continue

        if pending:
            self._backend.eval(*pending)

        return [self._backend.to_numpy(vec).tolist() for vec in activations]


def _parse_sources(values: list[str] | None) -> set[AtlasSource] | None:
    if not values:
        return None
    allowed = {s.value for s in AtlasSource}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise typer.BadParameter(
            f"Invalid sources: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed))}"
        )
    return {AtlasSource(value) for value in values}


def _parse_domains(values: list[str] | None) -> set[AtlasDomain] | None:
    if not values:
        return None
    allowed = {d.value for d in AtlasDomain}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise typer.BadParameter(
            f"Invalid domains: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed))}"
        )
    return {AtlasDomain(value) for value in values}


def _resolve_layers(layers: list[int] | None, num_layers: int) -> list[int]:
    if not layers:
        return list(range(num_layers))
    resolved: list[int] = []
    for layer in layers:
        layer_idx = layer if layer >= 0 else num_layers + layer
        if 0 <= layer_idx < num_layers:
            resolved.append(layer_idx)
    return sorted(set(resolved))


def _select_probes(
    sources: list[str] | None,
    domains: list[str] | None,
    max_probes: int,
):
    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]
    if max_probes > 0 and max_probes < len(probes):
        probes = probes[:max_probes]
    return probes


def _load_model_and_provider(model_path: str):
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)

    backend = MLXBackend()
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
    )

    return model, tokenizer, backend, provider, num_layers


def _cleanup_memory() -> None:
    import gc
    import time

    gc.collect()
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass
    time.sleep(1)


def _profile_payload(profile: ModelDensityProfile) -> dict:
    return {
        "_schema": "mc.geometry.density.profile.v1",
        "modelPath": profile.model_path,
        "layers": profile.layers,
        "overallDensity": profile.overall_density,
        "domainDensities": profile.domain_densities,
        "layerProfiles": [
            {
                "layer": layer_profile.layer,
                "conceptCount": len(layer_profile.concept_densities),
                "meanDensity": layer_profile.mean_density,
                "medianDensity": layer_profile.median_density,
                "concepts": [
                    {
                        "probeID": c.probe_id,
                        "name": c.name,
                        "domain": c.domain,
                        "layer": c.layer,
                        "intrinsicDimension": c.intrinsic_dimension,
                        "densityScore": c.density_score,
                        "activationVariance": c.activation_variance,
                        "clusterTightness": c.cluster_tightness,
                        "dimensionClass": c.dimension_class,
                    }
                    for c in layer_profile.concept_densities
                ],
            }
            for layer_profile in profile.layer_profiles.values()
        ],
    }


@app.command("profile")
def density_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Layers to analyze (repeatable, default: all)"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_probes: int = typer.Option(0, "--max-probes", help="Limit probes (0 = all)"),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Compute knowledge density profile for a model."""
    context = _context(ctx)

    model, _, backend, provider, num_layers = _load_model_and_provider(model_path)
    resolved_layers = _resolve_layers(layers, num_layers)
    probes = _select_probes(sources, domains, max_probes)

    analyzer = KnowledgeDensityAnalyzer(backend=backend)
    raw_profile = analyzer.analyze_model(probes, provider, resolved_layers)
    profile = ModelDensityProfile(
        model_path=model_path,
        layers=raw_profile.layers,
        layer_profiles=raw_profile.layer_profiles,
        domain_densities=raw_profile.domain_densities,
        overall_density=raw_profile.overall_density,
    )

    payload = _profile_payload(profile)

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))

    if context.output_format == "text":
        layer_lines = []
        for layer_profile in profile.layer_profiles.values():
            layer_lines.append(
                f"  layer {layer_profile.layer}: "
                f"concepts={len(layer_profile.concept_densities)} "
                f"mean={layer_profile.mean_density:.4f} "
                f"median={layer_profile.median_density:.4f}"
            )

        domain_lines = [
            f"  {domain}: {density:.4f}"
            for domain, density in sorted(profile.domain_densities.items())
        ]

        lines = [
            "DENSITY PROFILE",
            f"Model: {model_path}",
            f"Layers analyzed: {', '.join(str(layer) for layer in profile.layers)}",
            f"Overall density: {profile.overall_density:.4f}",
        ]
        if domain_lines:
            lines.append("Domain densities:")
            lines.extend(domain_lines)
        if layer_lines:
            lines.append("Layer summaries:")
            lines.extend(layer_lines)

        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)

    del model
    _cleanup_memory()


@app.command("diff")
def density_diff(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model directory"),
    target_path: str = typer.Argument(..., help="Path to target model directory"),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Layers to analyze (repeatable, default: all)"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_probes: int = typer.Option(0, "--max-probes", help="Limit probes (0 = all)"),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Diff knowledge density profiles between source and target models."""
    context = _context(ctx)

    probes = _select_probes(sources, domains, max_probes)

    source_model, _, source_backend, source_provider, source_layers = _load_model_and_provider(
        source_path
    )
    target_model, _, target_backend, target_provider, target_layers = _load_model_and_provider(
        target_path
    )

    resolved_layers = _resolve_layers(layers, min(source_layers, target_layers))

    source_analyzer = KnowledgeDensityAnalyzer(backend=source_backend)
    source_profile_raw = source_analyzer.analyze_model(
        probes, source_provider, resolved_layers
    )
    source_profile = ModelDensityProfile(
        model_path=source_path,
        layers=source_profile_raw.layers,
        layer_profiles=source_profile_raw.layer_profiles,
        domain_densities=source_profile_raw.domain_densities,
        overall_density=source_profile_raw.overall_density,
    )

    target_analyzer = KnowledgeDensityAnalyzer(backend=target_backend)
    target_profile_raw = target_analyzer.analyze_model(
        probes, target_provider, resolved_layers
    )
    target_profile = ModelDensityProfile(
        model_path=target_path,
        layers=target_profile_raw.layers,
        layer_profiles=target_profile_raw.layer_profiles,
        domain_densities=target_profile_raw.domain_densities,
        overall_density=target_profile_raw.overall_density,
    )

    differ = KnowledgeDiffer()
    diff = differ.diff(source_profile, target_profile)
    graft_mask = compute_graft_mask(diff)

    payload = {
        "_schema": "mc.geometry.density.diff.v1",
        "sourcePath": diff.source_path,
        "targetPath": diff.target_path,
        "layers": resolved_layers,
        "overallSourceDensity": diff.overall_source_density,
        "overallTargetDensity": diff.overall_target_density,
        "overallOpportunity": diff.overall_opportunity,
        "totalConcepts": diff.total_concepts,
        "positiveOpportunityCount": diff.positive_opportunity_count,
        "nonpositiveOpportunityCount": diff.nonpositive_opportunity_count,
        "domainDiffs": [
            {
                "domain": dd.domain,
                "meanSourceDensity": dd.mean_source_density,
                "meanTargetDensity": dd.mean_target_density,
                "meanOpportunity": dd.mean_opportunity,
                "conceptCount": dd.concept_count,
                "positiveOpportunityCount": dd.positive_opportunity_count,
            }
            for dd in diff.domain_diffs.values()
        ],
        "layerDiffs": [
            {
                "layer": ld.layer,
                "meanOpportunity": ld.mean_opportunity,
                "positiveOpportunityCount": ld.positive_opportunity_count,
                "nonpositiveOpportunityCount": ld.nonpositive_opportunity_count,
            }
            for ld in diff.layer_diffs.values()
        ],
        "rankedOpportunities": [
            {
                "probeID": opp.probe_id,
                "name": opp.name,
                "domain": opp.domain,
                "layer": opp.layer,
                "sourceDensity": opp.source_density,
                "targetDensity": opp.target_density,
                "opportunityScore": opp.opportunity_score,
            }
            for opp in diff.ranked_opportunities
        ],
        "graftMask": graft_mask,
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))

    if context.output_format == "text":
        lines = [
            "DENSITY DIFF",
            f"Source: {source_path}",
            f"Target: {target_path}",
            f"Layers analyzed: {', '.join(str(layer) for layer in resolved_layers)}",
            f"Overall source density: {diff.overall_source_density:.4f}",
            f"Overall target density: {diff.overall_target_density:.4f}",
            f"Overall opportunity: {diff.overall_opportunity:.4f}",
            f"Total concepts compared: {diff.total_concepts}",
            f"Positive opportunities: {diff.positive_opportunity_count}",
            f"Nonpositive opportunities: {diff.nonpositive_opportunity_count}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
    else:
        write_output(payload, context.output_format, context.pretty)

    del source_model
    del target_model
    _cleanup_memory()
