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

"""Knowledge density research CLI commands.

Experimental commands for testing the knowledge density hypothesis:
- Merge is not "blend two models" but "diff knowledge states, graft into sparse regions"
- Dense regions (well-learned) should not be modified
- Sparse regions are graft opportunities

Commands:
- concept-density: Measure knowledge density per concept at a layer
- knowledge-diff: Diff knowledge states between source and target models
- sparse-regions: Find sparse regions where grafting would add value
"""

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
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    """Activation provider for knowledge density analysis."""

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


def _load_model_and_provider(model_path: str, k_neighbors: int):
    """Load model and create activation provider."""
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
    frechet_max_k = max(k_neighbors, 20)
    provider = BackboneActivationProvider(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        frechet_k_neighbors=k_neighbors,
        frechet_max_k_neighbors=frechet_max_k,
    )

    return model, tokenizer, backend, provider, num_layers


@app.command("concept-density")
def concept_density(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, "--layer", "-l", help="Layer to analyze (default is last)"),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_probes: int = typer.Option(0, "--max-probes", help="Limit probes (0 = all)"),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Measure knowledge density per concept at a model layer.

    Knowledge density indicates how "well-learned" a concept is:
    - High density = model has compressed representation efficiently (mastered)
    - Low density = representation is sparse/incomplete (gap)

    This helps identify which concepts are graft opportunities (sparse)
    vs which should not be touched (dense).
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
    )

    model, tokenizer, backend, provider, num_layers = _load_model_and_provider(
        model_path, k_neighbors
    )

    target_layer = layer if layer >= 0 else num_layers - 1

    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]
    if max_probes > 0 and max_probes < len(probes):
        probes = probes[:max_probes]

    analyzer = KnowledgeDensityAnalyzer(backend=backend)
    config = KnowledgeDensityConfig()

    logger.info("Analyzing %d probes at layer %d...", len(probes), target_layer)
    profile = analyzer.analyze_layer(probes, provider, target_layer, config)

    # Build output payload
    payload = {
        "_schema": "mc.geometry.research.concept_density.v1",
        "modelPath": model_path,
        "layer": profile.layer,
        "totalConcepts": len(profile.concept_densities),
        "meanDensity": profile.mean_density,
        "medianDensity": profile.median_density,
        "sparseConceptCount": profile.sparse_concept_count,
        "denseConceptCount": profile.dense_concept_count,
        "densityThreshold": profile.density_threshold,
        "concepts": [
            {
                "probeID": c.probe_id,
                "name": c.name,
                "domain": c.domain,
                "intrinsicDimension": c.intrinsic_dimension,
                "densityScore": c.density_score,
                "activationVariance": c.activation_variance,
                "clusterTightness": c.cluster_tightness,
                "dimensionClass": c.dimension_class,
            }
            for c in sorted(profile.concept_densities, key=lambda x: x.density_score)
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "CONCEPT DENSITY ANALYSIS",
            f"Model: {model_path}",
            f"Layer: {target_layer}",
            f"Concepts: {len(profile.concept_densities)}",
            f"Mean Density: {profile.mean_density:.3f}",
            f"Median Density: {profile.median_density:.3f}",
            f"Threshold: {profile.density_threshold:.3f}",
            f"Sparse (graft opportunities): {profile.sparse_concept_count}",
            f"Dense (do not touch): {profile.dense_concept_count}",
            "",
            "Top 10 Sparse Concepts (lowest density):",
        ]
        sparse = sorted(profile.concept_densities, key=lambda x: x.density_score)[:10]
        for c in sparse:
            lines.append(f"  {c.name}: density={c.density_score:.3f}, dim={c.intrinsic_dimension:.1f}")

        lines.append("")
        lines.append("Top 10 Dense Concepts (highest density):")
        dense = sorted(profile.concept_densities, key=lambda x: x.density_score, reverse=True)[:10]
        for c in dense:
            lines.append(f"  {c.name}: density={c.density_score:.3f}, dim={c.intrinsic_dimension:.1f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("knowledge-diff")
def knowledge_diff(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model directory"),
    target_path: str = typer.Argument(..., help="Path to target model directory"),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Layers to analyze (repeatable, default: 0, mid, last)"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_probes: int = typer.Option(0, "--max-probes", help="Limit probes (0 = all)"),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Diff knowledge states between source and target models.

    Identifies graft opportunities where source has dense representation
    but target is sparse. Positive opportunity scores indicate concepts
    where grafting would add value.

    Output includes ranked list of graft opportunities and concepts
    that should NOT be grafted (target already knows).
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
        ModelDensityProfile,
    )
    from modelcypher.core.domain.geometry.knowledge_diff import KnowledgeDiffer

    # Load source model
    logger.info("Loading source model: %s", source_path)
    _, _, source_backend, source_provider, source_num_layers = _load_model_and_provider(
        source_path, k_neighbors
    )

    # Load target model
    logger.info("Loading target model: %s", target_path)
    _, _, target_backend, target_provider, target_num_layers = _load_model_and_provider(
        target_path, k_neighbors
    )

    # Resolve layers to analyze
    num_layers = min(source_num_layers, target_num_layers)
    if not layers:
        layers = [0, num_layers // 2, num_layers - 1]

    resolved_layers: list[int] = []
    for layer in layers:
        layer_idx = layer if layer >= 0 else num_layers + layer
        if 0 <= layer_idx < num_layers:
            resolved_layers.append(layer_idx)
    resolved_layers = sorted(set(resolved_layers))

    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]
    if max_probes > 0 and max_probes < len(probes):
        probes = probes[:max_probes]

    config = KnowledgeDensityConfig()

    # Analyze source model
    logger.info("Analyzing source model density...")
    source_analyzer = KnowledgeDensityAnalyzer(backend=source_backend)
    source_profile = source_analyzer.analyze_model(probes, source_provider, resolved_layers, config)
    source_profile = ModelDensityProfile(
        model_path=source_path,
        layers=source_profile.layers,
        layer_profiles=source_profile.layer_profiles,
        domain_densities=source_profile.domain_densities,
        overall_density=source_profile.overall_density,
        sparse_concepts=source_profile.sparse_concepts,
        dense_concepts=source_profile.dense_concepts,
    )

    # Analyze target model
    logger.info("Analyzing target model density...")
    target_analyzer = KnowledgeDensityAnalyzer(backend=target_backend)
    target_profile = target_analyzer.analyze_model(probes, target_provider, resolved_layers, config)
    target_profile = ModelDensityProfile(
        model_path=target_path,
        layers=target_profile.layers,
        layer_profiles=target_profile.layer_profiles,
        domain_densities=target_profile.domain_densities,
        overall_density=target_profile.overall_density,
        sparse_concepts=target_profile.sparse_concepts,
        dense_concepts=target_profile.dense_concepts,
    )

    # Compute knowledge diff
    logger.info("Computing knowledge diff...")
    differ = KnowledgeDiffer()
    diff = differ.diff(source_profile, target_profile)

    # Build output payload
    payload = {
        "_schema": "mc.geometry.research.knowledge_diff.v1",
        "sourcePath": diff.source_path,
        "targetPath": diff.target_path,
        "layers": resolved_layers,
        "totalConcepts": diff.total_concepts,
        "overallSourceDensity": diff.overall_source_density,
        "overallTargetDensity": diff.overall_target_density,
        "overallOpportunity": diff.overall_opportunity,
        "highOpportunityCount": diff.high_opportunity_count,
        "noGraftCount": diff.no_graft_count,
        "domainDiffs": [
            {
                "domain": dd.domain,
                "meanSourceDensity": dd.mean_source_density,
                "meanTargetDensity": dd.mean_target_density,
                "meanOpportunity": dd.mean_opportunity,
                "conceptCount": dd.concept_count,
                "highOpportunityCount": dd.high_opportunity_count,
            }
            for dd in diff.domain_diffs.values()
        ],
        "topGraftOpportunities": [
            {
                "probeID": o.probe_id,
                "name": o.name,
                "domain": o.domain,
                "layer": o.layer,
                "sourceDensity": o.source_density,
                "targetDensity": o.target_density,
                "opportunityScore": o.opportunity_score,
                "classification": o.classification,
            }
            for o in diff.ranked_opportunities[:50]  # Top 50
        ],
        "noGraftConcepts": [
            {
                "probeID": o.probe_id,
                "name": o.name,
                "domain": o.domain,
                "layer": o.layer,
                "opportunityScore": o.opportunity_score,
            }
            for o in diff.no_graft_concepts[:20]  # Top 20 no-graft
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "KNOWLEDGE STATE DIFF",
            f"Source: {source_path}",
            f"Target: {target_path}",
            f"Layers: {', '.join(str(l) for l in resolved_layers)}",
            "",
            f"Total concepts compared: {diff.total_concepts}",
            f"Overall source density: {diff.overall_source_density:.3f}",
            f"Overall target density: {diff.overall_target_density:.3f}",
            f"Overall opportunity: {diff.overall_opportunity:.3f}",
            "",
            f"High graft opportunities: {diff.high_opportunity_count}",
            f"Do not graft (target dense): {diff.no_graft_count}",
            "",
            "Domain Summary:",
        ]
        for dd in sorted(diff.domain_diffs.values(), key=lambda x: x.mean_opportunity, reverse=True):
            lines.append(
                f"  {dd.domain}: opportunity={dd.mean_opportunity:.3f}, "
                f"source={dd.mean_source_density:.3f}, target={dd.mean_target_density:.3f}"
            )

        lines.append("")
        lines.append("Top 10 Graft Opportunities (source knows, target doesn't):")
        for o in diff.ranked_opportunities[:10]:
            lines.append(
                f"  [{o.domain}] {o.name} L{o.layer}: "
                f"score={o.opportunity_score:.3f} (source={o.source_density:.2f}, target={o.target_density:.2f})"
            )

        lines.append("")
        lines.append("Top 5 No-Graft Concepts (target already knows):")
        for o in diff.no_graft_concepts[:5]:
            lines.append(
                f"  [{o.domain}] {o.name} L{o.layer}: score={o.opportunity_score:.3f}"
            )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("sparse-regions")
def sparse_regions(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Layers to analyze (repeatable)"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_probes: int = typer.Option(0, "--max-probes", help="Limit probes (0 = all)"),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Find sparse regions in a model where grafting would add value.

    Sparse regions are concepts where the model has incomplete/gap
    representations. These are opportunities for knowledge transfer
    from another model.

    Reports per-layer and per-domain sparse concept counts, plus
    a ranked list of the most sparse concepts.
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
        ModelDensityProfile,
    )

    model, tokenizer, backend, provider, num_layers = _load_model_and_provider(
        model_path, k_neighbors
    )

    # Resolve layers
    if not layers:
        layers = [0, num_layers // 2, num_layers - 1]

    resolved_layers: list[int] = []
    for layer in layers:
        layer_idx = layer if layer >= 0 else num_layers + layer
        if 0 <= layer_idx < num_layers:
            resolved_layers.append(layer_idx)
    resolved_layers = sorted(set(resolved_layers))

    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]
    if max_probes > 0 and max_probes < len(probes):
        probes = probes[:max_probes]

    analyzer = KnowledgeDensityAnalyzer(backend=backend)
    config = KnowledgeDensityConfig()

    logger.info("Analyzing %d probes across %d layers...", len(probes), len(resolved_layers))
    profile = analyzer.analyze_model(probes, provider, resolved_layers, config)
    profile = ModelDensityProfile(
        model_path=model_path,
        layers=profile.layers,
        layer_profiles=profile.layer_profiles,
        domain_densities=profile.domain_densities,
        overall_density=profile.overall_density,
        sparse_concepts=profile.sparse_concepts,
        dense_concepts=profile.dense_concepts,
    )

    # Build output payload
    payload = {
        "_schema": "mc.geometry.research.sparse_regions.v1",
        "modelPath": model_path,
        "layers": resolved_layers,
        "overallDensity": profile.overall_density,
        "totalSparseCount": len(profile.sparse_concepts),
        "totalDenseCount": len(profile.dense_concepts),
        "domainDensities": profile.domain_densities,
        "layerSummaries": [
            {
                "layer": lp.layer,
                "meanDensity": lp.mean_density,
                "medianDensity": lp.median_density,
                "sparseCount": lp.sparse_concept_count,
                "denseCount": lp.dense_concept_count,
                "threshold": lp.density_threshold,
            }
            for lp in profile.layer_profiles.values()
        ],
        "sparseConcepts": [
            {
                "probeID": c.probe_id,
                "name": c.name,
                "domain": c.domain,
                "layer": c.layer,
                "densityScore": c.density_score,
                "intrinsicDimension": c.intrinsic_dimension,
            }
            for c in sorted(profile.sparse_concepts, key=lambda x: x.density_score)[:100]
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Model: {model_path}",
            f"Layers: {', '.join(str(l) for l in resolved_layers)}",
            f"Overall Density: {profile.overall_density:.3f}",
            f"Total Sparse Concepts: {len(profile.sparse_concepts)}",
            f"Total Dense Concepts: {len(profile.dense_concepts)}",
            "",
            "Layer Summary:",
        ]
        for lp in sorted(profile.layer_profiles.values(), key=lambda x: x.layer):
            lines.append(
                f"  L{lp.layer}: sparse={lp.sparse_concept_count}, dense={lp.dense_concept_count}, "
                f"mean_density={lp.mean_density:.3f}"
            )

        lines.append("")
        lines.append("Domain Densities:")
        for domain, density in sorted(profile.domain_densities.items(), key=lambda x: x[1]):
            lines.append(f"  {domain}: {density:.3f}")

        lines.append("")
        lines.append("Top 15 Sparse Concepts (most sparse):")
        for c in sorted(profile.sparse_concepts, key=lambda x: x.density_score)[:15]:
            lines.append(f"  [{c.domain}] {c.name} L{c.layer}: density={c.density_score:.3f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
