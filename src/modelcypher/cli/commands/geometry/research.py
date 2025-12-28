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


@app.command("graft-boundary")
def graft_boundary(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model directory"),
    target_path: str = typer.Argument(..., help="Path to target model directory"),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Layers to analyze (repeatable)"
    ),
    density_brackets: str = typer.Option(
        "0.3,0.5,0.7,0.9",
        "--density-brackets",
        help="Comma-separated density thresholds for binning",
    ),
    max_probes: int = typer.Option(50, "--max-probes", help="Limit probes (0 = all)"),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Save results to JSON file"
    ),
) -> None:
    """Analyze graft boundary by correlating density with null space.

    Identifies the density threshold where grafting is likely safe vs harmful
    by analyzing the relationship between concept density and null space
    availability.

    Key insight: Sparse concepts (low density) should have more null space
    available, making grafting safer. Dense concepts have less null space,
    making grafting risky.

    Outputs:
    - Per-density-bracket analysis
    - Null space correlation per layer
    - Recommended graft mask (which layers/concepts to graft)
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
        ModelDensityProfile,
    )
    from modelcypher.core.domain.geometry.knowledge_diff import (
        KnowledgeDiffer,
        compute_graft_mask,
    )
    from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilter

    # Parse density brackets
    brackets = [float(b.strip()) for b in density_brackets.split(",")]
    brackets = sorted(brackets)

    # Load target model (primary model for null space analysis)
    logger.info("Loading target model: %s", target_path)
    target_model, target_tokenizer, target_backend, target_provider, target_num_layers = (
        _load_model_and_provider(target_path, k_neighbors)
    )

    # Load source model
    logger.info("Loading source model: %s", source_path)
    _, _, source_backend, source_provider, source_num_layers = _load_model_and_provider(
        source_path, k_neighbors
    )

    # Resolve layers
    num_layers = min(source_num_layers, target_num_layers)
    if not layers:
        # Analyze more layers for boundary detection
        layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    resolved_layers: list[int] = []
    for layer in layers:
        layer_idx = layer if layer >= 0 else num_layers + layer
        if 0 <= layer_idx < num_layers:
            resolved_layers.append(layer_idx)
    resolved_layers = sorted(set(resolved_layers))

    probes = UnifiedAtlasInventory.all_probes()
    if max_probes > 0 and max_probes < len(probes):
        probes = probes[:max_probes]

    config = KnowledgeDensityConfig()

    # Step 1: Compute knowledge diff
    logger.info("Computing knowledge diff...")
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

    differ = KnowledgeDiffer()
    diff = differ.diff(source_profile, target_profile)

    # Step 2: Compute null space profile for target model
    logger.info("Computing null space profile for target...")
    null_filter = NullSpaceFilter(backend=target_backend)

    # Collect activations for null space analysis
    layer_activations: dict[int, list] = {}
    for layer_idx in resolved_layers:
        activations = []
        for probe in probes[:20]:  # Sample of probes for null space
            texts = list(probe.support_texts or [])[:3]
            if texts:
                acts = target_provider.get_activations(texts, layer_idx)
                activations.extend(acts)
        if activations:
            act_array = target_backend.stack(
                [target_backend.array(a) for a in activations], axis=0
            )
            target_backend.eval(act_array)
            layer_activations[layer_idx] = act_array

    null_profile = null_filter.compute_model_null_space_profile(layer_activations)

    # Step 3: Bin concepts by density and analyze
    def get_bracket_idx(density: float) -> int:
        for i, threshold in enumerate(brackets):
            if density < threshold:
                return i
        return len(brackets)

    bracket_labels = []
    prev = 0.0
    for b in brackets:
        bracket_labels.append(f"{prev:.1f}-{b:.1f}")
        prev = b
    bracket_labels.append(f"{prev:.1f}-1.0")

    # Analyze each bracket
    bracket_analysis = []
    for bracket_idx, label in enumerate(bracket_labels):
        # Find concepts in this bracket
        concepts_in_bracket = [
            opp for opp in diff.ranked_opportunities
            if get_bracket_idx(opp.target_density) == bracket_idx
        ]

        if not concepts_in_bracket:
            bracket_analysis.append({
                "bracket": label,
                "bracketIdx": bracket_idx,
                "conceptCount": 0,
                "meanOpportunity": 0.0,
                "meanTargetDensity": 0.0,
                "meanSourceDensity": 0.0,
                "layerDistribution": {},
                "graftRecommendation": "skip",
            })
            continue

        mean_opp = sum(c.opportunity_score for c in concepts_in_bracket) / len(concepts_in_bracket)
        mean_target = sum(c.target_density for c in concepts_in_bracket) / len(concepts_in_bracket)
        mean_source = sum(c.source_density for c in concepts_in_bracket) / len(concepts_in_bracket)

        # Layer distribution
        layer_dist: dict[int, int] = {}
        for c in concepts_in_bracket:
            layer_dist[c.layer] = layer_dist.get(c.layer, 0) + 1

        # Graft recommendation based on opportunity score
        if mean_opp > 0.2:
            recommendation = "high_priority_graft"
        elif mean_opp > 0.05:
            recommendation = "consider_graft"
        elif mean_opp > -0.05:
            recommendation = "neutral"
        else:
            recommendation = "do_not_graft"

        bracket_analysis.append({
            "bracket": label,
            "bracketIdx": bracket_idx,
            "conceptCount": len(concepts_in_bracket),
            "meanOpportunity": mean_opp,
            "meanTargetDensity": mean_target,
            "meanSourceDensity": mean_source,
            "layerDistribution": layer_dist,
            "graftRecommendation": recommendation,
        })

    # Step 4: Correlate null space with density
    layer_null_density_correlation = []
    for layer_idx in resolved_layers:
        if layer_idx not in null_profile.per_layer:
            continue

        null_info = null_profile.per_layer[layer_idx]

        # Get concepts at this layer
        layer_concepts = [
            opp for opp in diff.ranked_opportunities
            if opp.layer == layer_idx
        ]

        if not layer_concepts:
            continue

        mean_density = sum(c.target_density for c in layer_concepts) / len(layer_concepts)
        mean_opp = sum(c.opportunity_score for c in layer_concepts) / len(layer_concepts)

        layer_null_density_correlation.append({
            "layer": layer_idx,
            "nullFraction": null_info.null_fraction,
            "nullDim": null_info.null_dim,
            "totalDim": null_info.total_dim,
            "meanTargetDensity": mean_density,
            "meanOpportunity": mean_opp,
            "conceptCount": len(layer_concepts),
            "isGraftable": layer_idx in null_profile.graftable_layers,
        })

    # Step 5: Generate graft mask for recommended threshold
    # Find the boundary - first bracket where recommendation is NOT high_priority_graft
    graft_boundary_density = None
    for ba in bracket_analysis:
        if ba["graftRecommendation"] in ["neutral", "do_not_graft"]:
            graft_boundary_density = float(ba["bracket"].split("-")[0])
            break

    if graft_boundary_density is None:
        graft_boundary_density = brackets[-1] if brackets else 0.5

    # Generate graft mask
    graft_mask = compute_graft_mask(diff, include_low_opportunity=False)

    # Build payload
    payload = {
        "_schema": "mc.geometry.research.graft_boundary.v1",
        "sourcePath": source_path,
        "targetPath": target_path,
        "layers": resolved_layers,
        "densityBrackets": brackets,
        "graftBoundaryDensity": graft_boundary_density,
        "bracketAnalysis": bracket_analysis,
        "nullSpaceCorrelation": layer_null_density_correlation,
        "graftableLayers": null_profile.graftable_layers,
        "meanNullFraction": null_profile.mean_null_fraction,
        "totalConcepts": diff.total_concepts,
        "highOpportunityCount": diff.high_opportunity_count,
        "graftMaskSummary": {
            "totalProbes": len(graft_mask),
            "probesWithGraft": sum(
                1 for probe_layers in graft_mask.values()
                if any(probe_layers.values())
            ),
        },
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "GRAFT BOUNDARY ANALYSIS",
            f"Source: {source_path}",
            f"Target: {target_path}",
            f"Layers: {', '.join(str(l) for l in resolved_layers)}",
            "",
            f"Estimated Graft Boundary: density < {graft_boundary_density:.2f}",
            f"Graftable Layers (by null space): {null_profile.graftable_layers}",
            f"Mean Null Fraction: {null_profile.mean_null_fraction:.3f}",
            "",
            "DENSITY BRACKET ANALYSIS:",
            "-" * 60,
        ]

        for ba in bracket_analysis:
            lines.append(
                f"  [{ba['bracket']}] "
                f"concepts={ba['conceptCount']}, "
                f"opportunity={ba['meanOpportunity']:.3f}, "
                f"recommendation={ba['graftRecommendation']}"
            )

        lines.append("")
        lines.append("NULL SPACE / DENSITY CORRELATION BY LAYER:")
        lines.append("-" * 60)
        for corr in layer_null_density_correlation:
            graftable = "GRAFTABLE" if corr["isGraftable"] else "limited"
            lines.append(
                f"  L{corr['layer']}: null_frac={corr['nullFraction']:.3f}, "
                f"density={corr['meanTargetDensity']:.3f}, "
                f"opportunity={corr['meanOpportunity']:.3f} [{graftable}]"
            )

        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 60)
        lines.append(f"  1. Graft concepts with target density < {graft_boundary_density:.2f}")
        lines.append(f"  2. Focus on layers: {null_profile.graftable_layers}")
        lines.append(f"  3. High-opportunity concepts to graft: {diff.high_opportunity_count}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("zero-shot-transfer")
def zero_shot_transfer(
    ctx: typer.Context,
    source_path: str = typer.Argument(
        ..., help="Path to source model directory (knowledge donor)"
    ),
    target_path: str = typer.Argument(
        ..., help="Path to target model directory (graft recipient)"
    ),
    layers: list[int] | None = typer.Option(
        None, "--layer", "-l", help="Specific layers to analyze"
    ),
    density_threshold: float = typer.Option(
        0.5, "--density-threshold", help="Density threshold for sparse/dense classification"
    ),
    max_probes: int = typer.Option(
        50, "--max-probes", help="Maximum probes to analyze"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Path to save results JSON"
    ),
) -> None:
    """Validate zero-shot transfer by analyzing graft candidates.

    Identifies sparse concepts in target that can receive knowledge from source.
    Creates a validation plan for testing transfer without fine-tuning.

    Success criteria:
    - Sparse concept activations shift toward source after grafting
    - Dense concept activations remain stable
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
        ModelDensityProfile,
    )
    from modelcypher.core.domain.geometry.knowledge_diff import KnowledgeDiffer
    from modelcypher.core.domain.geometry.cka import compute_cka

    # Load models using helper
    logger.info("Loading target model: %s", target_path)
    target_model, target_tokenizer, b, target_provider, target_n_layers = (
        _load_model_and_provider(target_path, k_neighbors=10)
    )

    logger.info("Loading source model: %s", source_path)
    _, _, _, source_provider, source_n_layers = (
        _load_model_and_provider(source_path, k_neighbors=10)
    )

    # Determine layers to analyze
    num_layers = min(target_n_layers, source_n_layers)

    if layers:
        resolved_layers = [l for l in layers if l < num_layers]
    else:
        # Analyze key layers: early, middle, late
        resolved_layers = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]
        resolved_layers = sorted(set(resolved_layers))

    logger.info("Analyzing layers: %s", resolved_layers)

    # Load unified atlas probes
    probes = UnifiedAtlasInventory.all_probes()[:max_probes]

    # Analyze density for target
    density_analyzer = KnowledgeDensityAnalyzer(backend=b)
    differ = KnowledgeDiffer()

    target_profile = density_analyzer.analyze_model(
        probes=probes,
        activation_provider=target_provider,
        layers=resolved_layers,
        config=KnowledgeDensityConfig(),
    )

    source_profile = density_analyzer.analyze_model(
        probes=probes,
        activation_provider=source_provider,
        layers=resolved_layers,
        config=KnowledgeDensityConfig(),
    )

    # Set model paths
    target_profile = ModelDensityProfile(
        model_path=target_path,
        layers=target_profile.layers,
        layer_profiles=target_profile.layer_profiles,
        domain_densities=target_profile.domain_densities,
        overall_density=target_profile.overall_density,
        sparse_concepts=target_profile.sparse_concepts,
        dense_concepts=target_profile.dense_concepts,
    )
    source_profile = ModelDensityProfile(
        model_path=source_path,
        layers=source_profile.layers,
        layer_profiles=source_profile.layer_profiles,
        domain_densities=source_profile.domain_densities,
        overall_density=source_profile.overall_density,
        sparse_concepts=source_profile.sparse_concepts,
        dense_concepts=source_profile.dense_concepts,
    )

    # Compute knowledge diff
    diff = differ.diff(source_profile, target_profile)

    # Identify transfer candidates: high opportunity concepts
    transfer_candidates = [
        opp for opp in diff.ranked_opportunities
        if opp.target_density < density_threshold
        and opp.opportunity_score > 0.1
    ]

    # Identify stability checks: dense concepts (should not change)
    stability_checks = [
        opp for opp in diff.ranked_opportunities
        if opp.target_density >= density_threshold
    ]

    # Group transfer candidates by layer
    candidates_by_layer: dict[int, list] = {}
    for cand in transfer_candidates:
        if cand.layer not in candidates_by_layer:
            candidates_by_layer[cand.layer] = []
        candidates_by_layer[cand.layer].append({
            "probeId": cand.probe_id,
            "name": cand.name,
            "domain": cand.domain,
            "targetDensity": cand.target_density,
            "sourceDensity": cand.source_density,
            "opportunityScore": cand.opportunity_score,
        })

    # Group stability checks by layer
    stability_by_layer: dict[int, list] = {}
    for check in stability_checks:
        if check.layer not in stability_by_layer:
            stability_by_layer[check.layer] = []
        stability_by_layer[check.layer].append({
            "probeId": check.probe_id,
            "name": check.name,
            "domain": check.domain,
            "targetDensity": check.target_density,
            "sourceDensity": check.source_density,
        })

    # Compute CKA similarity between source and target at each layer
    # This measures how similar the representations are currently
    layer_cka: dict[int, float] = {}
    for layer in resolved_layers:
        # Get activations for a sample of probes
        sample_probes = probes[:10]
        sample_texts = []
        for probe in sample_probes:
            if probe.support_texts:
                sample_texts.extend(list(probe.support_texts)[:2])

        if sample_texts:
            try:
                target_acts = target_provider.get_activations(sample_texts, layer)
                source_acts = source_provider.get_activations(sample_texts, layer)

                if target_acts and source_acts:
                    target_arr = b.stack([b.array(a) for a in target_acts], axis=0)
                    source_arr = b.stack([b.array(a) for a in source_acts], axis=0)
                    b.eval(target_arr)
                    b.eval(source_arr)

                    # Truncate to shared dimension
                    min_dim = min(target_arr.shape[1], source_arr.shape[1])
                    target_arr = target_arr[:, :min_dim]
                    source_arr = source_arr[:, :min_dim]

                    cka = compute_cka(target_arr, source_arr, b)
                    layer_cka[layer] = float(cka)
            except Exception as e:
                logger.debug("CKA failed for layer %d: %s", layer, e)
                layer_cka[layer] = 0.0
        else:
            layer_cka[layer] = 0.0

    # Build validation plan
    validation_plan = {
        "preGraftChecks": [
            "Measure baseline perplexity on sparse concept prompts",
            "Measure baseline perplexity on dense concept prompts",
            "Record activation norms for transfer candidates",
        ],
        "graftOperation": {
            "targetLayers": list(candidates_by_layer.keys()),
            "conceptsToGraft": len(transfer_candidates),
            "expectedBehavior": "Sparse concept activations shift toward source pattern",
        },
        "postGraftChecks": [
            "Measure post-graft perplexity on sparse concept prompts (should improve)",
            "Measure post-graft perplexity on dense concept prompts (should be stable)",
            "Compare activation shift magnitude for sparse vs dense concepts",
        ],
        "successCriteria": {
            "sparseImprovementThreshold": 0.10,  # 10% perplexity improvement
            "denseDegradationLimit": 0.02,  # 2% degradation limit
        },
    }

    payload = {
        "_schema": "mc.geometry.research.zero_shot_transfer.v1",
        "sourcePath": source_path,
        "targetPath": target_path,
        "layers": resolved_layers,
        "densityThreshold": density_threshold,
        "transferCandidates": {
            "total": len(transfer_candidates),
            "byLayer": {str(k): v for k, v in candidates_by_layer.items()},
            "topOpportunities": [
                {
                    "probeId": c.probe_id,
                    "name": c.name,
                    "domain": c.domain,
                    "layer": c.layer,
                    "targetDensity": c.target_density,
                    "sourceDensity": c.source_density,
                    "opportunityScore": c.opportunity_score,
                }
                for c in transfer_candidates[:10]
            ],
        },
        "stabilityChecks": {
            "total": len(stability_checks),
            "byLayer": {str(k): len(v) for k, v in stability_by_layer.items()},
        },
        "layerCkaSimilarity": {str(k): v for k, v in layer_cka.items()},
        "validationPlan": validation_plan,
        "summary": {
            "canTransfer": len(transfer_candidates) > 0,
            "transferConfidence": (
                "high" if len(transfer_candidates) > 20
                else "medium" if len(transfer_candidates) > 5
                else "low"
            ),
            "stabilityRisk": (
                "low" if len(stability_checks) > len(transfer_candidates) * 2
                else "medium" if len(stability_checks) > len(transfer_candidates)
                else "high"
            ),
        },
    }

    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Results saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "ZERO-SHOT TRANSFER VALIDATION PLAN",
            f"Source (donor): {source_path}",
            f"Target (recipient): {target_path}",
            f"Density threshold: {density_threshold}",
            "",
            "TRANSFER CANDIDATES (sparse in target, dense in source):",
            "-" * 60,
            f"  Total candidates: {len(transfer_candidates)}",
        ]

        for layer, cands in sorted(candidates_by_layer.items()):
            lines.append(f"  Layer {layer}: {len(cands)} concepts")

        lines.append("")
        lines.append("TOP 10 TRANSFER OPPORTUNITIES:")
        lines.append("-" * 60)
        for c in transfer_candidates[:10]:
            lines.append(
                f"  {c.name} (L{c.layer}): "
                f"target={c.target_density:.3f}, source={c.source_density:.3f}, "
                f"opportunity={c.opportunity_score:.3f}"
            )

        lines.append("")
        lines.append("STABILITY CHECKS (should not change):")
        lines.append("-" * 60)
        lines.append(f"  Total dense concepts: {len(stability_checks)}")
        for layer, checks in sorted(stability_by_layer.items()):
            lines.append(f"  Layer {layer}: {len(checks)} concepts to protect")

        lines.append("")
        lines.append("LAYER CKA SIMILARITY (current alignment):")
        lines.append("-" * 60)
        for layer, cka in sorted(layer_cka.items()):
            alignment = "high" if cka > 0.7 else "medium" if cka > 0.4 else "low"
            lines.append(f"  Layer {layer}: CKA={cka:.3f} ({alignment} alignment)")

        lines.append("")
        lines.append("TRANSFER RECOMMENDATION:")
        lines.append("-" * 60)
        if len(transfer_candidates) > 20:
            lines.append("  HIGH CONFIDENCE: Many transfer opportunities identified")
            lines.append(f"  Focus layers: {list(candidates_by_layer.keys())}")
        elif len(transfer_candidates) > 5:
            lines.append("  MEDIUM CONFIDENCE: Some transfer opportunities")
            lines.append("  Consider targeted grafting on highest-opportunity concepts")
        else:
            lines.append("  LOW CONFIDENCE: Few transfer opportunities")
            lines.append("  Target may already be dense; grafting may not add value")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("build-eval-dataset")
def build_eval_dataset(
    ctx: typer.Context,
    output_path: str = typer.Argument(..., help="Path to output JSONL file"),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    max_texts_per_probe: int = typer.Option(
        10, "--max-texts-per-probe", help="Max support texts per probe"
    ),
    include_name: bool = typer.Option(
        True, "--include-name/--no-include-name", help="Include probe name as text"
    ),
    include_description: bool = typer.Option(
        True, "--include-description/--no-include-description", help="Include probe description as text"
    ),
) -> None:
    """Build evaluation dataset from UnifiedAtlas probe support texts.

    Extracts support texts from probes matching the specified filters and
    writes them to a JSONL file suitable for perplexity evaluation with
    `mc eval run`.

    Example usage:
        mc geometry research build-eval-dataset math-eval.jsonl --domain mathematical
        mc geometry research build-eval-dataset dense-eval.jsonl --domain temporal --domain spatial
    """
    context = _context(ctx)

    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]

    # Collect texts
    texts: list[str] = []
    probe_count = 0
    for probe in probes:
        probe_texts: list[str] = []

        if include_name and probe.name:
            probe_texts.append(probe.name)
        if include_description and probe.description:
            probe_texts.append(probe.description)

        if probe.support_texts:
            for text in list(probe.support_texts)[:max_texts_per_probe]:
                if text and text.strip():
                    probe_texts.append(text.strip())

        if probe_texts:
            probe_count += 1
            texts.extend(probe_texts)

    # Write JSONL
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")

    # Build summary payload
    domains_used = sorted(set(p.domain.value if hasattr(p.domain, 'value') else str(p.domain) for p in probes))
    sources_used = sorted(set(p.source.value if hasattr(p.source, 'value') else str(p.source) for p in probes))

    payload = {
        "_schema": "mc.geometry.research.build_eval_dataset.v1",
        "outputPath": str(output_path),
        "totalTexts": len(texts),
        "probeCount": probe_count,
        "domainsUsed": domains_used,
        "sourcesUsed": sources_used,
        "includesName": include_name,
        "includesDescription": include_description,
        "maxTextsPerProbe": max_texts_per_probe,
    }

    if context.output_format == "text":
        lines = [
            "EVALUATION DATASET CREATED",
            f"Output: {output_path}",
            f"Total texts: {len(texts)}",
            f"Probes used: {probe_count}",
            f"Domains: {', '.join(domains_used)}",
            f"Sources: {', '.join(sources_used)}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
