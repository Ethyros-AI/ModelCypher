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

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    """Activation provider for knowledge density analysis.

    Uses arithmetic mean for token aggregation (mean-pooling). This is
    appropriate for aggregating tokens within a single sequence. The
    manifold-aware Fréchet mean is used later in the intrinsic dimension
    estimation when comparing ACROSS texts/concepts.
    """

    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend

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
                # Arithmetic mean for token aggregation (mean-pooling)
                # Fréchet mean is used later in intrinsic dimension estimation
                # when comparing across texts, not for within-text pooling
                mean = self._backend.mean(hidden[0], axis=0)
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


def _load_model_and_provider(model_path: str, k_neighbors: int = 10):
    """Load model and create activation provider.

    Args:
        model_path: Path to the model directory.
        k_neighbors: k for geodesic distance computation in intrinsic dimension.
                    Not used for token aggregation (which uses arithmetic mean).
    """
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
            bracket_analysis.append(
                {
                    "bracket": label,
                    "bracketIdx": bracket_idx,
                    "conceptCount": 0,
                    "meanOpportunity": 0.0,
                    "meanTargetDensity": 0.0,
                    "meanSourceDensity": 0.0,
                    "layerDistribution": {},
                }
            )
            continue

        mean_opp = sum(c.opportunity_score for c in concepts_in_bracket) / len(concepts_in_bracket)
        mean_target = sum(c.target_density for c in concepts_in_bracket) / len(concepts_in_bracket)
        mean_source = sum(c.source_density for c in concepts_in_bracket) / len(concepts_in_bracket)

        # Layer distribution
        layer_dist: dict[int, int] = {}
        for c in concepts_in_bracket:
            layer_dist[c.layer] = layer_dist.get(c.layer, 0) + 1

        # Raw measurements only - no interpretation strings (per CLAUDE.md "No Vibes")
        bracket_analysis.append({
            "bracket": label,
            "bracketIdx": bracket_idx,
            "conceptCount": len(concepts_in_bracket),
            "meanOpportunity": mean_opp,
            "meanTargetDensity": mean_target,
            "meanSourceDensity": mean_source,
            "layerDistribution": layer_dist,
            # opportunityPositive indicates whether grafting adds value for this bracket
            "opportunityPositive": mean_opp > 0,
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
    # Find the boundary - first bracket where opportunity score is not positive
    graft_boundary_density = None
    for ba in bracket_analysis:
        if not ba["opportunityPositive"]:
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
                f"opportunity={ba['meanOpportunity']:.3f}"
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
    from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka

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

                    result = compute_cka(
                        target_arr,
                        source_arr,
                        backend=b,
                        estimator=HSICEstimator.AUTO,
                        feature_bias_correction=True,
                    )
                    layer_cka[layer] = (
                        float(result.cka_corrected)
                        if result.cka_corrected is not None
                        else float(result.cka)
                    )
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
            # Raw measurements only - no interpretation strings (per CLAUDE.md "No Vibes")
            "transferCandidateCount": len(transfer_candidates),
            "stabilityCheckCount": len(stability_checks),
            "stabilityToTransferRatio": (
                len(stability_checks) / len(transfer_candidates)
                if len(transfer_candidates) > 0
                else float("inf")
            ),
            "meanLayerCka": (
                sum(layer_cka.values()) / len(layer_cka)
                if layer_cka
                else 0.0
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
        lines.append("LAYER CKA SIMILARITY:")
        lines.append("-" * 60)
        for layer, cka in sorted(layer_cka.items()):
            # Raw values only - no interpretation (per CLAUDE.md "No Vibes")
            lines.append(f"  Layer {layer}: CKA={cka:.4f}")

        lines.append("")
        lines.append("SUMMARY STATISTICS:")
        lines.append("-" * 60)
        mean_cka = sum(layer_cka.values()) / len(layer_cka) if layer_cka else 0.0
        ratio = (
            len(stability_checks) / len(transfer_candidates)
            if len(transfer_candidates) > 0
            else float("inf")
        )
        lines.append(f"  Transfer candidates: {len(transfer_candidates)}")
        lines.append(f"  Stability checks: {len(stability_checks)}")
        lines.append(f"  Stability/transfer ratio: {ratio:.2f}")
        lines.append(f"  Mean layer CKA: {mean_cka:.4f}")
        lines.append(f"  Target layers: {list(candidates_by_layer.keys())}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("full-profile")
def full_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    output_path: str = typer.Option(
        None, "--output-path", "-o", help="Save profile to JSON file"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
    checkpoint_dir: str | None = typer.Option(
        None, "--checkpoint-dir", help="Directory for incremental checkpoints"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from last checkpoint"
    ),
) -> None:
    """Generate comprehensive knowledge density profile for a model.

    Profiles ALL layers and ALL domains to create a complete map of where
    the model is strong (dense) and weak (sparse) across the entire
    representation space.

    This is compute-intensive but produces the data needed for informed
    knowledge transplant decisions. No sampling, no shortcuts.

    Output includes:
    - Per-layer density statistics for each domain
    - Per-concept density scores at every layer
    - Domain strength rankings by layer
    - Overall model capability fingerprint

    Use --checkpoint-dir to save progress incrementally (recommended for
    large models). Use --resume to continue from last checkpoint.
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
    )

    # Load model
    model, tokenizer, backend, provider, num_layers = _load_model_and_provider(
        model_path, k_neighbors
    )

    # All layers, no sampling
    all_layers = list(range(num_layers))

    # Parse filters
    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    # Get ALL probes (no max_probes limit)
    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]

    logger.info(
        "Full profile: %d probes × %d layers = %d measurements",
        len(probes), num_layers, len(probes) * num_layers
    )

    # Check for checkpoint to resume from
    checkpoint_data: dict = {}
    completed_layers: set[int] = set()
    if checkpoint_dir and resume:
        checkpoint_path = Path(checkpoint_dir) / "full_profile_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_data = json.loads(checkpoint_path.read_text())
            completed_layers = set(checkpoint_data.get("completedLayers", []))
            logger.info("Resuming from checkpoint: %d layers complete", len(completed_layers))

    analyzer = KnowledgeDensityAnalyzer(backend=backend)
    config = KnowledgeDensityConfig()

    # Results structure
    layer_profiles: dict[int, dict] = checkpoint_data.get("layerProfiles", {})
    # Convert string keys back to int
    layer_profiles = {int(k): v for k, v in layer_profiles.items()}

    # Process each layer
    for layer_idx in all_layers:
        if layer_idx in completed_layers:
            logger.info("Layer %d already complete, skipping", layer_idx)
            continue

        logger.info("Processing layer %d/%d...", layer_idx + 1, num_layers)

        try:
            layer_result = analyzer.analyze_layer(probes, provider, layer_idx, config)

            # Store results
            layer_profiles[layer_idx] = {
                "layer": layer_idx,
                "totalConcepts": len(layer_result.concept_densities),
                "meanDensity": layer_result.mean_density,
                "medianDensity": layer_result.median_density,
                "sparseCount": layer_result.sparse_concept_count,
                "denseCount": layer_result.dense_concept_count,
                "densityThreshold": layer_result.density_threshold,
                "concepts": [
                    {
                        "probeID": c.probe_id,
                        "name": c.name,
                        "domain": c.domain,
                        "densityScore": c.density_score,
                        "intrinsicDimension": c.intrinsic_dimension,
                        "activationVariance": c.activation_variance,
                        "clusterTightness": c.cluster_tightness,
                    }
                    for c in layer_result.concept_densities
                ],
            }
            completed_layers.add(layer_idx)

            # Save checkpoint after each layer
            if checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                checkpoint_file = checkpoint_path / "full_profile_checkpoint.json"
                checkpoint_file.write_text(json.dumps({
                    "modelPath": model_path,
                    "completedLayers": sorted(completed_layers),
                    "totalLayers": num_layers,
                    "layerProfiles": {str(k): v for k, v in layer_profiles.items()},
                }, indent=2))
                logger.info("Checkpoint saved: %d/%d layers", len(completed_layers), num_layers)

        except Exception as exc:
            logger.error("Failed to process layer %d: %s", layer_idx, exc)
            # Continue to next layer, don't abort

    # Compute domain summaries across all layers
    domain_summaries: dict[str, dict] = {}
    for layer_idx, lp in layer_profiles.items():
        for concept in lp.get("concepts", []):
            domain = concept["domain"]
            if domain not in domain_summaries:
                domain_summaries[domain] = {
                    "domain": domain,
                    "layerDensities": {},
                    "conceptCount": 0,
                    "totalDensitySum": 0.0,
                }
            if layer_idx not in domain_summaries[domain]["layerDensities"]:
                domain_summaries[domain]["layerDensities"][layer_idx] = {
                    "densities": [],
                    "meanDensity": 0.0,
                }
            domain_summaries[domain]["layerDensities"][layer_idx]["densities"].append(
                concept["densityScore"]
            )
            domain_summaries[domain]["conceptCount"] += 1
            domain_summaries[domain]["totalDensitySum"] += concept["densityScore"]

    # Compute means per domain per layer
    for domain, summary in domain_summaries.items():
        for layer_idx, layer_data in summary["layerDensities"].items():
            densities = layer_data["densities"]
            if densities:
                layer_data["meanDensity"] = sum(densities) / len(densities)
            del layer_data["densities"]  # Don't need raw list in output
        if summary["conceptCount"] > 0:
            summary["overallMeanDensity"] = summary["totalDensitySum"] / summary["conceptCount"]
        else:
            summary["overallMeanDensity"] = 0.0
        del summary["totalDensitySum"]

    # Find strongest/weakest layers per domain
    for domain, summary in domain_summaries.items():
        layer_means = [
            (int(layer_idx), data["meanDensity"])
            for layer_idx, data in summary["layerDensities"].items()
        ]
        if layer_means:
            layer_means.sort(key=lambda x: x[1], reverse=True)
            summary["strongestLayers"] = [lm[0] for lm in layer_means[:5]]
            summary["weakestLayers"] = [lm[0] for lm in layer_means[-5:]]

    # Build final payload
    payload = {
        "_schema": "mc.geometry.research.full_profile.v1",
        "modelPath": model_path,
        "totalLayers": num_layers,
        "totalProbes": len(probes),
        "completedLayers": sorted(completed_layers),
        "domainSummaries": list(domain_summaries.values()),
        "layerProfiles": [
            layer_profiles[i] for i in sorted(layer_profiles.keys())
        ],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(payload, indent=2))
        logger.info("Full profile saved to %s", output_path)

    if context.output_format == "text":
        lines = [
            "FULL MODEL KNOWLEDGE PROFILE",
            f"Model: {model_path}",
            f"Layers: {num_layers}",
            f"Probes: {len(probes)}",
            f"Completed: {len(completed_layers)}/{num_layers} layers",
            "",
            "DOMAIN SUMMARY (strongest → weakest by overall density):",
            "-" * 60,
        ]

        sorted_domains = sorted(
            domain_summaries.values(),
            key=lambda x: x.get("overallMeanDensity", 0),
            reverse=True
        )
        for ds in sorted_domains:
            lines.append(
                f"  {ds['domain']}: mean_density={ds.get('overallMeanDensity', 0):.3f}, "
                f"strongest_layers={ds.get('strongestLayers', [])[:3]}"
            )

        lines.append("")
        lines.append("LAYER-BY-LAYER SUMMARY:")
        lines.append("-" * 60)
        for layer_idx in sorted(layer_profiles.keys()):
            lp = layer_profiles[layer_idx]
            lines.append(
                f"  L{layer_idx}: mean={lp['meanDensity']:.3f}, "
                f"sparse={lp['sparseCount']}, dense={lp['denseCount']}"
            )

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


def _cleanup_memory() -> None:
    """Aggressively clean up memory between model operations.

    This is critical when profiling multiple models sequentially.
    Without cleanup, memory accumulates and can crash the system.
    """
    import gc

    # Force Python garbage collection
    gc.collect()
    gc.collect()  # Second pass catches circular refs

    # Clear MLX cache if available
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Brief pause to let system reclaim memory
    import time

    time.sleep(1)


@app.command("batch-profile")
def batch_profile(
    ctx: typer.Context,
    model_paths: list[str] = typer.Argument(..., help="Paths to model directories"),
    output_dir: str = typer.Option(
        None, "--output-dir", "-o", help="Directory for profile outputs"
    ),
    sources: list[str] | None = typer.Option(
        None, "--source", "-s", help="Filter by atlas source (repeatable)"
    ),
    domains: list[str] | None = typer.Option(
        None, "--domain", "-d", help="Filter by atlas domain (repeatable)"
    ),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", help="k for geodesic distance graph"
    ),
) -> None:
    """Profile multiple models SEQUENTIALLY with automatic resource management.

    IMPORTANT: This command profiles models ONE AT A TIME to maximize resource
    utilization and prevent system crashes. Each model gets full CPU/GPU access.

    Memory is aggressively cleaned between models to prevent accumulation.
    Checkpointing is automatic - interrupted profiles can be resumed.

    Example:
        mc geometry research batch-profile /path/to/model1 /path/to/model2 -o ./profiles

    For uber model experiments:
        mc geometry research batch-profile \\
            /models/Qwen3-8B-4bit \\
            /models/Qwen2.5-Math-7B \\
            /models/Mistral-7B \\
            -o /experiments/profiles
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.knowledge_density import (
        KnowledgeDensityAnalyzer,
        KnowledgeDensityConfig,
    )

    # Resolve output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path.cwd() / "profiles"
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_base = out_path / "checkpoints"
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # Parse filters
    source_filter = _parse_sources(sources)
    domain_filter = _parse_domains(domains)

    # Get probes once (same for all models)
    probes = UnifiedAtlasInventory.all_probes()
    if source_filter:
        probes = [probe for probe in probes if probe.source in source_filter]
    if domain_filter:
        probes = [probe for probe in probes if probe.domain in domain_filter]

    logger.info("Batch profiling %d models with %d probes", len(model_paths), len(probes))

    results: list[dict] = []

    for idx, model_path in enumerate(model_paths, 1):
        model_name = Path(model_path).name
        profile_output = out_path / f"{model_name}.json"
        checkpoint_dir = checkpoint_base / model_name

        logger.info("")
        logger.info("=" * 60)
        logger.info("MODEL %d/%d: %s", idx, len(model_paths), model_name)
        logger.info("=" * 60)

        # Check if already complete
        if profile_output.exists():
            try:
                existing = json.loads(profile_output.read_text())
                completed = len(existing.get("completedLayers", []))
                total = existing.get("totalLayers", 0)
                if completed == total and total > 0:
                    logger.info("Already complete (%d/%d layers)", completed, total)
                    results.append({
                        "model": model_name,
                        "status": "already_complete",
                        "layers": total,
                        "output": str(profile_output),
                    })
                    continue
                logger.info("Resuming from checkpoint (%d/%d layers)", completed, total)
            except Exception:
                pass

        # Clean memory before loading new model
        logger.info("Cleaning memory before model load...")
        _cleanup_memory()

        try:
            # Load model
            logger.info("Loading model: %s", model_path)
            model, tokenizer, backend, provider, num_layers = _load_model_and_provider(
                model_path, k_neighbors
            )

            all_layers = list(range(num_layers))
            logger.info(
                "Full profile: %d probes x %d layers = %d measurements",
                len(probes), num_layers, len(probes) * num_layers
            )

            # Check for checkpoint to resume from
            checkpoint_data: dict = {}
            completed_layers: set[int] = set()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / "full_profile_checkpoint.json"

            if checkpoint_file.exists():
                checkpoint_data = json.loads(checkpoint_file.read_text())
                completed_layers = set(checkpoint_data.get("completedLayers", []))
                logger.info("Resuming from checkpoint: %d layers complete", len(completed_layers))

            analyzer = KnowledgeDensityAnalyzer(backend=backend)
            config = KnowledgeDensityConfig()

            # Results structure
            layer_profiles: dict[int, dict] = checkpoint_data.get("layerProfiles", {})
            layer_profiles = {int(k): v for k, v in layer_profiles.items()}

            # Process each layer
            for layer_idx in all_layers:
                if layer_idx in completed_layers:
                    continue

                logger.info("Processing layer %d/%d...", layer_idx + 1, num_layers)

                try:
                    layer_result = analyzer.analyze_layer(probes, provider, layer_idx, config)

                    layer_profiles[layer_idx] = {
                        "layer": layer_idx,
                        "totalConcepts": len(layer_result.concept_densities),
                        "meanDensity": layer_result.mean_density,
                        "medianDensity": layer_result.median_density,
                        "sparseCount": layer_result.sparse_concept_count,
                        "denseCount": layer_result.dense_concept_count,
                        "densityThreshold": layer_result.density_threshold,
                        "concepts": [
                            {
                                "probeID": c.probe_id,
                                "name": c.name,
                                "domain": c.domain,
                                "densityScore": c.density_score,
                                "intrinsicDimension": c.intrinsic_dimension,
                                "activationVariance": c.activation_variance,
                                "clusterTightness": c.cluster_tightness,
                            }
                            for c in layer_result.concept_densities
                        ],
                    }
                    completed_layers.add(layer_idx)

                    # Save checkpoint after each layer
                    checkpoint_file.write_text(json.dumps({
                        "modelPath": model_path,
                        "completedLayers": sorted(completed_layers),
                        "totalLayers": num_layers,
                        "layerProfiles": {str(k): v for k, v in layer_profiles.items()},
                    }, indent=2))
                    logger.info("Checkpoint saved: %d/%d layers", len(completed_layers), num_layers)

                except Exception as exc:
                    logger.error("Failed to process layer %d: %s", layer_idx, exc)

            # Compute domain summaries
            domain_summaries: dict[str, dict] = {}
            for layer_idx, lp in layer_profiles.items():
                for concept in lp.get("concepts", []):
                    domain = concept["domain"]
                    if domain not in domain_summaries:
                        domain_summaries[domain] = {
                            "domain": domain,
                            "layerDensities": {},
                            "conceptCount": 0,
                            "totalDensitySum": 0.0,
                        }
                    if layer_idx not in domain_summaries[domain]["layerDensities"]:
                        domain_summaries[domain]["layerDensities"][layer_idx] = {
                            "densities": [],
                            "meanDensity": 0.0,
                        }
                    domain_summaries[domain]["layerDensities"][layer_idx]["densities"].append(
                        concept["densityScore"]
                    )
                    domain_summaries[domain]["conceptCount"] += 1
                    domain_summaries[domain]["totalDensitySum"] += concept["densityScore"]

            for domain, summary in domain_summaries.items():
                for layer_idx, layer_data in summary["layerDensities"].items():
                    densities = layer_data["densities"]
                    if densities:
                        layer_data["meanDensity"] = sum(densities) / len(densities)
                    del layer_data["densities"]
                if summary["conceptCount"] > 0:
                    summary["overallMeanDensity"] = summary["totalDensitySum"] / summary["conceptCount"]
                else:
                    summary["overallMeanDensity"] = 0.0
                del summary["totalDensitySum"]

            # Build final payload
            final_payload = {
                "_schema": "mc.geometry.research.full_profile.v1",
                "modelPath": model_path,
                "totalLayers": num_layers,
                "totalProbes": len(probes),
                "completedLayers": sorted(completed_layers),
                "domainSummaries": list(domain_summaries.values()),
                "layerProfiles": [
                    layer_profiles[i] for i in sorted(layer_profiles.keys())
                ],
            }

            profile_output.write_text(json.dumps(final_payload, indent=2))
            logger.info("Profile saved to %s", profile_output)

            results.append({
                "model": model_name,
                "status": "complete",
                "layers": num_layers,
                "output": str(profile_output),
            })

        except Exception as exc:
            logger.error("Failed to profile %s: %s", model_name, exc)
            results.append({
                "model": model_name,
                "status": "failed",
                "error": str(exc),
            })

        # Clean memory after each model (CRITICAL)
        logger.info("Cleaning memory after model completion...")
        _cleanup_memory()

    # Final output
    summary = {
        "_schema": "mc.geometry.research.batch_profile.v1",
        "outputDir": str(out_path),
        "totalModels": len(model_paths),
        "completedModels": sum(1 for r in results if r.get("status") in ("complete", "already_complete")),
        "failedModels": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }

    if context.output_format == "text":
        lines = [
            "",
            "=" * 60,
            "BATCH PROFILING COMPLETE",
            "=" * 60,
            f"Output directory: {out_path}",
            f"Models processed: {len(model_paths)}",
            f"Completed: {summary['completedModels']}",
            f"Failed: {summary['failedModels']}",
            "",
            "Results:",
        ]
        for r in results:
            status_icon = "[done]" if r.get("status") in ("complete", "already_complete") else "[fail]"
            lines.append(f"  {status_icon} {r['model']}: {r.get('status', 'unknown')}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(summary, context.output_format, context.pretty)
