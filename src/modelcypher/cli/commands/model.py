"""Model management CLI commands.

Provides commands for:
- Model listing, registration, deletion, fetching
- Model search via HuggingFace Hub
- Model probing for architecture details
- Model merge operations and validation
- Alignment analysis between models

Commands:
    mc model list
    mc model register <alias> --path <path>
    mc model merge --source <model> --target <model>
    mc model search <query>
    mc model probe <path>
"""

from __future__ import annotations

import json
from typing import Optional

import typer

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.presenters import model_payload, model_search_payload
from modelcypher.core.domain.model_search import (
    MemoryFitStatus,
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchPage,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.core.use_cases.model_merge_service import ModelMergeService
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
from modelcypher.core.use_cases.model_search_service import ModelSearchService
from modelcypher.core.use_cases.model_service import ModelService
from modelcypher.core.use_cases.invariant_layer_mapping_service import (
    InvariantLayerMappingService,
    LayerMappingConfig,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("list")
def model_list(ctx: typer.Context) -> None:
    """List all registered models."""
    context = _context(ctx)
    service = ModelService()
    models = [model_payload(model) for model in service.list_models()]
    write_output(models, context.output_format, context.pretty)


@app.command("register")
def model_register(
    ctx: typer.Context,
    alias: str = typer.Argument(...),
    path: str = typer.Option(..., "--path"),
    architecture: str = typer.Option(..., "--architecture"),
    parameters: Optional[int] = typer.Option(None, "--parameters"),
    default_chat: bool = typer.Option(False, "--default-chat"),
) -> None:
    """Register a local model.

    Examples:
        mc model register my-llama --path ./models/llama --architecture llama
    """
    context = _context(ctx)
    service = ModelService()
    service.register_model(alias, path, architecture, parameters=parameters, default_chat=default_chat)
    write_output({"registered": alias}, context.output_format, context.pretty)


@app.command("merge")
def model_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source"),
    target: str = typer.Option(..., "--target"),
    output_dir: str = typer.Option(..., "--output-dir"),
    alpha: float = typer.Option(0.5, "--alpha"),
    rank: int = typer.Option(32, "--rank"),
    module_scope: Optional[str] = typer.Option(None, "--module-scope"),
    anchor_mode: str = typer.Option("semantic-primes", "--anchor-mode"),
    intersection: Optional[str] = typer.Option(None, "--intersection"),
    fisher_source: Optional[str] = typer.Option(None, "--fisher-source"),
    fisher_target: Optional[str] = typer.Option(None, "--fisher-target"),
    fisher_strength: float = typer.Option(0.0, "--fisher-strength"),
    fisher_epsilon: float = typer.Option(1e-6, "--fisher-epsilon"),
    adaptive_alpha: bool = typer.Option(False, "--adaptive-alpha"),
    source_crm: Optional[str] = typer.Option(None, "--source-crm"),
    target_crm: Optional[str] = typer.Option(None, "--target-crm"),
    transition_gate_strength: float = typer.Option(0.0, "--transition-gate-strength"),
    transition_gate_min_ratio: float = typer.Option(0.7, "--transition-gate-min-ratio"),
    transition_gate_max_ratio: float = typer.Option(1.3, "--transition-gate-max-ratio"),
    consistency_gate_strength: float = typer.Option(0.0, "--consistency-gate-strength"),
    consistency_gate_layer_samples: int = typer.Option(6, "--consistency-gate-layer-samples"),
    shared_subspace: bool = typer.Option(False, "--shared-subspace"),
    shared_subspace_method: str = typer.Option("cca", "--shared-subspace-method"),
    shared_subspace_blend: Optional[float] = typer.Option(None, "--shared-subspace-blend"),
    shared_subspace_per_layer: bool = typer.Option(
        True,
        "--shared-subspace-per-layer/--no-shared-subspace-per-layer",
    ),
    shared_subspace_anchor_prefixes: Optional[str] = typer.Option(
        None,
        "--shared-subspace-anchor-prefixes",
    ),
    shared_subspace_anchor_weights: Optional[str] = typer.Option(
        None,
        "--shared-subspace-anchor-weights",
    ),
    shared_subspace_pca_mode: Optional[str] = typer.Option(
        None,
        "--shared-subspace-pca-mode",
    ),
    shared_subspace_pca_variance: Optional[float] = typer.Option(
        None,
        "--shared-subspace-pca-variance",
    ),
    shared_subspace_variance_threshold: Optional[float] = typer.Option(
        None,
        "--shared-subspace-variance-threshold",
    ),
    shared_subspace_min_correlation: Optional[float] = typer.Option(
        None,
        "--shared-subspace-min-correlation",
    ),
    transport_guided: bool = typer.Option(False, "--use-transport-guided"),
    transport_coupling_threshold: float = typer.Option(0.001, "--transport-coupling-threshold"),
    transport_blend_alpha: float = typer.Option(0.5, "--transport-blend-alpha"),
    transport_min_samples: int = typer.Option(5, "--transport-min-samples"),
    transport_max_samples: int = typer.Option(32, "--transport-max-samples"),
    use_layer_mapping: bool = typer.Option(
        False,
        "--use-layer-mapping",
        help="Run multi-atlas layer mapping to compute per-layer adaptive alpha (enables --adaptive-alpha)",
    ),
    layer_mapping_scope: str = typer.Option(
        "multiAtlas",
        "--layer-mapping-scope",
        help="Invariant scope for layer mapping: sequenceInvariants, multiAtlas",
    ),
    merge_method: str = typer.Option(
        "rotational",
        "--merge-method",
        help="Merge method: rotational (Procrustes alignment, may corrupt output) or linear (simple interpolation, preserves structure)",
    ),
    dimension_blending: bool = typer.Option(
        False,
        "--dimension-blending",
        help="Enable per-dimension alpha blending (requires --use-layer-mapping and --merge-method linear)",
    ),
    verb_noun_blending: bool = typer.Option(
        False,
        "--verb-noun",
        help="Use VerbNoun dimension classification instead of domain-based (skill vs knowledge)",
    ),
    verb_noun_strength: float = typer.Option(
        0.7,
        "--verb-noun-strength",
        help="VerbNoun modulation strength (0=ignore, 1=full effect)",
    ),
    output_quant: Optional[str] = typer.Option(None, "--output-quant"),
    output_quant_group_size: Optional[int] = typer.Option(None, "--output-quant-group-size"),
    output_quant_mode: Optional[str] = typer.Option(None, "--output-quant-mode"),
    verbose: bool = typer.Option(False, "--verbose"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_path: Optional[str] = typer.Option(None, "--report-path"),
) -> None:
    """Merge two models using geometric alignment.

    Examples:
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged --alpha 0.7
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged --use-layer-mapping
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged --use-layer-mapping --merge-method linear --dimension-blending
    """
    import tempfile
    from pathlib import Path as PathLib

    context = _context(ctx)

    # If using layer mapping, run multi-atlas layer mapping first
    effective_intersection = intersection
    effective_adaptive_alpha = adaptive_alpha
    computed_alpha_by_layer: dict[int, float] | None = None
    computed_alpha_vectors: dict[int, "np.ndarray"] | None = None

    if use_layer_mapping:
        typer.echo("Running multi-atlas layer mapping...", err=True)
        layer_mapping_service = InvariantLayerMappingService()

        layer_config = LayerMappingConfig(
            source_model_path=source,
            target_model_path=target,
            invariant_scope=layer_mapping_scope,
            use_triangulation=True,
            collapse_threshold=0.35,
            sample_layer_count=12,
        )

        try:
            layer_result = layer_mapping_service.map_layers(layer_config)
            typer.echo(
                f"Layer mapping complete: {layer_result.report.summary.mapped_layers} layers mapped, "
                f"alignment quality {layer_result.report.summary.alignment_quality:.3f}",
                err=True,
            )

            # Convert to intersection map
            intersection_map = InvariantLayerMappingService.to_intersection_map(layer_result)

            # Compute and display per-layer alpha
            computed_alpha_by_layer = InvariantLayerMappingService.alpha_by_layer(layer_result, alpha)
            typer.echo("Per-layer adaptive alpha:", err=True)
            for layer_idx in sorted(computed_alpha_by_layer.keys())[:10]:
                layer_alpha = computed_alpha_by_layer[layer_idx]
                typer.echo(f"  Layer {layer_idx}: alpha={layer_alpha:.3f}", err=True)
            if len(computed_alpha_by_layer) > 10:
                typer.echo(f"  ... and {len(computed_alpha_by_layer) - 10} more layers", err=True)

            # Compute per-dimension alpha vectors
            if merge_method.lower() == "linear" and (dimension_blending or verb_noun_blending):
                typer.echo("Computing per-dimension alpha vectors...", err=True)
                try:
                    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

                    # Get probes for the scope
                    probes = UnifiedAtlasInventory.all_probes()

                    # Load fingerprints (this re-runs probe extraction - could be optimized)
                    from modelcypher.core.domain.geometry.invariant_layer_mapper import (
                        Config as MapperConfig,
                        InvariantScope,
                    )
                    mapper_config = MapperConfig(
                        invariant_scope=InvariantScope.MULTI_ATLAS if layer_mapping_scope == "multiAtlas" else InvariantScope.SEQUENCE_INVARIANTS,
                    )
                    source_fps = layer_mapping_service._load_fingerprints(source, mapper_config)
                    target_fps = layer_mapping_service._load_fingerprints(target, mapper_config)

                    if verb_noun_blending:
                        # Use VerbNoun classification (skill vs knowledge dimensions)
                        typer.echo("Using VerbNoun classification (skill vs knowledge)...", err=True)
                        from modelcypher.core.domain.geometry.verb_noun_classifier import (
                            VerbNounDimensionClassifier,
                            VerbNounConfig,
                            get_prime_probe_ids,
                            get_gate_probe_ids,
                        )

                        # Get prime and gate probe IDs
                        prime_ids = get_prime_probe_ids()
                        gate_ids = get_gate_probe_ids()
                        typer.echo(f"  Prime probes: {len(prime_ids)}, Gate probes: {len(gate_ids)}", err=True)

                        # Get layer indices from the layer result
                        layer_indices = list(layer_result.report.source_sample_layers)
                        hidden_dim = 2048  # TODO: get from model config

                        # Convert fingerprints to dicts
                        source_fp_dicts = InvariantLayerMappingService.fingerprints_to_dicts(source_fps)
                        target_fp_dicts = InvariantLayerMappingService.fingerprints_to_dicts(target_fps)

                        # Classify using VerbNoun (use source fingerprints for classification)
                        vn_config = VerbNounConfig.default()
                        vn_result = VerbNounDimensionClassifier.classify_from_fingerprints(
                            source_fp_dicts,
                            prime_ids,
                            gate_ids,
                            layer_indices,
                            hidden_dim,
                            vn_config,
                        )

                        computed_alpha_vectors = vn_result.alpha_vectors_by_layer

                        # Display VerbNoun summary
                        typer.echo(f"VerbNoun classification: {len(vn_result.layer_classifications)} layers analyzed", err=True)
                        typer.echo(f"  Mean verb fraction: {vn_result.mean_verb_fraction:.1%}", err=True)
                        typer.echo(f"  Mean noun fraction: {vn_result.mean_noun_fraction:.1%}", err=True)

                        for layer_idx, classification in list(vn_result.layer_classifications.items())[:3]:
                            typer.echo(
                                f"  Layer {layer_idx}: {classification.verb_count} verb (α=0.2), "
                                f"{classification.noun_count} noun (α=0.8), {classification.mixed_count} mixed",
                                err=True,
                            )
                        if len(vn_result.layer_classifications) > 3:
                            typer.echo(f"  ... and {len(vn_result.layer_classifications) - 3} more layers", err=True)

                    else:
                        # Use domain-based classification
                        typer.echo("Using domain-based classification...", err=True)

                        # Compute dimension profiles and alpha vectors
                        source_profiles, target_profiles = InvariantLayerMappingService.compute_dimension_profiles(
                            source_fps, target_fps, probes
                        )

                        computed_alpha_vectors = InvariantLayerMappingService.compute_dimension_alpha_vectors(
                            source_profiles, target_profiles, merge_direction="instruct_to_coder"
                        )

                        # Display summary
                        from modelcypher.core.domain.geometry.dimension_blender import DimensionBlender
                        summary = DimensionBlender.summarize_profiles(source_profiles)
                        typer.echo(f"Dimension profiles: {summary['layer_count']} layers analyzed", err=True)
                        for layer_idx, layer_info in list(summary["layers"].items())[:3]:
                            dist = layer_info.get("domain_distribution", {})
                            top_domains = sorted(dist.items(), key=lambda x: -x[1])[:3]
                            domain_str = ", ".join(f"{d}:{c}" for d, c in top_domains)
                            typer.echo(f"  Layer {layer_idx}: {layer_info['classified_count']}/{layer_info['dimension_count']} dims classified ({domain_str})", err=True)
                        if len(summary["layers"]) > 3:
                            typer.echo(f"  ... and {len(summary['layers']) - 3} more layers", err=True)

                except Exception as e:
                    import traceback
                    typer.echo(f"Dimension blending failed: {e}", err=True)
                    typer.echo(traceback.format_exc(), err=True)
                    typer.echo("Falling back to per-layer alpha...", err=True)
                    computed_alpha_vectors = None

            # Save intersection map to temp file
            intersection_payload = InvariantLayerMappingService.intersection_map_payload(intersection_map)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(intersection_payload, f, indent=2)
                effective_intersection = f.name
                typer.echo(f"Intersection map saved to: {effective_intersection}", err=True)

            # Enable adaptive alpha when using layer mapping
            effective_adaptive_alpha = True

        except Exception as e:
            typer.echo(f"Layer mapping failed: {e}", err=True)
            typer.echo("Proceeding with standard merge...", err=True)

    service = ModelMergeService(FileSystemStore())
    report = service.merge(
        source_id=source,
        target_id=target,
        output_dir=output_dir,
        alpha=alpha,
        alignment_rank=rank,
        module_scope=module_scope,
        anchor_mode=anchor_mode,
        intersection_path=effective_intersection,
        fisher_source=fisher_source,
        fisher_target=fisher_target,
        fisher_strength=fisher_strength,
        fisher_epsilon=fisher_epsilon,
        adaptive_alpha=effective_adaptive_alpha,
        source_crm=source_crm,
        target_crm=target_crm,
        transition_gate_strength=transition_gate_strength,
        transition_gate_min_ratio=transition_gate_min_ratio,
        transition_gate_max_ratio=transition_gate_max_ratio,
        consistency_gate_strength=consistency_gate_strength,
        consistency_gate_layer_samples=consistency_gate_layer_samples,
        shared_subspace=shared_subspace,
        shared_subspace_method=shared_subspace_method,
        shared_subspace_blend=shared_subspace_blend,
        shared_subspace_per_layer=shared_subspace_per_layer,
        shared_subspace_anchor_prefixes=shared_subspace_anchor_prefixes,
        shared_subspace_anchor_weights=shared_subspace_anchor_weights,
        shared_subspace_pca_mode=shared_subspace_pca_mode,
        shared_subspace_pca_variance=shared_subspace_pca_variance,
        shared_subspace_variance_threshold=shared_subspace_variance_threshold,
        shared_subspace_min_correlation=shared_subspace_min_correlation,
        transport_guided=transport_guided,
        transport_coupling_threshold=transport_coupling_threshold,
        transport_blend_alpha=transport_blend_alpha,
        transport_min_samples=transport_min_samples,
        transport_max_samples=transport_max_samples,
        dry_run=dry_run,
        output_quant=output_quant,
        output_quant_group_size=output_quant_group_size,
        output_quant_mode=output_quant_mode,
        merge_method=merge_method,
        alpha_by_layer=computed_alpha_by_layer,
        alpha_vectors=computed_alpha_vectors,
    )
    if report_path:
        from pathlib import Path

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_output(report, context.output_format, context.pretty)


@app.command("geometric-merge")
def model_geometric_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source model path"),
    target: str = typer.Option(..., "--target", help="Target model path"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output directory for merged model"),
    alpha: float = typer.Option(0.5, "--alpha", help="Base alpha (0=target, 1=source)"),
    smoothing_window: int = typer.Option(2, "--smoothing-window", help="Gaussian smoothing window size"),
    smoothing_sigma: float = typer.Option(1.0, "--smoothing-sigma", help="Gaussian smoothing sigma"),
    spectral_penalty: float = typer.Option(0.5, "--spectral-penalty", help="Spectral penalty strength"),
    use_svd_blending: bool = typer.Option(True, "--svd-blending/--no-svd-blending", help="Enable SVD-aware blending"),
    svd_rank_ratio: float = typer.Option(0.1, "--svd-rank-ratio", help="Fraction of singular values for high-rank"),
    high_rank_alpha: float = typer.Option(0.3, "--high-rank-alpha", help="Alpha for high-rank components (skills)"),
    low_rank_alpha: float = typer.Option(0.7, "--low-rank-alpha", help="Alpha for low-rank components (structure)"),
    use_correlation_weights: bool = typer.Option(True, "--correlation-weights/--no-correlation-weights", help="Enable correlation-based dimension weighting"),
    correlation_scale: float = typer.Option(5.0, "--correlation-scale", help="Scale factor for correlation weighting"),
    stability_alpha: float = typer.Option(0.7, "--stability-alpha", help="Alpha for low-correlation dimensions"),
    use_verb_noun: bool = typer.Option(True, "--verb-noun/--no-verb-noun", help="Enable VerbNoun modulation"),
    verb_noun_strength: float = typer.Option(0.7, "--verb-noun-strength", help="VerbNoun modulation strength"),
    output_quant: Optional[str] = typer.Option(None, "--output-quant", help="Output quantization (4bit, 8bit)"),
    output_quant_group_size: Optional[int] = typer.Option(None, "--output-quant-group-size"),
    output_quant_mode: Optional[str] = typer.Option(None, "--output-quant-mode"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without saving"),
    report_path: Optional[str] = typer.Option(None, "--report-path", help="Path to save merge report"),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Use preset config: default, skill-preserving, structure-preserving",
    ),
) -> None:
    """Merge models using the full geometric pipeline.

    The geometric merge applies:
    1. Gaussian alpha smoothing across layers (prevents tearing)
    2. Spectral penalty for ill-conditioned weights (stabilizes merge)
    3. SVD-aware blending (different alpha for skills vs structure)
    4. Correlation-based dimension weighting (respects relationships)
    5. VerbNoun modulation (subtle skill/knowledge adjustment)

    Examples:
        mc model geometric-merge --source ./instruct --target ./coder --output-dir ./merged
        mc model geometric-merge --source ./instruct --target ./coder --output-dir ./merged --alpha 0.4
        mc model geometric-merge --source ./instruct --target ./coder --output-dir ./merged --preset skill-preserving
    """
    from modelcypher.core.use_cases.model_merge_service import GeometricMergeConfig

    context = _context(ctx)

    # Build config from preset or individual options
    if preset:
        preset_lower = preset.lower().replace("-", "_")
        if preset_lower == "skill_preserving":
            config = GeometricMergeConfig.skill_preserving()
        elif preset_lower == "structure_preserving":
            config = GeometricMergeConfig.structure_preserving()
        else:
            config = GeometricMergeConfig.default()

        # Override base_alpha if explicitly provided
        if alpha != 0.5:
            config = GeometricMergeConfig(
                base_alpha=alpha,
                smoothing_window=config.smoothing_window,
                smoothing_sigma=config.smoothing_sigma,
                spectral_penalty_strength=config.spectral_penalty_strength,
                use_svd_blending=config.use_svd_blending,
                svd_rank_ratio=config.svd_rank_ratio,
                high_rank_alpha=config.high_rank_alpha,
                low_rank_alpha=config.low_rank_alpha,
                use_correlation_weights=config.use_correlation_weights,
                correlation_scale=config.correlation_scale,
                stability_alpha=config.stability_alpha,
                use_verb_noun=config.use_verb_noun,
                verb_noun_strength=config.verb_noun_strength,
            )
    else:
        config = GeometricMergeConfig(
            base_alpha=alpha,
            smoothing_window=smoothing_window,
            smoothing_sigma=smoothing_sigma,
            spectral_penalty_strength=spectral_penalty,
            use_svd_blending=use_svd_blending,
            svd_rank_ratio=svd_rank_ratio,
            high_rank_alpha=high_rank_alpha,
            low_rank_alpha=low_rank_alpha,
            use_correlation_weights=use_correlation_weights,
            correlation_scale=correlation_scale,
            stability_alpha=stability_alpha,
            use_verb_noun=use_verb_noun,
            verb_noun_strength=verb_noun_strength,
        )

    typer.echo("Starting geometric merge...", err=True)
    typer.echo(f"  Source: {source}", err=True)
    typer.echo(f"  Target: {target}", err=True)
    typer.echo(f"  Base alpha: {config.base_alpha}", err=True)
    typer.echo(f"  SVD blending: {config.use_svd_blending}", err=True)
    typer.echo(f"  VerbNoun modulation: {config.use_verb_noun} (strength={config.verb_noun_strength})", err=True)

    service = ModelMergeService(FileSystemStore())
    try:
        report = service.geometric_merge(
            source_id=source,
            target_id=target,
            output_dir=output_dir,
            config=config,
            dry_run=dry_run,
            output_quant=output_quant,
            output_quant_group_size=output_quant_group_size,
            output_quant_mode=output_quant_mode,
        )
    except Exception as e:
        error = ErrorDetail(
            code="MC-1010",
            title="Geometric merge failed",
            detail=str(e),
            hint="Check model paths and ensure both models have compatible architectures",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Display summary
    typer.echo(f"\nGeometric merge complete!", err=True)
    typer.echo(f"  Layers: {report.get('layerCount', 0)}", err=True)
    typer.echo(f"  Weights: {report.get('weightCount', 0)}", err=True)

    spectral = report.get("spectralAnalysis", {})
    if spectral:
        typer.echo(f"  Spectral confidence: {spectral.get('mean_confidence', 0):.3f}", err=True)
        typer.echo(f"  Ill-conditioned: {spectral.get('ill_conditioned_count', 0)}", err=True)

    svd = report.get("svdAnalysis", {})
    if svd:
        typer.echo(f"  Mean effective rank: {svd.get('mean_effective_rank', 0):.1f}", err=True)

    if report.get("outputPath"):
        typer.echo(f"  Output: {report['outputPath']}", err=True)

    if report_path:
        from pathlib import Path

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"  Report: {report_path}", err=True)

    write_output(report, context.output_format, context.pretty)


@app.command("unified-merge")
def model_unified_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source model path (skill donor)"),
    target: str = typer.Option(..., "--target", help="Target model path (knowledge base)"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output directory for merged model"),
    alpha: float = typer.Option(0.5, "--alpha", help="Base alpha (0=target, 1=source)"),
    # Probe stage
    intersection_mode: str = typer.Option("ensemble", "--intersection-mode", help="Intersection mode: jaccard, cka, ensemble"),
    # Permute stage
    enable_permutation: bool = typer.Option(True, "--permutation/--no-permutation", help="Enable permutation alignment"),
    # Rotate stage
    enable_rotation: bool = typer.Option(True, "--rotation/--no-rotation", help="Enable Procrustes rotation"),
    alignment_rank: int = typer.Option(32, "--alignment-rank", help="SVD rank for rotation computation"),
    use_transport: bool = typer.Option(False, "--transport/--no-transport", help="Use transport-guided (Gromov-Wasserstein) merging"),
    # Blend stage
    smoothing_window: int = typer.Option(2, "--smoothing-window", help="Gaussian smoothing window size"),
    spectral_penalty: float = typer.Option(0.5, "--spectral-penalty", help="Spectral penalty strength"),
    use_svd_blending: bool = typer.Option(True, "--svd-blending/--no-svd-blending", help="Enable SVD-aware blending"),
    use_correlation_weights: bool = typer.Option(True, "--correlation-weights/--no-correlation-weights", help="Enable correlation-based weighting"),
    use_verb_noun: bool = typer.Option(True, "--verb-noun/--no-verb-noun", help="Enable VerbNoun modulation"),
    verb_noun_strength: float = typer.Option(0.7, "--verb-noun-strength", help="VerbNoun modulation strength"),
    # Propagate stage
    enable_zipper: bool = typer.Option(True, "--zipper/--no-zipper", help="Enable zipper propagation"),
    # Output
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without saving"),
    report_path: Optional[str] = typer.Option(None, "--report-path", help="Path to save merge report"),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Use preset config: default, conservative, aggressive",
    ),
) -> None:
    """Execute unified geometric merge (THE ONE merge pipeline).

    This is the unified merge that combines ALL geometric techniques in the correct order:

    1. PROBE: Build intersection map from semantic fingerprints
    2. PERMUTE: Align MLP neurons using Re-Basin
    3. ROTATE: Apply Procrustes geometric alignment
    4. BLEND: Multi-stage alpha with spectral, SVD, correlation, and VerbNoun adjustments
    5. PROPAGATE: Carry rotations layer-to-layer (zipper)

    Examples:
        mc model unified-merge --source ./instruct --target ./coder --output-dir ./merged
        mc model unified-merge --source ./instruct --target ./coder --output-dir ./merged --alpha 0.4
        mc model unified-merge --source ./instruct --target ./coder --output-dir ./merged --preset aggressive
    """
    from modelcypher.core.use_cases.unified_geometric_merge import (
        UnifiedMergeConfig,
        unified_merge,
    )

    context = _context(ctx)

    # Build config from preset or individual options
    if preset:
        preset_lower = preset.lower()
        if preset_lower == "conservative":
            config = UnifiedMergeConfig.conservative()
        elif preset_lower == "aggressive":
            config = UnifiedMergeConfig.aggressive()
        else:
            config = UnifiedMergeConfig.default()

        # Override base_alpha if explicitly provided
        if alpha != 0.5:
            # Create new config with overridden alpha
            config = UnifiedMergeConfig(
                base_alpha=alpha,
                enable_permutation=config.enable_permutation,
                enable_rotation=config.enable_rotation,
                alignment_rank=config.alignment_rank,
                use_transport_guided=config.use_transport_guided,
                enable_alpha_smoothing=config.enable_alpha_smoothing,
                smoothing_window=config.smoothing_window,
                enable_spectral_penalty=config.enable_spectral_penalty,
                spectral_penalty_strength=config.spectral_penalty_strength,
                enable_svd_blending=config.enable_svd_blending,
                enable_correlation_weights=config.enable_correlation_weights,
                enable_verb_noun=config.enable_verb_noun,
                verb_noun_strength=config.verb_noun_strength,
                enable_zipper=config.enable_zipper,
            )
    else:
        config = UnifiedMergeConfig(
            base_alpha=alpha,
            intersection_mode=intersection_mode,
            enable_permutation=enable_permutation,
            enable_rotation=enable_rotation,
            alignment_rank=alignment_rank,
            use_transport_guided=use_transport,
            smoothing_window=smoothing_window,
            spectral_penalty_strength=spectral_penalty,
            enable_svd_blending=use_svd_blending,
            enable_correlation_weights=use_correlation_weights,
            enable_verb_noun=use_verb_noun,
            verb_noun_strength=verb_noun_strength,
            enable_zipper=enable_zipper,
        )

    typer.echo("=== UNIFIED GEOMETRIC MERGE ===", err=True)
    typer.echo(f"Source (skill donor): {source}", err=True)
    typer.echo(f"Target (knowledge base): {target}", err=True)
    typer.echo(f"Base alpha: {config.base_alpha}", err=True)
    alignment_mode = "GW transport" if config.use_transport_guided else f"Procrustes (rank={config.alignment_rank})"
    typer.echo(f"Alignment: {alignment_mode}", err=True)
    typer.echo(f"Permutation: {config.enable_permutation}", err=True)
    typer.echo(f"SVD blending: {config.enable_svd_blending}", err=True)
    typer.echo(f"Correlation weights: {config.enable_correlation_weights}", err=True)
    typer.echo(f"VerbNoun: {config.enable_verb_noun} (strength={config.verb_noun_strength})", err=True)
    typer.echo(f"Zipper: {config.enable_zipper}", err=True)

    try:
        result = unified_merge(
            source=source,
            target=target,
            output_dir=output_dir,
            config=config,
            dry_run=dry_run,
        )
    except Exception as e:
        error = ErrorDetail(
            code="MC-1020",
            title="Unified merge failed",
            detail=str(e),
            hint="Check model paths and ensure both models have compatible architectures",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Build report
    report = {
        "layerCount": result.layer_count,
        "weightCount": result.weight_count,
        "meanConfidence": result.mean_confidence,
        "meanProcrustesError": result.mean_procrustes_error,
        "outputPath": result.output_path,
        "probeMetrics": result.probe_metrics,
        "permuteMetrics": result.permute_metrics,
        "rotateMetrics": result.rotate_metrics,
        "blendMetrics": result.blend_metrics,
        "timestamp": result.timestamp.isoformat(),
    }

    # Display summary
    typer.echo(f"\nUnified merge complete!", err=True)
    typer.echo(f"  Layers: {result.layer_count}", err=True)
    typer.echo(f"  Weights: {result.weight_count}", err=True)
    typer.echo(f"  Mean confidence: {result.mean_confidence:.3f}", err=True)
    typer.echo(f"  Mean Procrustes error: {result.mean_procrustes_error:.4f}", err=True)

    rotate = result.rotate_metrics
    if rotate:
        procrustes = rotate.get('rotations_applied', 0)
        transport = rotate.get('transport_guided_applied', 0)
        if transport > 0:
            typer.echo(f"  Transport-guided: {transport} (GW distance={rotate.get('mean_gw_distance', 0):.4f})", err=True)
        if procrustes > 0:
            typer.echo(f"  Procrustes rotations: {procrustes}", err=True)

    blend = result.blend_metrics
    if blend:
        typer.echo(f"  SVD blended: {blend.get('svd_blended', 0)}", err=True)
        typer.echo(f"  Correlation weighted: {blend.get('correlation_weighted', 0)}", err=True)
        typer.echo(f"  VerbNoun modulated: {blend.get('verb_noun_modulated', 0)}", err=True)

    if result.output_path:
        typer.echo(f"  Output: {result.output_path}", err=True)

    if report_path:
        from pathlib import Path

        Path(report_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        typer.echo(f"  Report: {report_path}", err=True)

    write_output(report, context.output_format, context.pretty)


@app.command("delete")
def model_delete(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    """Delete a registered model.

    Examples:
        mc model delete my-llama
    """
    context = _context(ctx)
    service = ModelService()
    service.delete_model(model_id)
    write_output({"deleted": model_id}, context.output_format, context.pretty)


@app.command("fetch")
def model_fetch(
    ctx: typer.Context,
    repo_id: str = typer.Argument(...),
    revision: str = typer.Option("main", "--revision"),
    auto_register: bool = typer.Option(False, "--auto-register"),
    alias: Optional[str] = typer.Option(None, "--alias"),
    architecture: Optional[str] = typer.Option(None, "--architecture"),
) -> None:
    """Fetch a model from HuggingFace Hub.

    Examples:
        mc model fetch mlx-community/Llama-2-7b-mlx
        mc model fetch mlx-community/Llama-2-7b-mlx --auto-register --alias my-llama
    """
    context = _context(ctx)
    service = ModelService()
    result = service.fetch_model(repo_id, revision, auto_register, alias, architecture)
    write_output(result, context.output_format, context.pretty)


@app.command("search")
def model_search(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None),
    author: Optional[str] = typer.Option(None, "--author"),
    library: str = typer.Option("mlx", "--library"),
    quant: Optional[str] = typer.Option(None, "--quant"),
    sort: str = typer.Option("downloads", "--sort"),
    limit: int = typer.Option(20, "--limit"),
    cursor: Optional[str] = typer.Option(None, "--cursor"),
) -> None:
    """Search for models on HuggingFace Hub.

    Examples:
        mc model search llama
        mc model search llama --library mlx --quant 4bit
        mc model search --author mlx-community --sort downloads
    """
    context = _context(ctx)
    library_filter = _parse_model_search_library(library)
    quant_filter = _parse_model_search_quant(quant)
    sort_option = _parse_model_search_sort(sort)

    filters = ModelSearchFilters(
        query=query,
        architecture=None,
        max_size_gb=None,
        author=author,
        library=library_filter,
        quantization=quant_filter,
        sort_by=sort_option,
        limit=limit,
    )

    service = ModelSearchService()
    try:
        page = service.search(filters, cursor)
    except ModelSearchError as exc:
        error = ErrorDetail(
            code="MC-5002",
            title="Model search failed",
            detail=str(exc),
            hint="Check your network connection. For private models, set HF_TOKEN environment variable.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        _print_model_search_text(page)
        return

    write_output(model_search_payload(page), context.output_format, context.pretty)


@app.command("probe")
def model_probe(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Probe a model for architecture details.

    Examples:
        mc model probe ./models/llama-7b
    """
    context = _context(ctx)
    service = ModelProbeService()
    try:
        result = service.probe(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1001",
            title="Model probe failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "architecture": result.architecture,
        "parameterCount": result.parameter_count,
        "vocabSize": result.vocab_size,
        "hiddenSize": result.hidden_size,
        "numAttentionHeads": result.num_attention_heads,
        "quantization": result.quantization,
        "layerCount": len(result.layers),
        "layers": [
            {
                "name": layer.name,
                "type": layer.type,
                "parameters": layer.parameters,
                "shape": layer.shape,
            }
            for layer in result.layers[:20]  # Limit to first 20 layers for readability
        ],
    }

    if context.output_format == "text":
        lines = [
            "MODEL PROBE",
            f"Architecture: {result.architecture}",
            f"Parameters: {result.parameter_count:,}",
            f"Vocab Size: {result.vocab_size:,}",
            f"Hidden Size: {result.hidden_size}",
            f"Attention Heads: {result.num_attention_heads}",
            f"Layers: {len(result.layers)}",
        ]
        if result.quantization:
            lines.append(f"Quantization: {result.quantization}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("validate-merge")
def model_validate_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Path to source model"),
    target: str = typer.Option(..., "--target", help="Path to target model"),
) -> None:
    """Validate merge compatibility between two models.

    Examples:
        mc model validate-merge --source ./model-a --target ./model-b
    """
    context = _context(ctx)
    service = ModelProbeService()
    try:
        result = service.validate_merge(source, target)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1002",
            title="Merge validation failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "compatible": result.compatible,
        "architectureMatch": result.architecture_match,
        "vocabMatch": result.vocab_match,
        "dimensionMatch": result.dimension_match,
        "warnings": result.warnings,
    }

    if context.output_format == "text":
        status = "COMPATIBLE" if result.compatible else "INCOMPATIBLE"
        lines = [
            "MERGE VALIDATION",
            f"Status: {status}",
            f"Architecture Match: {'Yes' if result.architecture_match else 'No'}",
            f"Vocab Match: {'Yes' if result.vocab_match else 'No'}",
            f"Dimension Match: {'Yes' if result.dimension_match else 'No'}",
        ]
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze-alignment")
def model_analyze_alignment(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Analyze alignment drift between two models.

    Examples:
        mc model analyze-alignment --model-a ./base-model --model-b ./fine-tuned
    """
    context = _context(ctx)
    service = ModelProbeService()
    try:
        result = service.analyze_alignment(model_a, model_b)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1003",
            title="Alignment analysis failed",
            detail=str(exc),
            hint="Ensure both paths point to valid model directories",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "driftMagnitude": result.drift_magnitude,
        "assessment": result.assessment,
        "interpretation": result.interpretation,
        "layerDrifts": [
            {
                "layerName": drift.layer_name,
                "driftMagnitude": drift.drift_magnitude,
                "direction": drift.direction,
            }
            for drift in result.layer_drifts[:20]  # Limit to first 20 layers
        ],
    }

    if context.output_format == "text":
        lines = [
            "ALIGNMENT ANALYSIS",
            f"Drift Magnitude: {result.drift_magnitude:.4f}",
            f"Assessment: {result.assessment}",
            f"Interpretation: {result.interpretation}",
        ]
        if result.layer_drifts:
            lines.append("")
            lines.append("Layer Drifts (top 10):")
            for drift in sorted(result.layer_drifts, key=lambda d: d.drift_magnitude, reverse=True)[:10]:
                lines.append(f"  {drift.layer_name}: {drift.drift_magnitude:.4f} ({drift.direction})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Helper functions for model search


def _parse_model_search_library(value: str) -> ModelSearchLibraryFilter:
    normalized = value.lower()
    if normalized == "mlx":
        return ModelSearchLibraryFilter.mlx
    if normalized == "safetensors":
        return ModelSearchLibraryFilter.safetensors
    if normalized == "pytorch":
        return ModelSearchLibraryFilter.pytorch
    if normalized == "any":
        return ModelSearchLibraryFilter.any
    raise typer.BadParameter("Invalid library filter. Use: mlx, safetensors, pytorch, or any.")


def _parse_model_search_quant(value: Optional[str]) -> ModelSearchQuantization | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "4bit":
        return ModelSearchQuantization.four_bit
    if normalized == "8bit":
        return ModelSearchQuantization.eight_bit
    if normalized == "any":
        return ModelSearchQuantization.any
    raise typer.BadParameter("Invalid quantization filter. Use: 4bit, 8bit, or any.")


def _parse_model_search_sort(value: str) -> ModelSearchSortOption:
    normalized = value.lower()
    if normalized == "downloads":
        return ModelSearchSortOption.downloads
    if normalized == "likes":
        return ModelSearchSortOption.likes
    if normalized in {"lastmodified", "last_modified"}:
        return ModelSearchSortOption.last_modified
    if normalized == "trending":
        return ModelSearchSortOption.trending
    raise typer.BadParameter("Invalid sort option. Use: downloads, likes, lastModified, or trending.")


def _print_model_search_text(page: ModelSearchPage) -> None:
    if not page.models:
        write_output("No models found matching your query.", "text", False)
        return

    lines: list[str] = [f"Found {len(page.models)} models:\n"]
    for model in page.models:
        fit_indicator = ""
        if model.memory_fit_status == MemoryFitStatus.fits:
            fit_indicator = "[fits]"
        elif model.memory_fit_status == MemoryFitStatus.tight:
            fit_indicator = "[tight]"
        elif model.memory_fit_status == MemoryFitStatus.too_big:
            fit_indicator = "[too big]"

        header = f"{model.id} {fit_indicator}".rstrip()
        lines.append(header)
        downloads = _format_number(model.downloads)
        likes = _format_number(model.likes)
        lines.append(f"  Downloads: {downloads} | Likes: {likes}")
        if model.is_gated:
            lines.append("  [Gated - requires access request]")
        lines.append("")

    if page.has_more and page.next_cursor:
        lines.append(f"More results available. Use --cursor '{page.next_cursor}' for next page.")

    write_output("\n".join(lines).rstrip(), "text", False)


def _format_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)
