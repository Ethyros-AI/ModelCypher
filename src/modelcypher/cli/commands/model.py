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

            # Compute per-dimension alpha vectors if dimension blending is enabled
            if dimension_blending and merge_method.lower() == "linear":
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
                    typer.echo(f"Dimension blending failed: {e}", err=True)
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
