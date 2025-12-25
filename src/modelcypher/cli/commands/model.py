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


import typer

from modelcypher.cli.composition import get_model_merge_service
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
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
from modelcypher.core.use_cases.model_search_service import ModelSearchService
from modelcypher.core.use_cases.model_service import ModelService
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
    parameters: int | None = typer.Option(None, "--parameters"),
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


def _run_smoke_test(model_path: str, context: Any) -> dict:
    """Run a quick inference smoke test on a merged model.

    Returns dict with:
        passed: bool
        response: str (first 100 chars)
        tokens_per_second: float
        error: str | None
    """
    import logging
    from modelcypher.adapters.local_inference import LocalInferenceEngine

    logger = logging.getLogger(__name__)
    smoke_prompts = [
        "Hello, how are you today?",
        "What is 2 + 2?",
        "Complete this sentence: The quick brown fox",
    ]

    try:
        engine = LocalInferenceEngine()
        # Run 3 prompts, check for coherent output
        results = []
        for prompt in smoke_prompts:
            result = engine.run(
                model=model_path,
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
            )
            response = result.text if hasattr(result, 'text') else str(result)
            tps = result.tokens_per_second if hasattr(result, 'tokens_per_second') else 0
            results.append({
                "prompt": prompt,
                "response": response[:100],
                "tokens_per_second": tps,
            })

        # Check for obvious failures
        all_empty = all(len(r["response"].strip()) == 0 for r in results)
        all_garbage = all(
            len(set(r["response"])) < 5 or "�" in r["response"]
            for r in results
        )
        mean_tps = sum(r["tokens_per_second"] for r in results) / len(results)

        if all_empty:
            return {"passed": False, "error": "All responses empty", "results": results}
        if all_garbage:
            return {"passed": False, "error": "Responses appear garbled", "results": results}
        if mean_tps < 1.0:
            return {"passed": False, "error": f"Very slow inference: {mean_tps:.1f} tok/s", "results": results}

        logger.info("SMOKE TEST: PASSED (%.1f tok/s)", mean_tps)
        return {
            "passed": True,
            "tokens_per_second": mean_tps,
            "results": results,
            "error": None,
        }

    except Exception as e:
        logger.warning("SMOKE TEST: FAILED - %s", str(e))
        return {"passed": False, "error": str(e), "results": []}


@app.command("merge")
def model_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source"),
    target: str = typer.Option(..., "--target"),
    output_dir: str = typer.Option(..., "--output-dir"),
    alpha: float = typer.Option(0.5, "--alpha"),
    rank: int = typer.Option(32, "--rank"),
    module_scope: str | None = typer.Option(None, "--module-scope"),
    anchor_mode: str = typer.Option("unified", "--anchor-mode", help="unified, semantic-primes, geometric, rebasin"),
    intersection: str | None = typer.Option(None, "--intersection"),
    adaptive_alpha: bool = typer.Option(False, "--adaptive-alpha"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run inference smoke test after merge"),
) -> None:
    """Merge two models using geometric alignment.

    Uses the unified geometric merge pipeline with 343 probes.

    Examples:
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged --alpha 0.7
        mc model merge --source ./model-a --target ./model-b --output-dir ./merged --smoke-test
    """
    from modelcypher.cli.composition import get_model_merge_service

    context = _context(ctx)

    service = get_model_merge_service()
    try:
        result = service.merge(
            source_id=source,
            target_id=target,
            output_dir=output_dir,
            alpha=alpha,
            alignment_rank=rank,
            module_scope=module_scope,
            anchor_mode=anchor_mode,
            intersection_path=intersection,
            adaptive_alpha=adaptive_alpha,
        )

        # Run smoke test if requested
        if smoke_test:
            smoke_result = _run_smoke_test(output_dir, context)
            result["smokeTest"] = smoke_result
            if not smoke_result["passed"]:
                result["warning"] = f"Smoke test failed: {smoke_result.get('error', 'unknown')}"

        write_output(result, context.output_format, context.pretty)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1010",
            title="Merge failed",
            detail=str(e),
            hint="Check model paths and compatibility",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


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
    output_quant: str | None = typer.Option(None, "--output-quant", help="Output quantization (4bit, 8bit)"),
    output_quant_group_size: int | None = typer.Option(None, "--output-quant-group-size"),
    output_quant_mode: str | None = typer.Option(None, "--output-quant-mode"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without saving"),
    report_path: str | None = typer.Option(None, "--report-path", help="Path to save merge report"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Use preset config: default, skill-preserving, structure-preserving",
    ),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run inference smoke test after merge"),
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

    service = get_model_merge_service()
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

    # Run smoke test if requested
    if smoke_test and not dry_run:
        typer.echo("\nRunning smoke test...", err=True)
        smoke_result = _run_smoke_test(output_dir, context)
        report["smokeTest"] = smoke_result
        if smoke_result["passed"]:
            typer.echo(f"  PASSED ({smoke_result.get('tokens_per_second', 0):.1f} tok/s)", err=True)
        else:
            typer.echo(f"  FAILED: {smoke_result.get('error', 'unknown')}", err=True)
            report["warning"] = f"Smoke test failed: {smoke_result.get('error', 'unknown')}"

    write_output(report, context.output_format, context.pretty)


@app.command("unified-merge")
def model_unified_merge(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source model path (skill donor)"),
    target: str = typer.Option(..., "--target", help="Target model path (knowledge base)"),
    output_dir: str = typer.Option(..., "--output-dir", help="Output directory for merged model"),
    alpha: float = typer.Option(0.5, "--alpha", help="Base alpha (0=target, 1=source)"),
    # Probe stage
    probe_mode: str = typer.Option(
        "precise",
        "--probe-mode",
        help="Probe mode: 'precise' runs 403 probes through models (correct, slower), 'fast' uses weight CKA (faster, less accurate)",
    ),
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
    use_svd_blending: bool = typer.Option(False, "--svd-blending/--no-svd-blending", help="Enable SVD-aware blending (slower, uses CPU)"),
    use_correlation_weights: bool = typer.Option(False, "--correlation-weights/--no-correlation-weights", help="Enable correlation-based weighting (slower, uses CPU)"),
    use_verb_noun: bool = typer.Option(False, "--verb-noun/--no-verb-noun", help="Enable VerbNoun modulation (slower, uses CPU)"),
    verb_noun_strength: float = typer.Option(0.7, "--verb-noun-strength", help="VerbNoun modulation strength"),
    # Propagate stage
    enable_zipper: bool = typer.Option(True, "--zipper/--no-zipper", help="Enable zipper propagation"),
    # Output
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without saving"),
    report_path: str | None = typer.Option(None, "--report-path", help="Path to save merge report"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Use preset config: default, conservative, aggressive",
    ),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run inference smoke test after merge"),
) -> None:
    """Execute unified geometric merge.

    Combines geometric techniques in the correct order:

    1. PROBE: Build intersection map from probe responses
       - precise (default): Run 403 probes through BOTH models, compute CKA on activations
       - fast: Use weight-level CKA (faster but less accurate)
    2. PERMUTE: Align MLP neurons using Re-Basin
    3. ROTATE: Apply Procrustes geometric alignment
    4. BLEND: Multi-stage alpha with spectral, SVD, correlation, and VerbNoun adjustments
    5. PROPAGATE: Carry rotations layer-to-layer (zipper)

    Examples:
        mc model unified-merge --source ./instruct --target ./coder --output-dir ./merged
        mc model unified-merge --source ./instruct --target ./coder --output-dir ./merged --probe-mode fast
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

        # Override base_alpha and probe_mode if explicitly provided
        if alpha != 0.5 or probe_mode.lower() != "precise":
            # Create new config with overridden values
            config = UnifiedMergeConfig(
                probe_mode=probe_mode.lower(),
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
        # Validate probe_mode
        valid_probe_modes = ("precise", "fast")
        if probe_mode.lower() not in valid_probe_modes:
            typer.echo(f"Invalid probe mode: {probe_mode}. Must be one of: {valid_probe_modes}", err=True)
            raise typer.Exit(code=1)

        config = UnifiedMergeConfig(
            probe_mode=probe_mode.lower(),
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
    typer.echo(f"Probe mode: {config.probe_mode} (403 probes {'executed' if config.probe_mode == 'precise' else 'skipped - using weight CKA'})", err=True)
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
        zipper_prop = rotate.get('zipper_propagations', 0)
        zipper_app = rotate.get('zipper_applications', 0)
        if transport > 0:
            typer.echo(f"  Transport-guided: {transport} (GW distance={rotate.get('mean_gw_distance', 0):.4f})", err=True)
        if procrustes > 0:
            typer.echo(f"  Procrustes rotations: {procrustes}", err=True)
        if zipper_prop > 0 or zipper_app > 0:
            typer.echo(f"  Zipper: {zipper_prop} propagations, {zipper_app} applications", err=True)

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

    # Run smoke test if requested
    if smoke_test and not dry_run and result.output_path:
        typer.echo("\nRunning smoke test...", err=True)
        smoke_result = _run_smoke_test(result.output_path, context)
        report["smokeTest"] = smoke_result
        if smoke_result["passed"]:
            typer.echo(f"  PASSED ({smoke_result.get('tokens_per_second', 0):.1f} tok/s)", err=True)
        else:
            typer.echo(f"  FAILED: {smoke_result.get('error', 'unknown')}", err=True)
            report["warning"] = f"Smoke test failed: {smoke_result.get('error', 'unknown')}"

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
    alias: str | None = typer.Option(None, "--alias"),
    architecture: str | None = typer.Option(None, "--architecture"),
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
    query: str | None = typer.Argument(None),
    author: str | None = typer.Option(None, "--author"),
    library: str = typer.Option("mlx", "--library"),
    quant: str | None = typer.Option(None, "--quant"),
    sort: str = typer.Option("downloads", "--sort"),
    limit: int = typer.Option(20, "--limit"),
    cursor: str | None = typer.Option(None, "--cursor"),
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


@app.command("validate-knowledge")
def model_validate_knowledge(
    ctx: typer.Context,
    merged: str = typer.Option(..., "--merged", help="Path to merged model"),
    source: str | None = typer.Option(None, "--source", help="Path to source model (for baseline)"),
    domains: str | None = typer.Option(None, "--domains", help="Comma-separated domains: math,code,factual,reasoning,language,creative"),
    quick: bool = typer.Option(False, "--quick", help="Quick validation (skip variations)"),
    report_path: str | None = typer.Option(None, "--report-path", help="Path to save validation report"),
) -> None:
    """Validate knowledge transfer in merged model.

    Tests whether the merged model retains knowledge from the source model
    across multiple domains using targeted probes.

    Examples:
        mc model validate-knowledge --merged ./merged-model
        mc model validate-knowledge --merged ./merged-model --source ./source-model
        mc model validate-knowledge --merged ./merged-model --domains math,code
        mc model validate-knowledge --merged ./merged-model --quick
    """
    from modelcypher.core.use_cases.knowledge_transfer_service import (
        KnowledgeTransferService,
        KnowledgeTransferConfig,
    )
    from modelcypher.core.domain.merging.knowledge_transfer_validator import (
        KnowledgeDomain,
    )

    context = _context(ctx)

    # Parse domains
    domain_list = None
    if domains:
        domain_list = []
        for d in domains.split(","):
            d = d.strip().lower()
            try:
                domain_list.append(KnowledgeDomain(d))
            except ValueError:
                valid_domains = [dom.value for dom in KnowledgeDomain]
                error = ErrorDetail(
                    code="MC-1030",
                    title="Invalid domain",
                    detail=f"Unknown domain: {d}",
                    hint=f"Valid domains are: {', '.join(valid_domains)}",
                    trace_id=context.trace_id,
                )
                write_error(error.as_dict(), context.output_format, context.pretty)
                raise typer.Exit(code=1)

    # Build config
    config = KnowledgeTransferConfig(
        domains=domain_list if domain_list else list(KnowledgeDomain),
        include_variations=not quick,
    )

    typer.echo("Running knowledge transfer validation...", err=True)
    typer.echo(f"  Merged model: {merged}", err=True)
    if source:
        typer.echo(f"  Source model: {source}", err=True)
    typer.echo(f"  Domains: {', '.join(d.value for d in config.domains)}", err=True)

    service = KnowledgeTransferService()
    try:
        result = service.validate(
            merged_model=merged,
            source_model=source,
            config=config,
        )
    except Exception as e:
        error = ErrorDetail(
            code="MC-1031",
            title="Knowledge validation failed",
            detail=str(e),
            hint="Check model paths and ensure models are loaded correctly",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Display summary
    typer.echo(f"\nKnowledge Transfer Validation Complete!", err=True)
    typer.echo(f"  Status: {result.status.value.upper()}", err=True)
    typer.echo(f"  Overall Retention: {result.overall_retention:.1%}", err=True)
    typer.echo(f"  Probes Executed: {result.probes_executed}", err=True)
    typer.echo(f"  Time: {result.execution_time_seconds:.1f}s", err=True)

    typer.echo("\n  Per-Domain Retention:", err=True)
    for domain, domain_result in result.report.per_domain.items():
        status_icon = "✓" if domain_result.retention_score >= 0.8 else "✗"
        typer.echo(
            f"    {status_icon} {domain.value}: {domain_result.retention_score:.1%} "
            f"({domain_result.probes_tested} probes)",
            err=True,
        )

    if result.warnings:
        typer.echo("\n  Warnings:", err=True)
        for warning in result.warnings:
            typer.echo(f"    - {warning}", err=True)

    typer.echo(f"\n  Recommendation: {result.report.recommendation}", err=True)

    # Save report if requested
    if report_path:
        from pathlib import Path

        Path(report_path).write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        typer.echo(f"  Report saved: {report_path}", err=True)

    write_output(result.to_dict(), context.output_format, context.pretty)


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


@app.command("vocab-compare")
def model_vocab_compare(
    ctx: typer.Context,
    model_a: str = typer.Option(..., "--model-a", help="Path to first model"),
    model_b: str = typer.Option(..., "--model-b", help="Path to second model"),
) -> None:
    """Compare vocabularies between two models for cross-vocabulary merging.

    Analyzes tokenizer overlap and recommends merge strategies:
    - High overlap (>90%): FVT (Fast Vocabulary Transfer) only
    - Medium overlap (50-90%): FVT + Procrustes verification
    - Low overlap (<50%): Procrustes + Affine transformation

    Examples:
        mc model vocab-compare --model-a ./llama-3-8b --model-b ./qwen-2-7b
    """
    from pathlib import Path

    context = _context(ctx)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        error = ErrorDetail(
            code="MC-1020",
            title="Missing dependency",
            detail="transformers package required for vocabulary comparison",
            hint="Install with: pip install transformers",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        from modelcypher.core.domain.vocabulary import (
            compare_tokenizers,
            format_comparison_report,
        )
    except ImportError as e:
        error = ErrorDetail(
            code="MC-1021",
            title="Vocabulary comparison not available",
            detail=str(e),
            hint="Ensure modelcypher is properly installed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load tokenizers
    typer.echo(f"Loading tokenizer from {model_a}...", err=True)
    try:
        tokenizer_a = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1022",
            title="Failed to load tokenizer",
            detail=f"Model A: {e}",
            hint="Ensure the path contains a valid tokenizer",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    typer.echo(f"Loading tokenizer from {model_b}...", err=True)
    try:
        tokenizer_b = AutoTokenizer.from_pretrained(model_b, trust_remote_code=True)
    except Exception as e:
        error = ErrorDetail(
            code="MC-1022",
            title="Failed to load tokenizer",
            detail=f"Model B: {e}",
            hint="Ensure the path contains a valid tokenizer",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Perform comparison
    typer.echo("Analyzing vocabulary overlap...", err=True)
    result = compare_tokenizers(tokenizer_a, tokenizer_b)

    # Build payload
    payload = result.to_dict()
    payload["modelA"] = model_a
    payload["modelB"] = model_b

    if context.output_format == "text":
        report = format_comparison_report(result)
        typer.echo("")
        typer.echo(report)
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


def _parse_model_search_quant(value: str | None) -> ModelSearchQuantization | None:
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
