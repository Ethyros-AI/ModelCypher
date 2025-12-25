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

"""Geometry sparse region CLI commands.

Provides commands for analyzing sparse regions in model representations
for targeted LoRA injection.

Commands:
    mc geometry sparse domains
    mc geometry sparse locate <domain_stats_file> <baseline_stats_file>
    mc geometry sparse neurons --model <path> [--domain <domain>]
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("domains")
def geometry_sparse_domains(ctx: typer.Context) -> None:
    """List all built-in sparse region domains."""
    context = _context(ctx)
    service = GeometrySparseService()
    domains = service.list_domains()
    payload = service.domains_payload(domains)

    if context.output_format == "text":
        lines = [
            "SPARSE REGION DOMAINS",
            f"Total: {payload['count']}",
            "",
        ]
        for d in payload["domains"]:
            range_str = ""
            if d["expectedLayerRange"]:
                range_str = (
                    f" (layers {d['expectedLayerRange'][0]:.0%}-{d['expectedLayerRange'][1]:.0%})"
                )
            lines.append(f"  {d['name']}: {d['description']}{range_str}")
            lines.append(f"    Category: {d['category']}, Probes: {d['probeCount']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("locate")
def geometry_sparse_locate(
    ctx: typer.Context,
    domain_stats_file: str = typer.Argument(..., help="Path to domain layer stats JSON"),
    baseline_stats_file: str = typer.Argument(..., help="Path to baseline layer stats JSON"),
    domain_name: str = typer.Option("unknown", "--domain", help="Domain name"),
    base_rank: int = typer.Option(16, "--rank", help="Base LoRA rank"),
    sparsity_threshold: float = typer.Option(0.3, "--threshold", help="Sparsity threshold"),
) -> None:
    """
    Locate sparse regions for LoRA injection.

    Input files should contain JSON arrays of layer stats:
    [{"layer_index": 0, "mean_activation": 0.5, ...}, ...]
    """
    context = _context(ctx)
    service = GeometrySparseService()

    domain_stats = json.loads(Path(domain_stats_file).read_text())
    baseline_stats = json.loads(Path(baseline_stats_file).read_text())

    result = service.locate_sparse_regions(
        domain_stats=domain_stats,
        baseline_stats=baseline_stats,
        domain_name=domain_name,
        base_rank=base_rank,
        sparsity_threshold=sparsity_threshold,
    )

    payload = service.analysis_payload(result)
    payload["nextActions"] = [
        "mc geometry sparse domains to see available domain definitions",
        "mc geometry adapter sparsity for DARE analysis",
    ]

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Domain: {result.domain}",
            f"Sparse Layers: {len(result.sparse_layers)} {result.sparse_layers}",
            f"Skip Layers: {len(result.skip_layers)} {result.skip_layers}",
            "",
            "LORA RECOMMENDATION",
            f"  Quality: {result.recommendation.quality.value.upper()}",
            f"  Overall Rank: {result.recommendation.overall_rank}",
            f"  Alpha: {result.recommendation.alpha}",
            f"  Preservation: {result.recommendation.estimated_preservation:.0%}",
            "",
            result.recommendation.rationale,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("neurons")
def geometry_sparse_neurons(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompts_file: str = typer.Option(None, "--prompts", help="Path to prompts JSON file"),
    domain: str = typer.Option(None, "--domain", help="Use built-in domain probes"),
    layer_start: float = typer.Option(0.0, "--layer-start", help="Start layer fraction (0.0-1.0)"),
    layer_end: float = typer.Option(1.0, "--layer-end", help="End layer fraction (0.0-1.0)"),
    sparsity_threshold: float = typer.Option(
        0.8, "--threshold", help="Sparsity threshold for graft candidates"
    ),
    output_file: str = typer.Option(None, "--output", help="Path to save neuron sparsity map"),
) -> None:
    """
    Analyze per-neuron sparsity for fine-grained knowledge grafting.

    This identifies individual neurons that are sparse enough to be
    good candidates for knowledge transfer during model merging.

    Examples:
        mc geometry sparse neurons --model ./model --prompts ./prompts.json
        mc geometry sparse neurons --model ./model --domain math --threshold 0.9
        mc geometry sparse neurons --model ./model --layer-start 0.4 --layer-end 0.7
    """
    from modelcypher.cli.output import write_error
    from modelcypher.core.domain.geometry.neuron_sparsity_analyzer import (
        NeuronSparsityConfig,
        NeuronSparsityMap,
        compute_neuron_sparsity_map,
    )
    from modelcypher.utils.errors import ErrorDetail

    context = _context(ctx)

    # Load prompts
    prompts: list[str] = []
    if prompts_file:
        prompts_path = Path(prompts_file)
        if not prompts_path.exists():
            error = ErrorDetail(
                code="MC-2001",
                title="Prompts file not found",
                detail=f"File not found: {prompts_file}",
                hint="Provide a valid path to a JSON file containing prompts",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        prompts = json.loads(prompts_path.read_text())

    elif domain:
        # Use built-in domain probes
        service = GeometrySparseService()
        domains_list = service.list_domains()
        domain_def = next((d for d in domains_list if d.name.lower() == domain.lower()), None)
        if domain_def is None:
            error = ErrorDetail(
                code="MC-2002",
                title="Domain not found",
                detail=f"Unknown domain: {domain}",
                hint="Use 'mc geometry sparse domains' to list available domains",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
        prompts = domain_def.probes

    if not prompts:
        error = ErrorDetail(
            code="MC-2003",
            title="No prompts provided",
            detail="Either --prompts or --domain is required",
            hint="Provide prompts via --prompts file.json or --domain math",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    typer.echo("Running per-neuron sparsity analysis...", err=True)
    typer.echo(f"  Model: {model}", err=True)
    typer.echo(f"  Prompts: {len(prompts)}", err=True)
    typer.echo(f"  Layer range: {layer_start:.0%} - {layer_end:.0%}", err=True)
    typer.echo(f"  Sparsity threshold: {sparsity_threshold}", err=True)

    config = NeuronSparsityConfig(
        sparsity_threshold=sparsity_threshold,
        min_prompts=min(len(prompts), 20),
    )

    try:
        # Get model layer count
        from modelcypher.core.use_cases.model_probe_service import ModelProbeService

        probe_service = ModelProbeService()
        model_info = probe_service.probe(model)
        total_layers = len([layer for layer in model_info.layers if "layers." in layer.name])

        typer.echo(f"  Total layers: {total_layers}", err=True)

        # Collect activations using HiddenStateExtractor
        from modelcypher.core.domain.entropy.hidden_state_extractor import (
            ExtractorConfig,
            HiddenStateExtractor,
        )

        # Create extractor for neuron analysis
        extractor_config = ExtractorConfig.for_neuron_analysis_range(
            total_layers,
            start_fraction=layer_start,
            end_fraction=layer_end,
            hidden_dim=model_info.hidden_size,
        )
        extractor = HiddenStateExtractor(extractor_config)

        # Collect activations via inference
        from modelcypher.adapters.local_inference import LocalInferenceEngine

        engine = LocalInferenceEngine()
        extractor.start_neuron_collection()

        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                typer.echo(f"  Processing prompt {i + 1}/{len(prompts)}...", err=True)

            # Run inference - note: full activation capture requires
            # integration with the inference engine's forward pass
            try:
                engine.infer(model, prompt, max_tokens=50, temperature=0.0)
            except Exception as infer_err:
                typer.echo(f"  Warning: Inference failed for prompt {i + 1}: {infer_err}", err=True)

            extractor.finalize_prompt_activations()

        # Get collected activations
        activations = extractor.get_neuron_activations()

        if not activations:
            typer.echo(
                "  Note: No activations collected (requires model hook integration)", err=True
            )

        # Compute sparsity map
        sparsity_map = compute_neuron_sparsity_map(activations, config)

    except ImportError as e:
        typer.echo(f"  Note: Full analysis requires inference engine ({e})", err=True)
        sparsity_map = NeuronSparsityMap(stats={}, config=config, total_prompts=0)

    except Exception as e:
        error = ErrorDetail(
            code="MC-2004",
            title="Neuron analysis failed",
            detail=str(e),
            hint="Check model path and ensure model is loadable",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Build payload
    summary = sparsity_map.summary()
    payload = {
        "model": model,
        "config": {
            "sparsityThreshold": config.sparsity_threshold,
            "activationThreshold": config.activation_threshold,
            "layerRange": [layer_start, layer_end],
        },
        "summary": summary,
        "graftCandidates": sparsity_map.get_graft_candidates(),
        "deadNeurons": sparsity_map.dead_neurons,
        "layerSummaries": {
            str(layer): sparsity_map.get_layer_summary(layer)
            for layer in sorted(sparsity_map.stats.keys())[:10]
        },
    }

    # Save full map if requested
    if output_file:
        output_path = Path(output_file)
        full_data = {
            **payload,
            "fullStats": {
                str(layer): [
                    {
                        "neuronIdx": n.neuron_idx,
                        "sparsityScore": round(n.sparsity_score, 4),
                        "meanActivation": round(n.mean_activation, 6),
                        "activeFraction": round(n.active_fraction, 4),
                    }
                    for n in neurons
                ]
                for layer, neurons in sparsity_map.stats.items()
            },
        }
        output_path.write_text(json.dumps(full_data, indent=2), encoding="utf-8")
        typer.echo(f"  Full results saved: {output_file}", err=True)

    # Display summary
    typer.echo("\nNEURON SPARSITY ANALYSIS", err=True)
    typer.echo(f"  Layers analyzed: {summary.get('num_layers', 0)}", err=True)
    typer.echo(f"  Total neurons: {summary.get('total_neurons', 0)}", err=True)
    typer.echo(
        f"  Sparse neurons: {summary.get('total_sparse', 0)} ({summary.get('sparse_fraction', 0):.1%})",
        err=True,
    )
    typer.echo(
        f"  Dead neurons: {summary.get('total_dead', 0)} ({summary.get('dead_fraction', 0):.1%})",
        err=True,
    )
    typer.echo(f"  Mean sparsity: {summary.get('mean_sparsity', 0):.3f}", err=True)
    typer.echo(f"  Graft candidates: {summary.get('graft_candidates', 0)}", err=True)

    if sparsity_map.sparse_neurons:
        typer.echo("\n  Layers with sparse neurons:", err=True)
        for layer in sorted(sparsity_map.sparse_neurons.keys())[:5]:
            count = len(sparsity_map.sparse_neurons[layer])
            typer.echo(f"    Layer {layer}: {count} sparse neurons", err=True)

    write_output(payload, context.output_format, context.pretty)
