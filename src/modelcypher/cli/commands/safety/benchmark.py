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

"""Benchmark and diagnostic safety CLI commands.

Provides commands for benchmarking and diagnostics:
- benchmark: Run benchmark suite with geometric metrics
- reasoning-geometry-validation: Cross-model validation of reasoning signals
- lora-svd: Analyze LoRA adapter with SVD decomposition
- sparse-region: Explore sparse activation regions
- knowledge-type: Analyze whether a statement is fact or opinion
- curriculum-profile: Profile training problems by geometric difficulty
"""

from __future__ import annotations

from ._common import (
    ErrorDetail,
    Path,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


@app.command("benchmark")
def run_benchmark(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    suite: str = typer.Option(
        "quick", "--suite", "-s", help="Benchmark suite (quick, reasoning, factual, comprehensive)"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Samples per benchmark"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory for results"),
) -> None:
    """Run benchmark suite with geometric metrics.

    Suites:
        quick: gsm8k, arc_easy, boolq
        reasoning: gsm8k, arc_challenge, hellaswag
        factual: mmlu, arc_easy, boolq
        comprehensive: All of the above

    Examples:
        mc analyze benchmark /path/to/model --suite quick
        mc analyze benchmark /path/to/model --suite comprehensive -o ./results
    """
    from modelcypher.cli.composition import (
        get_benchmark_service,
        get_inference_engine,
        get_model_loader,
    )

    context = get_context(ctx)

    typer.echo(f"Running benchmark suite: {suite} (limit={limit} per benchmark)")

    try:
        model_loader = get_model_loader()
        inference_engine = get_inference_engine()

        loaded_model, tokenizer = model_loader.load(model)

        def generate_fn(m, t, prompt, max_tokens, verbose=False):
            return inference_engine.generate(m, t, prompt, max_tokens=max_tokens)

        service = get_benchmark_service()
        result = service.run_suite(
            model=loaded_model,
            tokenizer=tokenizer,
            suite_name=suite,
            generate_fn=generate_fn,
            limit_per_benchmark=limit,
            max_failures=5,
        )

        payload = result.to_dict()
        payload["modelPath"] = model

        if output_dir:
            import json
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"benchmark_{suite}.json", "w") as f:
                json.dump(payload, f, indent=2)
            payload["savedTo"] = str(output_path / f"benchmark_{suite}.json")

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4001",
            title="Benchmark failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("reasoning-geometry-validation")
def reasoning_geometry_validation(
    ctx: typer.Context,
    models: list[str] | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model registry key (repeatable). If omitted, uses all registered models.",
    ),
    benchmarks: list[str] | None = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Benchmark name (repeatable). If omitted, uses gsm8k + arithmetic.",
    ),
    samples: int = typer.Option(500, "--samples", "-n", help="Samples per benchmark"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max generated tokens per sample"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    batch_size: int = typer.Option(
        50, "--batch-size", help="Trajectory batch size for periodic cache clearing"
    ),
    output_dir: str = typer.Option(
        "results/reasoning_geometry_validation",
        "--output",
        "-o",
        help="Output directory for report and per-model JSON",
    ),
) -> None:
    """Run cross-model validation of reasoning geometry signals.

    Promoted entrypoint for the reasoning geometry experiment in
    ``scripts/reasoning_geometry_validation.py``.
    """
    from modelcypher.core.use_cases.reasoning_geometry_validation_service import (
        ReasoningGeometryValidationRequest,
        run_reasoning_geometry_validation,
    )

    context = get_context(ctx)
    request = ReasoningGeometryValidationRequest(
        models=tuple(models) if models else (),
        benchmarks=tuple(benchmarks) if benchmarks else ("gsm8k", "arithmetic"),
        samples=samples,
        max_tokens=max_tokens,
        seed=seed,
        batch_size=batch_size,
        output_dir=Path(output_dir),
    )

    try:
        result = run_reasoning_geometry_validation(request)
        payload = result.to_dict()
        payload["models"] = list(request.models) if request.models else "all"
        payload["benchmarks"] = list(request.benchmarks)
        payload["samples"] = request.samples
        payload["maxTokens"] = request.max_tokens
        payload["seed"] = request.seed
        payload["batchSize"] = request.batch_size
        write_output(payload, context.output_format, context.pretty)
    except Exception as e:
        error = ErrorDetail(
            code="MC-4011",
            title="Reasoning geometry validation failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("lora-svd")
def lora_svd_diagnostic(
    ctx: typer.Context,
    adapter_path: str = typer.Argument(..., help="Path to LoRA adapter"),
    base_model: str = typer.Option(..., "--base", "-b", help="Path to base model"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Show top-k layers by change"),
    baseline_artifact: str | None = typer.Option(
        None,
        "--baseline-artifact",
        help=(
            "Optional measured baseline artifact JSON for reference context "
            "(defaults to results/real_adapter_analysis/summary.json when present)."
        ),
    ),
) -> None:
    """Analyze LoRA adapter with SVD decomposition.

    Shows rank changes, null space components, and subspace overlap per layer.
    Useful for understanding what a LoRA adapter is actually doing geometrically.

    Examples:
        mc analyze lora-svd ./my-adapter --base /path/to/base
        mc analyze lora-svd ./my-adapter --base /path/to/base --top-k 10
    """
    from modelcypher.cli.commands._lora_baseline_artifact import load_reference_baseline
    from modelcypher.core.use_cases.lora_diagnostic_service import (
        run_diagnostic,
    )

    context = get_context(ctx)

    typer.echo(f"Analyzing LoRA adapter: {adapter_path}")

    report = run_diagnostic(model_path=base_model, adapter_path=adapter_path)
    reference_baseline = load_reference_baseline(baseline_artifact)

    # Sort by frobenius delta (relative change)
    sorted_reports = sorted(
        report.layer_svd, key=lambda r: abs(r.frobenius_delta), reverse=True
    )

    if context.output_format == "text":
        lines = [
            "LORA SVD DIAGNOSTIC",
            f"Adapter: {adapter_path}",
            f"Base model: {base_model}",
            f"Layers with LoRA: {report.layers_with_lora}",
            f"Total params modified: {report.total_params_modified}",
            "",
            f"TOP {top_k} LAYERS BY FROBENIUS CHANGE:",
        ]
        for r in sorted_reports[:top_k]:
            lines.append(
                f"  Layer {r.layer_idx} ({r.weight_name}): "
                f"rank {r.rank_before}->{r.rank_after} (delta{r.rank_delta:+d}), "
                f"frob_delta={r.frobenius_delta:.4f}"
            )
        if reference_baseline is not None:
            lines.extend(
                [
                    "",
                    "REFERENCE BASELINE (MEASURED ARTIFACT):",
                    f"  amplification_cv={reference_baseline['amplification_cv']:.6f}",
                    f"  weyl_utilization={reference_baseline['weyl_utilization']:.6f}",
                    f"  source={reference_baseline['source']}",
                    f"  artifact={reference_baseline['artifact_path']}",
                ]
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "adapter_path": adapter_path,
        "base_model": base_model,
        "layers_with_lora": report.layers_with_lora,
        "total_params_modified": report.total_params_modified,
        "avg_null_space_activation": report.avg_null_space_activation,
        "avg_subspace_overlap": report.avg_subspace_overlap,
        "peak_change_layer": report.peak_change_layer,
        "reference_baseline": reference_baseline,
        "top_layers": [
            {
                "layer_idx": r.layer_idx,
                "weight_name": r.weight_name,
                "rank_before": r.rank_before,
                "rank_after": r.rank_after,
                "rank_delta": r.rank_delta,
                "frobenius_delta": r.frobenius_delta,
            }
            for r in sorted_reports[:top_k]
        ],
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("sparse-region")
def sparse_region_analysis(
    ctx: typer.Context,
    list_domains: bool = typer.Option(False, "--list-domains", "-l", help="List available sparse region domains"),
    list_pairs: bool = typer.Option(False, "--list-pairs", "-p", help="List contrastive pairs for refusal detection"),
) -> None:
    """Explore sparse activation regions and refusal directions.

    Sparse regions in activation space can correspond to specific behaviors
    like refusal or domain-specific knowledge.

    Examples:
        mc analyze sparse-region --list-domains
        mc analyze sparse-region --list-pairs
    """
    from modelcypher.cli.composition import get_geometry_sparse_service
    from modelcypher.core.use_cases.geometry_sparse_service import (
        GeometrySparseService,
    )

    context = get_context(ctx)
    service = get_geometry_sparse_service()

    if list_pairs:
        pairs = service.get_contrastive_pairs()
        payload = GeometrySparseService.contrastive_pairs_payload(pairs)
    else:
        # Default to listing domains
        domains = service.list_domains()
        payload = GeometrySparseService.domains_payload(domains)

    write_output(payload, context.output_format, context.pretty)


@app.command("knowledge-type")
def knowledge_type_analysis(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    statement: str = typer.Option(..., "--statement", "-s", help="Statement to analyze"),
    counterfactual: str = typer.Option(..., "--counterfactual", "-c", help="Counterfactual version"),
    layer: int = typer.Option(..., "--layer", "-l", help="Layer index to analyze"),
    sensitivity_threshold: float | None = typer.Option(
        None,
        "--sensitivity-threshold",
        help=(
            "Optional explicit threshold for fact/opinion labeling. "
            "If omitted, command returns raw measurements only."
        ),
    ),
) -> None:
    """Analyze whether a statement is factual knowledge or opinion.

    Reports raw counterfactual sensitivity and spectrum metrics.
    Classification is optional and only applied when an explicit threshold
    is provided.

    Examples:
        mc analyze knowledge-type /path/to/model \\
            --statement "The capital of France is Paris" \\
            --counterfactual "The capital of France is Madrid" \\
            --layer 12
    """
    from modelcypher.cli.composition import get_knowledge_analyzer, get_model_loader

    context = get_context(ctx)

    typer.echo(f"Analyzing knowledge type at layer {layer}")

    try:
        model_loader = get_model_loader()
        loaded_model, tokenizer = model_loader.load(model)

        analyzer = get_knowledge_analyzer()
        result = analyzer.analyze_statement(
            model=loaded_model,
            tokenizer=tokenizer,
            statement=statement,
            counterfactual=counterfactual,
            layer_idx=layer,
        )

        payload = {
            "model": model,
            "statement": statement,
            "counterfactual": counterfactual,
            "layer": layer,
            "counterfactualSensitivity": result.counterfactual_sensitivity,
            "effectiveRank": result.effective_rank,
            "spectralEntropy": result.spectral_entropy,
        }
        if sensitivity_threshold is not None:
            payload["classification"] = (
                "fact"
                if result.counterfactual_sensitivity > sensitivity_threshold
                else "opinion"
            )
            payload["threshold"] = sensitivity_threshold

        if context.output_format == "text":
            lines = [
                "KNOWLEDGE TYPE ANALYSIS",
                f"Statement: {statement}",
                f"Counterfactual: {counterfactual}",
                f"Layer: {layer}",
                "",
                f"Counterfactual Sensitivity: {result.counterfactual_sensitivity:.4f}",
                "",
                f"Effective Rank: {result.effective_rank:.2f}",
                f"Spectral Entropy: {result.spectral_entropy:.4f}",
            ]
            if sensitivity_threshold is not None:
                classification = (
                    "fact"
                    if result.counterfactual_sensitivity > sensitivity_threshold
                    else "opinion"
                )
                lines.insert(6, f"Classification: {classification.upper()}")
                lines.insert(7, f"  (threshold: {sensitivity_threshold})")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4002",
            title="Knowledge analysis failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("curriculum-profile")
def curriculum_profile(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    problems_file: str = typer.Option(..., "--problems", "-p", help="JSON file with problems to profile"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output CSV file"),
    layer: int | None = typer.Option(None, "--layer", "-l", help="Layer index for profiling (default: final layer)"),
) -> None:
    """Profile training problems by geometric difficulty.

    Measures difficulty using geometric signals:
    - CKA similarity to reference
    - Activation barrier height
    - Fisher Information
    - Trajectory curvature
    - Local density
    - Intrinsic dimension

    Examples:
        mc analyze curriculum-profile /path/to/model --problems problems.json
        mc analyze curriculum-profile /path/to/model --problems problems.json -o difficulty.csv
    """
    import json

    from modelcypher.cli.composition import get_curriculum_profiler, get_model_loader

    context = get_context(ctx)

    typer.echo("Profiling curriculum difficulty")

    try:
        # Load problems from JSON file
        problems_path = Path(problems_file)
        if not problems_path.exists():
            raise FileNotFoundError(f"Problems file not found: {problems_file}")

        with open(problems_path) as f:
            data = json.load(f)

        # Support different JSON formats
        if isinstance(data, list):
            # Simple list of strings or list of dicts with "prompt" key
            if data and isinstance(data[0], str):
                problems = data
                problem_ids = None
            else:
                problems = [p.get("prompt", p.get("text", str(p))) for p in data]
                problem_ids = [p.get("id", f"p{i}") for i, p in enumerate(data)]
        elif isinstance(data, dict):
            problems = data.get("problems", data.get("prompts", []))
            problem_ids = data.get("ids")
        else:
            raise ValueError("Unsupported problems file format")

        if not problems:
            raise ValueError("No problems found in file")

        typer.echo(f"Loaded {len(problems)} problems from {problems_file}")

        # Load model
        model_loader = get_model_loader()
        loaded_model, tokenizer = model_loader.load(model)

        # Create profiler and profile problems
        profiler = get_curriculum_profiler(loaded_model, tokenizer, layer_idx=layer)

        def progress_callback(current, total):
            typer.echo(f"  Profiling problem {current}/{total}...", nl=False)
            typer.echo("\r", nl=False)

        profiles = profiler.profile_problems(
            problems=problems,
            problem_ids=problem_ids,
            progress_callback=progress_callback,
        )

        # Compute difficulty scores
        profiles.compute_difficulty_scores()

        # Save to CSV if output file specified
        if output_file:
            df = profiles.to_dataframe()
            if hasattr(df, "to_csv"):
                df.to_csv(output_file, index=False)
                typer.echo(f"Saved profiles to {output_file}")

        # Build output payload
        payload = profiles.as_dict()
        payload["modelPath"] = model
        payload["problemsFile"] = problems_file
        if output_file:
            payload["outputFile"] = output_file

        if context.output_format == "text":
            lines = [
                "CURRICULUM PROFILE",
                f"Model: {model}",
                f"Problems: {len(profiles.profiles)}",
                f"Reference Count: {profiles.reference_count}",
                "",
                "TOP 5 BY DIFFICULTY:",
            ]
            sorted_profiles = profiles.sort_by_difficulty("difficulty_score", ascending=False)
            for p in sorted_profiles[:5]:
                lines.append(
                    f"  {p.problem_id}: score={p.difficulty_score:.3f}, "
                    f"CKA={p.cka_similarity:.3f}, Fisher={p.fisher_mean:.6f}"
                )
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4003",
            title="Curriculum profiling failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
