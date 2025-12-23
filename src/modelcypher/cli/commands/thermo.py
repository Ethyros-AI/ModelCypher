"""Thermodynamic analysis CLI commands.

Provides commands for thermodynamic analysis of model training,
including entropy measurement, path integration, and unsafe pattern detection.

Commands:
    mc thermo analyze <job_id>
    mc thermo path --checkpoint <path> ...
    mc thermo entropy <job_id>
    mc thermo measure <prompt> --model <path>
    mc thermo detect <prompt> --model <path>
    mc thermo detect-batch <prompts_file> --model <path>
"""

from __future__ import annotations

from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def thermo_analyze(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID to analyze"),
) -> None:
    """Thermodynamic analysis of training."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.analyze(job_id)

    payload = {
        "jobId": result.job_id,
        "entropy": result.entropy,
        "temperature": result.temperature,
        "freeEnergy": result.free_energy,
        "interpretation": result.interpretation,
    }

    if context.output_format == "text":
        lines = [
            "THERMO ANALYSIS",
            f"Job: {result.job_id}",
            f"Entropy: {result.entropy:.4f}",
            f"Temperature: {result.temperature:.4f}",
            f"Free Energy: {result.free_energy:.4f}",
            "",
            result.interpretation,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("path")
def thermo_path(
    ctx: typer.Context,
    checkpoints: list[str] = typer.Option(..., "--checkpoint", help="Checkpoint paths"),
) -> None:
    """Path integration analysis between checkpoints."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    try:
        result = service.path(checkpoints)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1008",
            title="Path analysis failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "checkpoints": result.checkpoints,
        "pathLength": result.path_length,
        "curvature": result.curvature,
        "interpretation": result.interpretation,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy")
def thermo_entropy(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Entropy metrics over training."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.entropy(job_id)

    payload = {
        "jobId": result.job_id,
        "entropyHistory": result.entropy_history,
        "finalEntropy": result.final_entropy,
        "entropyTrend": result.entropy_trend,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("measure")
def thermo_measure(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to measure"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    modifiers: Optional[list[str]] = typer.Option(None, "--modifier", help="Modifier names to use"),
) -> None:
    """Measure entropy across linguistic modifiers for a prompt."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.measure(prompt, model, modifiers)

    payload = {
        "basePrompt": result.base_prompt,
        "measurements": [
            {
                "modifier": m.modifier,
                "meanEntropy": m.mean_entropy,
                "deltaH": m.delta_h,
                "ridgeCrossed": m.ridge_crossed,
                "behavioralOutcome": m.behavioral_outcome,
            }
            for m in result.measurements
        ],
        "statistics": {
            "meanEntropy": result.statistics.mean_entropy,
            "stdEntropy": result.statistics.std_entropy,
            "minEntropy": result.statistics.min_entropy,
            "maxEntropy": result.statistics.max_entropy,
            "meanDeltaH": result.statistics.mean_delta_h,
            "intensityCorrelation": result.statistics.intensity_correlation,
        },
        "timestamp": result.timestamp.isoformat(),
    }

    if context.output_format == "text":
        lines = [
            "THERMO MEASURE",
            f"Prompt: {result.base_prompt[:50]}{'...' if len(result.base_prompt) > 50 else ''}",
            "",
            "Measurements:",
        ]
        for m in result.measurements:
            delta_str = f"{m.delta_h:.4f}" if m.delta_h is not None else "N/A"
            lines.append(f"  {m.modifier}: entropy={m.mean_entropy:.4f}, delta_h={delta_str}, outcome={m.behavioral_outcome}")
        lines.append("")
        lines.append(f"Mean Entropy: {result.statistics.mean_entropy:.4f}")
        lines.append(f"Std Entropy: {result.statistics.std_entropy:.4f}")
        if result.statistics.intensity_correlation is not None:
            lines.append(f"Intensity Correlation: {result.statistics.intensity_correlation:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect")
def thermo_detect(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to analyze"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    preset: str = typer.Option("default", "--preset", help="Preset: default, strict, sensitive, quick"),
) -> None:
    """Detect unsafe prompt patterns via entropy differential."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()
    result = service.detect(prompt, model, preset)

    payload = {
        "prompt": result.prompt,
        "classification": result.classification,
        "riskLevel": result.risk_level,
        "confidence": result.confidence,
        "baselineEntropy": result.baseline_entropy,
        "intensityEntropy": result.intensity_entropy,
        "deltaH": result.delta_h,
        "processingTime": result.processing_time,
    }

    if context.output_format == "text":
        risk_labels = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        lines = [
            "THERMO DETECT",
            f"Prompt: {result.prompt[:50]}{'...' if len(result.prompt) > 50 else ''}",
            "",
            f"Classification: {result.classification.upper()}",
            f"Risk Level: {result.risk_level} ({risk_labels.get(result.risk_level, 'UNKNOWN')})",
            f"Confidence: {result.confidence:.2%}",
            "",
            f"Baseline Entropy: {result.baseline_entropy:.4f}",
            f"Intensity Entropy: {result.intensity_entropy:.4f}",
            f"Delta H: {result.delta_h:.4f}",
            f"Processing Time: {result.processing_time:.3f}s",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect-batch")
def thermo_detect_batch(
    ctx: typer.Context,
    prompts_file: str = typer.Argument(..., help="Path to prompts file (JSON array or newline-separated)"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    preset: str = typer.Option("default", "--preset", help="Preset: default, strict, sensitive, quick"),
) -> None:
    """Batch detect unsafe patterns across multiple prompts."""
    context = _context(ctx)
    from modelcypher.core.use_cases.thermo_service import ThermoService

    service = ThermoService()

    try:
        results = service.detect_batch(prompts_file, model, preset)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1010",
            title="Batch detection failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "promptsFile": prompts_file,
        "totalPrompts": len(results),
        "results": [
            {
                "prompt": r.prompt,
                "classification": r.classification,
                "riskLevel": r.risk_level,
                "confidence": r.confidence,
                "deltaH": r.delta_h,
            }
            for r in results
        ],
        "summary": {
            "safe": sum(1 for r in results if r.classification == "safe"),
            "unsafe": sum(1 for r in results if r.classification == "unsafe"),
            "ambiguous": sum(1 for r in results if r.classification == "ambiguous"),
        },
    }

    if context.output_format == "text":
        lines = [
            "THERMO DETECT BATCH",
            f"File: {prompts_file}",
            f"Total Prompts: {len(results)}",
            "",
            "Summary:",
            f"  Safe: {payload['summary']['safe']}",
            f"  Unsafe: {payload['summary']['unsafe']}",
            f"  Ambiguous: {payload['summary']['ambiguous']}",
            "",
            "Results:",
        ]
        for i, r in enumerate(results[:10]):  # Show first 10
            prompt_preview = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
            lines.append(f"  {i+1}. [{r.classification.upper()}] {prompt_preview}")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("ridge-detect")
def thermo_ridge_detect(
    ctx: typer.Context,
    baseline_file: str = typer.Argument(..., help="Path to baseline measurement JSON"),
    variants_file: str = typer.Argument(..., help="Path to variant measurements JSON"),
    preset: str = typer.Option("default", "--preset", help="Preset: default, strict, lenient"),
) -> None:
    """Detect ridge crossings between behavioral basins.

    Compares variant measurements to a baseline to identify
    which linguistic modifiers triggered basin transitions.
    """
    import json

    context = _context(ctx)
    from modelcypher.core.domain.thermo.ridge_cross_detector import (
        RidgeCrossConfiguration,
        RidgeCrossDetector,
    )
    from modelcypher.core.domain.thermo.linguistic_thermodynamics import ThermoMeasurement

    # Load baseline
    try:
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        baseline = ThermoMeasurement.from_dict(baseline_data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        error = ErrorDetail(
            code="MC-1011",
            title="Failed to load baseline",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load variants
    try:
        with open(variants_file) as f:
            variants_data = json.load(f)
        variants = [ThermoMeasurement.from_dict(v) for v in variants_data]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        error = ErrorDetail(
            code="MC-1012",
            title="Failed to load variants",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Configure detector
    if preset == "strict":
        config = RidgeCrossConfiguration.strict()
    elif preset == "lenient":
        config = RidgeCrossConfiguration.lenient()
    else:
        config = RidgeCrossConfiguration.default()

    detector = RidgeCrossDetector(configuration=config)
    analysis = detector.analyze_transitions(baseline, variants)

    payload = {
        "totalCrossings": len(analysis.events),
        "solutionCrossings": analysis.solution_crossings,
        "mostEffectiveModifier": (
            analysis.most_effective_modifier.value
            if analysis.most_effective_modifier
            else None
        ),
        "meanSuccessfulDeltaH": analysis.mean_successful_delta_h,
        "thresholdDeltaH": analysis.threshold_delta_h,
        "events": [
            {
                "fromBasin": e.from_basin.value,
                "toBasin": e.to_basin.value,
                "triggerModifier": e.trigger_modifier.value,
                "deltaH": e.delta_h,
                "isSolutionCrossing": e.is_solution_crossing,
            }
            for e in analysis.events
        ],
    }

    if context.output_format == "text":
        lines = [
            "RIDGE CROSS DETECTION",
            "",
            analysis.summary,
            "",
            "Events:",
        ]
        for e in analysis.events:
            lines.append(f"  {e.description}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("phase")
def thermo_phase(
    ctx: typer.Context,
    logits_file: str = typer.Argument(..., help="Path to logits JSON file (array of floats)"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Generation temperature"),
    intensity: float = typer.Option(0.0, "--intensity", "-i", help="Modifier intensity [0, 1]"),
) -> None:
    """Analyze thermodynamic phase from logits.

    Determines whether the model is in ordered, critical, or
    disordered phase based on the critical temperature T_c.
    """
    import json

    context = _context(ctx)
    from modelcypher.core.domain.thermo.phase_transition_theory import (
        PhaseTransitionTheory,
    )

    # Load logits
    try:
        with open(logits_file) as f:
            logits = json.load(f)
        if not isinstance(logits, list) or not all(isinstance(x, (int, float)) for x in logits):
            raise ValueError("Logits must be a JSON array of numbers")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1013",
            title="Failed to load logits",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    analysis = PhaseTransitionTheory.analyze(
        logits=logits,
        temperature=temperature,
        intensity_score=intensity,
    )

    payload = {
        "temperature": analysis.temperature,
        "estimatedTc": analysis.estimated_tc,
        "phase": analysis.phase.value,
        "phaseDisplayName": analysis.phase.display_name,
        "expectedModifierEffect": analysis.phase.expected_modifier_effect,
        "logitVariance": analysis.logit_variance,
        "effectiveVocabSize": analysis.effective_vocab_size,
        "predictedModifierEffect": analysis.predicted_modifier_effect,
        "confidence": analysis.confidence,
        "basinWeights": (
            {
                "refusal": analysis.basin_weights.refusal,
                "caution": analysis.basin_weights.caution,
                "solution": analysis.basin_weights.solution,
            }
            if analysis.basin_weights
            else None
        ),
    }

    if context.output_format == "text":
        lines = [
            "PHASE TRANSITION ANALYSIS",
            "",
            f"Temperature: {analysis.temperature:.4f}",
            f"Estimated T_c: {analysis.estimated_tc:.4f}",
            f"Phase: {analysis.phase.display_name}",
            "",
            f"Expected Modifier Effect: {analysis.phase.expected_modifier_effect}",
            "",
            f"Logit Variance: {analysis.logit_variance:.4f}",
            f"Effective Vocab Size: {analysis.effective_vocab_size}",
            f"Predicted Modifier Effect: {analysis.predicted_modifier_effect:.4f}",
            f"Confidence: {analysis.confidence:.2%}",
        ]
        if analysis.basin_weights:
            lines.append("")
            lines.append("Basin Weights:")
            lines.append(f"  Refusal: {analysis.basin_weights.refusal:.3f}")
            lines.append(f"  Caution: {analysis.basin_weights.caution:.3f}")
            lines.append(f"  Solution: {analysis.basin_weights.solution:.3f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("sweep")
def thermo_sweep(
    ctx: typer.Context,
    logits_file: str = typer.Argument(..., help="Path to logits JSON file (array of floats)"),
    temps: Optional[str] = typer.Option(None, "--temps", help="Comma-separated temperatures to sweep"),
) -> None:
    """Perform temperature sweep to analyze entropy behavior.

    Sweeps across temperatures to find the critical temperature T_c
    where the entropy derivative peaks.
    """
    import json

    context = _context(ctx)
    from modelcypher.core.domain.thermo.phase_transition_theory import (
        PhaseTransitionTheory,
    )

    # Load logits
    try:
        with open(logits_file) as f:
            logits = json.load(f)
        if not isinstance(logits, list) or not all(isinstance(x, (int, float)) for x in logits):
            raise ValueError("Logits must be a JSON array of numbers")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        error = ErrorDetail(
            code="MC-1014",
            title="Failed to load logits",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse temperatures
    temperatures: Optional[list[float]] = None
    if temps:
        try:
            temperatures = [float(t.strip()) for t in temps.split(",")]
        except ValueError as exc:
            error = ErrorDetail(
                code="MC-1015",
                title="Invalid temperatures",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    result = PhaseTransitionTheory.temperature_sweep(logits=logits, temperatures=temperatures)

    payload = {
        "temperatures": result.temperatures,
        "entropies": result.entropies,
        "derivatives": result.derivatives,
        "estimatedTc": result.estimated_tc,
        "observedPeakT": result.observed_peak_t,
    }

    if context.output_format == "text":
        lines = [
            "TEMPERATURE SWEEP",
            "",
            f"Estimated T_c: {result.estimated_tc:.4f}",
            f"Observed Peak T: {result.observed_peak_t:.4f}" if result.observed_peak_t else "Observed Peak T: N/A",
            "",
            "Sweep Results:",
            f"{'Temp':>8}  {'Entropy':>10}  {'dH/dT':>10}",
            "-" * 32,
        ]
        for t, h, d in zip(result.temperatures, result.entropies, result.derivatives):
            lines.append(f"{t:>8.4f}  {h:>10.4f}  {d:>10.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("benchmark")
def thermo_benchmark(
    ctx: typer.Context,
    prompts_file: str = typer.Argument(..., help="Path to prompts file (JSON array or newline-separated)"),
    model: Optional[str] = typer.Option(None, "--model", help="Path to model directory (uses simulated if not provided)"),
    modifiers: Optional[str] = typer.Option(None, "--modifiers", help="Comma-separated modifiers (default: all)"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature"),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Max tokens per generation"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save markdown report to file"),
) -> None:
    """Run statistical benchmark comparing modifier effectiveness.

    Compares entropy effects across linguistic modifiers with Welch's t-test
    and Cohen's d effect sizes.
    """
    import json
    from pathlib import Path

    context = _context(ctx)
    from modelcypher.core.domain.thermo.benchmark_runner import ThermoBenchmarkRunner
    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
    from modelcypher.core.domain.thermo.linguistic_thermodynamics import LinguisticModifier

    # Load prompts
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        error = ErrorDetail(
            code="MC-1016",
            title="Prompts file not found",
            detail=f"File not found: {prompts_file}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        content = prompts_path.read_text()
        # Try JSON first
        try:
            prompts = json.loads(content)
            if not isinstance(prompts, list):
                raise ValueError("JSON must be an array")
        except json.JSONDecodeError:
            # Fall back to newline-separated
            prompts = [line.strip() for line in content.splitlines() if line.strip()]
    except Exception as exc:
        error = ErrorDetail(
            code="MC-1017",
            title="Failed to load prompts",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not prompts:
        error = ErrorDetail(
            code="MC-1018",
            title="No prompts found",
            detail="Prompts file is empty",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse modifiers
    modifier_list: Optional[list[LinguisticModifier]] = None
    if modifiers:
        try:
            modifier_list = [LinguisticModifier(m.strip().lower()) for m in modifiers.split(",")]
        except ValueError as exc:
            error = ErrorDetail(
                code="MC-1019",
                title="Invalid modifier",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    # Create calorimeter
    simulated = model is None
    calorimeter = LinguisticCalorimeter(
        model_path=model,
        simulated=simulated,
    )

    # Run benchmark
    runner = ThermoBenchmarkRunner(calorimeter=calorimeter)
    result = runner.run_modifier_comparison(
        prompts=prompts,
        modifiers=modifier_list,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Generate report
    report = runner.generate_report(result)

    # Save to file if requested
    if output_file:
        Path(output_file).write_text(report)

    payload = {
        "corpusSize": result.corpus_size,
        "baselineMean": result.baseline_mean,
        "baselineStd": result.baseline_std,
        "bestModifier": result.best_modifier.value,
        "bestEffectSize": result.best_effect_size,
        "modifiers": [
            {
                "modifier": s.modifier.value,
                "sampleSize": s.sample_size,
                "meanEntropy": s.mean_entropy,
                "stdEntropy": s.std_entropy,
                "ridgeCrossRate": s.ridge_cross_rate,
                "significance": (
                    {
                        "tStatistic": s.significance.t_statistic,
                        "pValue": s.significance.p_value,
                        "isSignificant": s.significance.is_significant,
                    }
                    if s.significance
                    else None
                ),
                "effectSize": (
                    {
                        "cohensD": s.effect_size.cohens_d,
                        "ciLower": s.effect_size.ci_lower,
                        "ciUpper": s.effect_size.ci_upper,
                        "interpretation": s.effect_size.interpretation,
                    }
                    if s.effect_size
                    else None
                ),
            }
            for s in result.modifiers
        ],
        "timestamp": result.timestamp.isoformat(),
    }

    if context.output_format == "text":
        write_output(report, context.output_format, context.pretty)
        if output_file:
            write_output(f"\nReport saved to: {output_file}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("parity")
def thermo_parity(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to test across languages"),
    model: Optional[str] = typer.Option(None, "--model", help="Path to model directory (uses simulated if not provided)"),
    modifier: str = typer.Option("caps", "--modifier", "-m", help="Modifier to test"),
    languages: Optional[str] = typer.Option(None, "--languages", "-l", help="Comma-separated languages (en,zh,ar,sw)"),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save markdown report to file"),
) -> None:
    """Run cross-lingual parity test for modifier consistency.

    Tests whether the entropy cooling pattern holds across languages with
    different resource levels (high, medium, low).
    """
    from pathlib import Path

    context = _context(ctx)
    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
    from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
        LinguisticModifier,
        PromptLanguage,
    )
    from modelcypher.core.domain.thermo.multilingual_calibrator import MultilingualCalibrator

    # Parse modifier
    try:
        test_modifier = LinguisticModifier(modifier.lower())
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1020",
            title="Invalid modifier",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse languages
    language_list: Optional[list[PromptLanguage]] = None
    if languages:
        try:
            language_list = [PromptLanguage(l.strip().lower()) for l in languages.split(",")]
        except ValueError as exc:
            error = ErrorDetail(
                code="MC-1021",
                title="Invalid language",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    # Create calorimeter and calibrator
    simulated = model is None
    calorimeter = LinguisticCalorimeter(
        model_path=model,
        simulated=simulated,
    )
    calibrator = MultilingualCalibrator()

    # Run parity test
    result = calibrator.cross_lingual_parity_test(
        prompt=prompt,
        modifier=test_modifier,
        calorimeter=calorimeter,
        languages=language_list,
        temperature=temperature,
    )

    # Generate report
    report = result.generate_markdown()

    # Save to file if requested
    if output_file:
        Path(output_file).write_text(report)

    payload = {
        "prompt": result.prompt,
        "modifier": result.modifier.value,
        "coolingPatternHolds": result.cooling_pattern_holds,
        "meanParityScore": result.mean_parity_score,
        "weakestLanguage": result.weakest_language.value if result.weakest_language else None,
        "strongestLanguage": result.strongest_language.value if result.strongest_language else None,
        "results": [
            {
                "language": r.language.value,
                "resourceLevel": r.language.resource_level.value,
                "baselineEntropy": r.baseline_entropy,
                "modifiedEntropy": r.modified_entropy,
                "deltaH": r.delta_h,
                "showsCooling": r.shows_cooling,
                "parityScore": r.parity_score,
            }
            for r in result.results
        ],
        "timestamp": result.timestamp.isoformat(),
    }

    if context.output_format == "text":
        write_output(report, context.output_format, context.pretty)
        if output_file:
            write_output(f"\nReport saved to: {output_file}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
