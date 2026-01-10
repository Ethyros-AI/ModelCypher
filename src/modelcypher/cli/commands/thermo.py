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

"""Thermodynamic analysis CLI commands.

Provides commands for thermodynamic analysis of model training,
including entropy measurement, path integration, and differential analysis.

Commands:
    mc thermo analyze <job_id>
    mc thermo path --checkpoint <path> ...
    mc thermo path-integration <prompt> --model <path>
    mc thermo entropy <job_id>
    mc thermo measure <prompt> --model <path>
    mc thermo detect <prompt> --model <path>
    mc thermo detect-batch <prompts_file> --model <path>
"""

from __future__ import annotations

import json
import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.validation import validate_model_path
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _get_thermo_service(embedder=None):
    from modelcypher.cli.composition import get_model_loader
    from modelcypher.core.use_cases.thermo_service import ThermoService

    return ThermoService(
        embedder=embedder,
        model_loader=get_model_loader(),
    )


@app.command("analyze")
def thermo_analyze(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID to analyze"),
) -> None:
    """Thermodynamic analysis of training."""
    context = _context(ctx)
    service = _get_thermo_service()
    result = service.analyze(job_id)

    payload = {
        "jobId": result.job_id,
        "entropy": result.entropy,
        "temperature": result.temperature,
        "freeEnergy": result.free_energy,
    }

    if context.output_format == "text":
        lines = [
            "THERMO ANALYSIS",
            f"Job: {result.job_id}",
            f"Entropy: {result.entropy:.4f}",
            f"Temperature: {result.temperature:.4f}",
            f"Free Energy: {result.free_energy:.4f}",
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

    # Validate checkpoint paths early for clear error messages
    from modelcypher.cli.validation import validate_file_exists
    for checkpoint in checkpoints:
        validate_file_exists(checkpoint, description="Checkpoint", context=context)

    service = _get_thermo_service()
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
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("path-integration")
def thermo_path_integration(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to analyze"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
) -> None:
    """Integrate entropy trajectories with gate detections.

    Returns ALL gates with their similarity scores.
    The similarity scores ARE the signal - no threshold filtering.
    """
    context = _context(ctx)

    # Validate model path early for clear error messages
    validate_model_path(model, context=context)

    from modelcypher.adapters.embedding_defaults import EmbeddingDefaults

    service = _get_thermo_service(embedder=EmbeddingDefaults.make_default_embedder())
    result = service.path_integration(
        prompt=prompt,
        model_path=model,
    )

    measurement = result.measurement
    assessment = measurement.assessment

    payload = {
        "_schema": "mc.thermo.path_integration.v1",
        "modelId": result.model_id,
        "prompt": result.prompt,
        "responseText": result.response_text,
        "meanEntropy": measurement.mean_entropy,
        "entropyVariance": measurement.entropy_variance,
        "firstTokenEntropy": measurement.first_token_entropy,
        "entropyTrajectory": measurement.entropy_trajectory,
        "gateSequence": measurement.gate_sequence,
        "gateCount": measurement.gate_count,
        "gateDetails": [
            {
                "gateId": gate.gate_id,
                "gateName": gate.gate_name,
                "localEntropy": gate.local_entropy,
                "similarity": gate.similarity,
            }
            for gate in measurement.gate_details
        ],
        "entropyPathCorrelation": measurement.entropy_path_correlation,
        "gateTransitionEntropies": [
            {
                "fromGate": item.from_gate,
                "toGate": item.to_gate,
                "entropyDelta": item.entropy_delta,
            }
            for item in measurement.gate_transition_entropies
        ],
        "assessment": {
            "correlation": assessment.correlation,
            "spikeRate": assessment.spike_rate,
            "measurementCount": assessment.measurement_count,
        },
    }

    if context.output_format == "text":
        sequence = " -> ".join(measurement.gate_sequence) if measurement.gate_sequence else "(none)"
        corr_text = (
            f"{assessment.correlation:.3f}" if assessment.correlation is not None else "N/A"
        )
        lines = [
            "THERMO PATH INTEGRATION",
            f"Model: {result.model_id}",
            f"Mean Entropy: {measurement.mean_entropy:.3f}",
            f"Entropy Variance: {measurement.entropy_variance:.3f}",
            f"Gate Count: {measurement.gate_count}",
            f"Gate Sequence: {sequence}",
            f"Entropy-Gate Correlation: {corr_text}",
            f"Spike Rate: {assessment.spike_rate:.3f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy")
def thermo_entropy(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Job ID"),
) -> None:
    """Entropy metrics over training."""
    context = _context(ctx)
    service = _get_thermo_service()
    result = service.entropy(job_id)

    payload = {
        "jobId": result.job_id,
        "entropyHistory": result.entropy_history,
        "finalEntropy": result.final_entropy,
        "entropyDelta": result.entropy_delta,
        "entropyRatio": result.entropy_ratio,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("measure")
def thermo_measure(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to measure"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
) -> None:
    """Measure entropy across linguistic modifiers for a prompt."""
    context = _context(ctx)

    # Validate model path early for clear error messages
    validate_model_path(model, context=context)

    service = _get_thermo_service()
    result = service.measure(prompt, model)

    payload = {
        "basePrompt": result.base_prompt,
        "measurements": [
            {
                "modifier": m.modifier,
                "meanEntropy": m.mean_entropy,
                "deltaH": m.delta_h,
            }
            for m in result.measurements
        ],
        "statistics": {
            "meanEntropy": result.statistics.mean_entropy,
            "stdEntropy": result.statistics.std_entropy,
            "minEntropy": result.statistics.min_entropy,
            "maxEntropy": result.statistics.max_entropy,
            "meanDeltaH": result.statistics.mean_delta_h,
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
            lines.append(
                f"  {m.modifier}: entropy={m.mean_entropy:.4f}, delta_h={delta_str}"
            )
        lines.append("")
        lines.append(f"Mean Entropy: {result.statistics.mean_entropy:.4f}")
        lines.append(f"Std Entropy: {result.statistics.std_entropy:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("detect")
def thermo_detect(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to analyze"),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
) -> None:
    """Measure prompt entropy differential."""
    context = _context(ctx)

    # Validate model path early for clear error messages
    validate_model_path(model, context=context)

    service = _get_thermo_service()
    result = service.detect(prompt, model)

    payload = {
        "prompt": result.prompt,
        "baselineEntropy": result.baseline_entropy,
        "intensityEntropy": result.intensity_entropy,
        "deltaH": result.delta_h,
        "processingTime": result.processing_time,
    }

    if context.output_format == "text":
        lines = [
            "THERMO DETECT",
            f"Prompt: {result.prompt[:50]}{'...' if len(result.prompt) > 50 else ''}",
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
    prompts_file: str = typer.Argument(
        ..., help="Path to prompts file (JSON array or newline-separated)"
    ),
    model: str = typer.Option(..., "--model", help="Path to model directory"),
) -> None:
    """Batch measure entropy differentials across multiple prompts."""
    context = _context(ctx)

    # Validate inputs early for clear error messages
    from modelcypher.cli.validation import validate_file_exists
    validate_file_exists(prompts_file, description="Prompts file", context=context)
    validate_model_path(model, context=context)

    service = _get_thermo_service()

    try:
        results = service.detect_batch(prompts_file, model)
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
                "baselineEntropy": r.baseline_entropy,
                "intensityEntropy": r.intensity_entropy,
                "deltaH": r.delta_h,
                "processingTime": r.processing_time,
            }
            for r in results
        ],
    }

    if context.output_format == "text":
        lines = [
            "THERMO DETECT BATCH",
            f"File: {prompts_file}",
            f"Total Prompts: {len(results)}",
            "",
            "Results:",
        ]
        for i, r in enumerate(results[:10]):  # Show first 10
            prompt_preview = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
            lines.append(f"  {i + 1}. ΔH={r.delta_h:.4f} {prompt_preview}")
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
) -> None:
    """Detect ridge crossings between behavioral basins.

    Compares variant measurements to a baseline to identify
    which linguistic modifiers triggered basin transitions.
    """
    import json

    context = _context(ctx)

    # Validate file paths early for clear error messages
    from modelcypher.cli.validation import validate_file_exists
    validate_file_exists(baseline_file, description="Baseline file", context=context)
    validate_file_exists(variants_file, description="Variants file", context=context)

    from modelcypher.core.domain.thermo.linguistic_thermodynamics import ThermoMeasurement
    from modelcypher.core.domain.thermo.ridge_cross_detector import RidgeCrossDetector

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

    detector = RidgeCrossDetector()
    analysis = detector.analyze_transitions(baseline, variants)

    payload = {
        "totalCrossings": len(analysis.events),
        "events": [
            {
                "fromBasin": e.from_basin.value,
                "toBasin": e.to_basin.value,
                "triggerModifier": e.trigger_modifier.value,
                "deltaH": e.delta_h,
            }
            for e in analysis.events
        ],
    }

    if context.output_format == "text":
        lines = [
            "RIDGE CROSS DETECTION",
            "",
            f"Total crossings: {len(analysis.events)}",
        ]
        if analysis.events:
            lines.append("")
            lines.append("Events:")
            for e in analysis.events:
                lines.append(
                    f"  {e.trigger_modifier.value}: {e.from_basin.value} -> {e.to_basin.value} "
                    f"(delta_H={e.delta_h:.4f})"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("phase")
def thermo_phase(
    ctx: typer.Context,
    logits_file: str = typer.Argument(..., help="Path to logits JSON file (array of floats)"),
) -> None:
    """Analyze thermodynamic phase from logits.

    Computes critical temperature and entropy statistics from logits.
    Returns raw measurements only (no phase labels or expected effects).
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

    # AUTO-DERIVE temperature using critical temperature formula
    # Formula: T_c = σ_z / √(2 × ln(V_eff))
    # This is derived from statistical mechanics (Boltzmann-softmax equivalence)
    stats = PhaseTransitionTheory.compute_logit_statistics(logits)
    v_eff = PhaseTransitionTheory.effective_vocabulary_size(logits, temperature=1.0)
    derived_temperature = PhaseTransitionTheory.estimate_critical_temperature(
        logit_std_dev=stats.std_dev,
        effective_vocab_size=v_eff,
    )

    analysis = PhaseTransitionTheory.analyze(
        logits=logits,
        temperature=derived_temperature,
    )

    payload = {
        "temperature": analysis.temperature,
        "estimatedTc": analysis.estimated_tc,
        "temperatureRatio": (
            analysis.temperature / analysis.estimated_tc if analysis.estimated_tc > 0 else None
        ),
        "logitVariance": analysis.logit_variance,
        "effectiveVocabSize": analysis.effective_vocab_size,
        "entropy": analysis.entropy,
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
        ratio = (
            analysis.temperature / analysis.estimated_tc if analysis.estimated_tc > 0 else None
        )
        ratio_text = f"{ratio:.4f}" if ratio is not None else "N/A"
        lines = [
            "PHASE TRANSITION ANALYSIS",
            "",
            f"Temperature: {analysis.temperature:.4f}",
            f"Estimated T_c: {analysis.estimated_tc:.4f}",
            f"T/T_c Ratio: {ratio_text}",
            "",
            f"Entropy: {analysis.entropy:.4f} nats",
            f"Logit Variance: {analysis.logit_variance:.4f}",
            f"Effective Vocab Size: {analysis.effective_vocab_size}",
        ]
        if analysis.basin_weights:
            lines.append("")
            lines.append("Basin Weights (requires calibrated topology):")
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

    result = PhaseTransitionTheory.temperature_sweep(logits=logits, temperatures=None)

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
            f"Observed Peak T: {result.observed_peak_t:.4f}"
            if result.observed_peak_t
            else "Observed Peak T: N/A",
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
    prompts_file: str = typer.Argument(
        ..., help="Path to prompts file (JSON array or newline-separated)"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Path to model directory (uses simulated if not provided)"
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save markdown report to file"
    ),
) -> None:
    """Run statistical benchmark comparing modifier effectiveness.

    Compares entropy effects across linguistic modifiers with Welch's t-test
    and Cohen's d effect sizes.
    """
    import json
    from pathlib import Path

    context = _context(ctx)

    # Validate inputs early for clear error messages
    from modelcypher.cli.validation import validate_file_exists
    validate_file_exists(prompts_file, description="Prompts file", context=context)
    if model is not None:
        validate_model_path(model, context=context)

    from modelcypher.core.domain.thermo.benchmark_runner import ThermoBenchmarkRunner
    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter

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

    # Create calorimeter
    from modelcypher.cli.composition import get_model_loader

    simulated = model is None
    calorimeter = LinguisticCalorimeter(
        model_path=model,
        simulated=simulated,
        model_loader=get_model_loader(),
    )

    # Run benchmark
    runner = ThermoBenchmarkRunner(calorimeter=calorimeter)
    result = runner.run_modifier_comparison(
        prompts=prompts,
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
                        "degreesOfFreedom": s.significance.degrees_of_freedom,
                    }
                    if s.significance
                    else None
                ),
                "effectSize": (
                    {
                        "cohensD": s.effect_size.cohens_d,
                        "standardError": s.effect_size.standard_error,
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
    model: str | None = typer.Option(
        None, "--model", help="Path to model directory (uses simulated if not provided)"
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save JSON report to file"
    ),
) -> None:
    """Run cross-lingual parity test for modifier consistency.

    Measures entropy deltas across languages for all modifiers.
    """
    from pathlib import Path

    context = _context(ctx)

    # Validate model path early for clear error messages
    if model is not None:
        validate_model_path(model, context=context)

    from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter
    from modelcypher.core.domain.thermo.linguistic_thermodynamics import (
        LinguisticModifier,
        PromptLanguage,
    )
    from modelcypher.core.domain.thermo.multilingual_calibrator import MultilingualCalibrator

    # Create calorimeter and calibrator
    from modelcypher.cli.composition import get_model_loader

    simulated = model is None
    calorimeter = LinguisticCalorimeter(
        model_path=model,
        simulated=simulated,
        model_loader=get_model_loader(),
    )
    calibrator = MultilingualCalibrator()

    modifiers = [m for m in LinguisticModifier if m != LinguisticModifier.BASELINE]
    reports = [
        calibrator.cross_lingual_parity_test(
            prompt=prompt,
            modifier=modifier,
            calorimeter=calorimeter,
        )
        for modifier in modifiers
    ]

    payload = {
        "prompt": prompt,
        "modifiersTested": [m.value for m in modifiers],
        "languagesTested": [lang.value for lang in PromptLanguage],
        "reports": [
            {
                "modifier": report.modifier.value,
                "referenceLanguage": report.reference_language.value
                if report.reference_language
                else None,
                "effectVariance": report.effect_variance,
                "results": [
                    {
                        "language": r.language.value,
                        "resourceScore": r.language.resource_score,
                        "baselineEntropy": r.baseline_entropy,
                        "modifiedEntropy": r.modified_entropy,
                        "deltaH": r.delta_h,
                        "relativeEffect": r.relative_effect,
                    }
                    for r in report.results
                ],
                "timestamp": report.timestamp.isoformat(),
            }
            for report in reports
        ],
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))

    if context.output_format == "text":
        lines = [
            "THERMO PARITY",
            f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
            f"Modifiers: {', '.join(m.value for m in modifiers)}",
            f"Languages: {', '.join(lang.value for lang in PromptLanguage)}",
        ]
        for report in reports:
            lines.append(f"  {report.modifier.value}: variance={report.effect_variance:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        if output_file:
            write_output(f"\nReport saved to: {output_file}", context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("calibrate")
def thermo_calibrate(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Path to model directory"),
    output: str = typer.Option(
        None, "--output", "-o", help="Output path for calibration JSON"
    ),
    probes_file: str | None = typer.Option(
        None, "--probes", "-p", help="Path to custom probes file (JSON array or newline-separated)"
    ),
) -> None:
    """Calibrate thermodynamic parameters from empirical measurement.

    This command measures actual entropy distributions and behavioral outcomes
    to derive calibrated energy levels and modifier effects. The result
    replaces hardcoded placeholder values with physics derived from real
    observations.

    The calibration measures:
    - Basin energy levels: E(x) = -T * log(p(x)/p(ref))
    - Modifier intensity scores: measured delta_H for each modifier
    - Entropy percentiles from baseline distributions

    Example:
        mc thermo calibrate --model /path/to/model -o calibration.json
        mc thermo calibrate --model /path/to/model --probes prompts.txt -o cal.json
    """
    import json
    from pathlib import Path

    context = _context(ctx)
    from modelcypher.core.domain.thermo.thermo_calibrator import (
        ThermoCalibrator,
        get_default_calibration_probes,
    )

    model_path = Path(model).expanduser().resolve()
    if not model_path.exists():
        error = ErrorDetail(
            code="MC-1030",
            title="Model not found",
            detail=f"Model directory not found: {model}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Load probes
    if probes_file:
        probes_path = Path(probes_file)
        if not probes_path.exists():
            error = ErrorDetail(
                code="MC-1031",
                title="Probes file not found",
                detail=f"File not found: {probes_file}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

        try:
            content = probes_path.read_text()
            try:
                probes = json.loads(content)
                if not isinstance(probes, list):
                    raise ValueError("JSON must be an array of strings")
            except json.JSONDecodeError:
                probes = [line.strip() for line in content.splitlines() if line.strip()]
        except Exception as exc:
            error = ErrorDetail(
                code="MC-1032",
                title="Failed to load probes",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
    else:
        probes = get_default_calibration_probes()

    if not probes:
        error = ErrorDetail(
            code="MC-1033",
            title="Insufficient probes",
            detail="Calibration probes are empty",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    from modelcypher.cli.composition import get_model_loader

    calibrator = ThermoCalibrator(
        model_path=model,
        model_loader=get_model_loader(),
    )

    # Progress callback for text output
    def progress_callback(current: int, total: int, progress) -> None:
        if context.output_format == "text":
            pct = (current + 1) / total * 100
            print(f"\rCalibrating: {current + 1}/{total} ({pct:.1f}%)", end="", flush=True)

    # Run calibration
    try:
        calibration = calibrator.calibrate(
            probes=probes,
            progress_callback=progress_callback if context.output_format == "text" else None,
        )
    except Exception as exc:
        if context.output_format == "text":
            print()  # Newline after progress
        error = ErrorDetail(
            code="MC-1034",
            title="Calibration failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if context.output_format == "text":
        print()  # Newline after progress

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"{model_path.name}_thermo_calibration.json")

    # Save calibration
    calibration.save(output_path)

    # Build payload
    payload = {
        "_schema": "mc.thermo.calibrate.v1",
        "modelId": calibration.model_id,
        "outputPath": str(output_path),
        "isComplete": calibration.is_complete,
        "basinTopology": (
            {
                "refusalEnergy": calibration.basin_topology.refusal_energy.value,
                "cautionEnergy": calibration.basin_topology.caution_energy.value,
                "solutionEnergy": calibration.basin_topology.solution_energy.value,
                "transitionRidge": calibration.basin_topology.transition_ridge.value,
                "temperature": calibration.basin_topology.temperature,
            }
            if calibration.basin_topology
            else None
        ),
        "modifierProfile": (
            {
                "modifierCount": len(calibration.modifier_profile.effects),
                "effects": {
                    name: {
                        "meanDeltaH": effect.mean_delta_h,
                        "intensityScore": effect.intensity_score,
                        "sampleCount": effect.sample_count,
                    }
                    for name, effect in calibration.modifier_profile.effects.items()
                }
            }
            if calibration.modifier_profile
            else None
        ),
        "entropyPercentiles": calibration.thresholds.percentiles if calibration.thresholds else None,
        "timestamp": calibration.calibration_timestamp.isoformat(),
    }

    if context.output_format == "text":
        lines = [
            "THERMODYNAMIC CALIBRATION COMPLETE",
            "",
            f"Model: {calibration.model_id}",
            f"Output: {output_path}",
            f"Complete: {'yes' if calibration.is_complete else 'no'}",
            "",
        ]

        if calibration.basin_topology:
            bt = calibration.basin_topology
            lines.extend([
                "Basin Topology (measured energies):",
                f"  Refusal:    E = {bt.refusal_energy.value:.4f} (reference)",
                f"  Caution:    E = {bt.caution_energy.value:.4f}",
                f"  Solution:   E = {bt.solution_energy.value:.4f}",
                f"  Ridge:      E = {bt.transition_ridge.value:.4f}",
                "",
            ])

        if calibration.modifier_profile:
            lines.append("Modifier Effects (measured delta_H):")
            for name, effect in calibration.modifier_profile.effects.items():
                lines.append(
                    f"  {name}: delta_H = {effect.mean_delta_h:+.4f}, "
                    f"intensity = {effect.intensity_score:.2f}"
                )
            lines.append("")

        if calibration.thresholds:
            th = calibration.thresholds.percentiles
            lines.extend([
                "Entropy Percentiles (measured):",
                f"  p25: {th.get(25, 0.0):.4f}",
                f"  p50: {th.get(50, 0.0):.4f}",
                f"  p75: {th.get(75, 0.0):.4f}",
                f"  p95: {th.get(95, 0.0):.4f}",
                "",
            ])

        lines.append("Calibration saved. Use with:")
        lines.append("  from modelcypher.core.domain.thermo import ThermoCalibration")
        lines.append(f"  cal = ThermoCalibration.load(Path('{output_path}'))")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
