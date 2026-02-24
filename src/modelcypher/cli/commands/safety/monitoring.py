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

"""Monitoring and runtime safety CLI commands.

Provides commands for runtime monitoring and analysis:
- circuit-breaker: Evaluate circuit breaker state
- persona: Analyze persona drift for a training job
- uncertainty-modes: List available uncertainty response modes
- entropy-pattern: Analyze entropy/variance samples for patterns
- entropy-baseline-verify: Verify observed entropy deltas against baseline
- crm-build: Build Concept Response Matrix for a model
- crm-compare: Compare two Concept Response Matrices
"""

from __future__ import annotations

from modelcypher.cli.composition import get_geometry_safety_service
from modelcypher.cli.exit_codes import EXIT_RUNTIME

from ._common import (
    ErrorDetail,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


@app.command("circuit-breaker")
def safety_circuit_breaker(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Evaluate circuit breaker state.

    Examples:
        mc safety circuit-breaker --job abc123
    """
    context = get_context(ctx)
    service = get_geometry_safety_service()
    state, _signals = service.evaluate_circuit_breaker(
        job_id=job_id,
    )

    output = {
        "severity": state.severity,
        "dominantSource": state.dominant_source.value if state.dominant_source else None,
    }

    if context.output_format == "text":
        lines = [
            "CIRCUIT BREAKER EVALUATION",
            f"Severity: {output['severity']:.3f}",
        ]
        if output["dominantSource"] is not None:
            lines.append(f"Dominant Source: {output['dominantSource']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("persona")
def safety_persona(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Analyze persona drift for a training job.

    Examples:
        mc safety persona --job abc123
    """
    context = get_context(ctx)
    service = get_geometry_safety_service()
    drift_info = service.persona_drift(job_id)
    if drift_info is None:
        raise typer.BadParameter(f"Job '{job_id}' not found or has no persona drift metrics.")

    output = {
        "jobId": job_id,
        "overallDriftMagnitude": drift_info.overall_drift_magnitude,
        "driftingTraits": drift_info.drifting_traits,
        "refusalDistance": drift_info.refusal_distance,
        "isApproachingRefusal": drift_info.is_approaching_refusal,
    }

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT ANALYSIS",
            f"Job: {output['jobId']}",
            f"Drift Magnitude: {output['overallDriftMagnitude']:.4f}",
        ]
        if output["driftingTraits"]:
            lines.append(f"Drifting Traits: {', '.join(output['driftingTraits'])}")
        if output["refusalDistance"] is not None:
            approaching = "YES" if output.get("isApproachingRefusal") else "NO"
            lines.append(
                f"Refusal Distance: {output['refusalDistance']:.4f} (Approaching: {approaching})"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("uncertainty-modes")
def uncertainty_modes(
    ctx: typer.Context,
) -> None:
    """List available uncertainty response modes.

    Modes determine how the model responds when uncertainty is detected:
    - Butler: No exploration, answer with available knowledge or decline
    - Autonomous: Research gaps automatically (query memory, search)
    - Human-in-loop: Pause at uncertainty, ask for guidance

    Examples:
        mc analyze uncertainty-modes
    """
    from modelcypher.core.use_cases.entropy_monitor import (
        UncertaintyAction,
        UncertaintyMode,
    )

    context = get_context(ctx)

    payload = {
        "modes": [
            {
                "name": mode.value,
                "description": {
                    "butler": "No exploration. Answer with available knowledge or decline.",
                    "autonomous": "Research gaps automatically. Query memory, search, augment context.",
                    "human_in_loop": "Pause at uncertainty. Ask user for guidance before proceeding.",
                }[mode.value],
            }
            for mode in UncertaintyMode
        ],
        "actions": [
            {
                "name": action.value,
                "description": {
                    "proceed": "Continue generation - uncertainty is acceptable.",
                    "abstain": "Stop generation - uncertainty too high.",
                    "retrieve": "Pause and retrieve - query memory/search before continuing.",
                    "ask": "Pause and ask - request user guidance.",
                    "warn": "Continue with warning - trajectory instability detected.",
                }[action.value],
            }
            for action in UncertaintyAction
        ],
    }

    if context.output_format == "text":
        lines = [
            "UNCERTAINTY RESPONSE MODES",
            "",
            "MODES (user-configurable):",
        ]
        for m in payload["modes"]:
            lines.append(f"  {m['name']}: {m['description']}")
        lines.append("")
        lines.append("ACTIONS (system-determined):")
        for a in payload["actions"]:
            lines.append(f"  {a['name']}: {a['description']}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy-pattern")
def entropy_pattern_analysis(
    ctx: typer.Context,
    samples_file: str = typer.Option(..., "--samples", "-s", help="JSON file with (entropy, variance) samples"),
    detect_distress: bool = typer.Option(False, "--detect-distress", "-d", help="Detect distress patterns"),
) -> None:
    """Analyze entropy/variance samples for patterns.

    Detects trends, anomalies, and potential distress signals in
    entropy time series data.

    Examples:
        mc analyze entropy-pattern --samples samples.json
        mc analyze entropy-pattern --samples samples.json --detect-distress
    """
    import json

    from modelcypher.cli.composition import get_entropy_probe_service

    context = get_context(ctx)
    service = get_entropy_probe_service()

    # Load samples from file
    try:
        with open(samples_file) as f:
            data = json.load(f)
        samples = [(s["entropy"], s["variance"]) for s in data["samples"]]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        payload = {"error": f"Failed to load samples: {e}"}
        write_output(payload, context.output_format, context.pretty)
        return

    pattern = service.analyze_pattern(samples)

    if detect_distress:
        distress = service.detect_distress(samples)
        payload = {
            "pattern": {
                "trend": pattern.trend.value if hasattr(pattern.trend, "value") else str(pattern.trend),
                "sample_count": len(samples),
            },
            "distress_detected": distress is not None,
            "distress": {
                "type": distress.distress_type if distress else None,
                "confidence": distress.confidence if distress else None,
            } if distress else None,
        }
    else:
        payload = {
            "pattern": {
                "trend": pattern.trend.value if hasattr(pattern.trend, "value") else str(pattern.trend),
                "sample_count": len(samples),
            },
        }

    write_output(payload, context.output_format, context.pretty)


@app.command("entropy-baseline-verify")
def entropy_baseline_verify(
    ctx: typer.Context,
    baseline_file: str = typer.Option(..., "--baseline", "-b", help="Path to baseline JSON"),
    deltas_file: str = typer.Option(..., "--deltas", "-d", help="JSON file with observed delta values"),
    adapter_path: str | None = typer.Option(None, "--adapter", "-a", help="Path to adapter (for reporting)"),
) -> None:
    """Verify observed entropy deltas against declared baseline.

    Compares observed delta values against a previously computed baseline
    to detect unexpected entropy shifts.

    Examples:
        mc analyze entropy-baseline-verify --baseline baseline.json --deltas observed.json
    """
    import json

    from modelcypher.cli.composition import get_entropy_probe_service

    context = get_context(ctx)
    service = get_entropy_probe_service()

    # Load deltas
    try:
        with open(deltas_file) as f:
            data = json.load(f)
        deltas = data.get("deltas", data.get("values", []))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        payload = {"error": f"Failed to load deltas: {e}"}
        write_output(payload, context.output_format, context.pretty)
        return

    result = service.verify_baseline(
        baseline_path=baseline_file,
        observed_deltas=deltas,
        adapter_path=adapter_path,
    )

    payload = {
        "baseline_file": baseline_file,
        "deltas_file": deltas_file,
        "adapter_path": adapter_path,
        "verified": result.verified,
        "comparison": {
            "mean_difference": result.comparison.mean_difference,
            "std_difference": result.comparison.std_difference,
        } if hasattr(result, "comparison") else None,
    }

    write_output(payload, context.output_format, context.pretty)


@app.command("crm-build")
def crm_build(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to model"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for CRM"),
    adapter: str | None = typer.Option(None, "--adapter", "-a", help="Optional adapter path"),
) -> None:
    """Build Concept Response Matrix for a model.

    Computes activations for semantic anchors (primes, gates, emotions)
    and stores them for cross-architecture comparison.

    Examples:
        mc analyze crm-build /path/to/model --output ./crm/model1
        mc analyze crm-build /path/to/model --output ./crm/model1 --adapter ./adapter
    """
    from modelcypher.core.use_cases.concept_response_matrix_service import (
        ConceptResponseMatrixService,
    )
    from modelcypher.ports.inference import HiddenStateEngine

    context = get_context(ctx)

    typer.echo(f"Building CRM for: {model}")

    try:
        # Create HiddenStateEngine for the model
        engine = HiddenStateEngine(model_path=model)

        # Create service with the engine
        service = ConceptResponseMatrixService(engine=engine)

        # Build CRM
        result = service.build(
            model_path=model,
            output_path=output,
            adapter=adapter,
        )

        payload = {
            "modelPath": result.model_path,
            "outputPath": result.output_path,
            "layerCount": result.layer_count,
            "hiddenDim": result.hidden_dim,
            "anchorCount": result.anchor_count,
            "primeCount": result.prime_count,
            "gateCount": result.gate_count,
            "sequenceInvariantCount": result.sequence_invariant_count,
            "emotionCount": result.emotion_count,
            "primeNumberCount": result.prime_number_count,
        }

        if context.output_format == "text":
            lines = [
                "CRM BUILD COMPLETE",
                f"Model: {result.model_path}",
                f"Output: {result.output_path}",
                "",
                f"Layers: {result.layer_count}",
                f"Hidden Dim: {result.hidden_dim}",
                "",
                "ANCHORS PROCESSED:",
                f"  Total: {result.anchor_count}",
                f"  Semantic Primes: {result.prime_count}",
                f"  Computational Gates: {result.gate_count}",
                f"  Sequence Invariants: {result.sequence_invariant_count}",
                f"  Emotions: {result.emotion_count}",
                f"  Prime Numbers: {result.prime_number_count}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4004",
            title="CRM build failed",
            detail=str(e),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)


@app.command("crm-compare")
def crm_compare(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Path to source CRM"),
    target: str = typer.Argument(..., help="Path to target CRM"),
    include_matrix: bool = typer.Option(False, "--include-matrix", help="Include full CKA matrix in output"),
) -> None:
    """Compare two Concept Response Matrices.

    Computes CKA alignment between source and target CRMs to measure
    cross-architecture semantic similarity.

    Examples:
        mc analyze crm-compare ./crm/model1 ./crm/model2
    """
    from modelcypher.cli.composition import get_concept_response_matrix_service

    context = get_context(ctx)

    typer.echo(f"Comparing CRMs: {source} vs {target}")

    try:
        service = get_concept_response_matrix_service()

        result = service.compare(
            source_path=source,
            target_path=target,
            include_matrix=include_matrix,
        )

        payload = {
            "sourcePath": result.source_path,
            "targetPath": result.target_path,
            "commonAnchorCount": result.common_anchor_count,
            "meanCKA": result.mean_cka,
            "alignmentPrecision": result.alignment_precision,
            "aligned": result.aligned,
            "layerCorrespondence": result.layer_correspondence,
        }
        if result.cka_matrix is not None:
            payload["ckaMatrix"] = result.cka_matrix

        if context.output_format == "text":
            status = "ALIGNED" if result.aligned else "MISALIGNED"
            lines = [
                "CRM COMPARISON",
                f"Source: {result.source_path}",
                f"Target: {result.target_path}",
                "",
                f"Common Anchors: {result.common_anchor_count}",
                f"Mean CKA: {result.mean_cka:.4f}",
                f"Alignment Precision: {result.alignment_precision:.4f}",
                f"Status: {status}",
                "",
                "LAYER CORRESPONDENCE:",
            ]
            for corr in result.layer_correspondence[:10]:
                lines.append(
                    f"  Source L{corr['sourceLayer']} -> Target L{corr['targetLayer']}: "
                    f"CKA={corr['cka']:.4f}"
                )
            if len(result.layer_correspondence) > 10:
                lines.append(f"  ... and {len(result.layer_correspondence) - 10} more layers")
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4005",
            title="CRM comparison failed",
            detail=str(e),
            hint="Check model path and backend status (mc system status).",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)
