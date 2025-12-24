"""Geometry safety CLI commands.

Provides commands for:
- Circuit breaker evaluation
- Persona drift analysis
- Jailbreak testing
- Red team probes
- Behavioral safety probes

Commands:
    mc geometry safety circuit-breaker --job <id>
    mc geometry safety persona --job <id>
    mc geometry safety jailbreak-test --model <path>
    mc geometry safety probe-redteam --name <name>
    mc geometry safety probe-behavioral --name <name>
"""

from __future__ import annotations


import typer

from modelcypher.cli.composition import get_geometry_safety_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("circuit-breaker")
def geometry_safety_circuit_breaker(
    ctx: typer.Context,
    job_id: str | None = typer.Option(None, "--job"),
    entropy: float | None = typer.Option(None, "--entropy"),
    refusal_distance: float | None = typer.Option(None, "--refusal-distance"),
    persona_drift: float | None = typer.Option(None, "--persona-drift"),
    oscillation: bool = typer.Option(False, "--oscillation"),
) -> None:
    """Evaluate circuit breaker state.

    Examples:
        mc geometry safety circuit-breaker --job abc123
        mc geometry safety circuit-breaker --entropy 0.8 --refusal-distance 0.2
    """
    context = _context(ctx)
    service = get_geometry_safety_service()
    state, _signals = service.evaluate_circuit_breaker(
        job_id=job_id,
        entropy_signal=entropy,
        refusal_distance=refusal_distance,
        persona_drift_magnitude=persona_drift,
        has_oscillation=oscillation,
    )

    output = {
        "tripped": state.is_tripped,
        "severity": state.severity,
        "state": "tripped" if state.is_tripped else ("warning" if state.severity >= 0.5 else "nominal"),
        "interpretation": state.interpretation,
        "recommendedAction": state.recommended_action.description,
    }

    if context.output_format == "text":
        lines = [
            "CIRCUIT BREAKER EVALUATION",
            f"State: {output['state'].upper()}",
            f"Severity: {output['severity']:.3f}",
            f"Interpretation: {output['interpretation']}",
            f"Recommended Action: {output['recommendedAction']}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("persona")
def geometry_safety_persona(
    ctx: typer.Context,
    job_id: str = typer.Option(..., "--job"),
) -> None:
    """Analyze persona drift for a training job.

    Examples:
        mc geometry safety persona --job abc123
    """
    context = _context(ctx)
    service = get_geometry_safety_service()
    drift_info = service.persona_drift(job_id)
    if drift_info is None:
        raise typer.BadParameter(f"Job '{job_id}' not found or has no persona drift metrics.")

    output = {
        "jobId": job_id,
        "overallDriftMagnitude": drift_info.overall_drift_magnitude,
        "driftAssessment": drift_info.assessment,
        "driftingTraits": drift_info.drifting_traits,
        "refusalDistance": drift_info.refusal_distance,
        "isApproachingRefusal": drift_info.is_approaching_refusal,
    }

    if context.output_format == "text":
        lines = [
            "PERSONA DRIFT ANALYSIS",
            f"Job: {output['jobId']}",
            f"Drift Magnitude: {output['overallDriftMagnitude']:.4f} ({output['driftAssessment']})",
        ]
        if output["driftingTraits"]:
            lines.append(f"Drifting Traits: {', '.join(output['driftingTraits'])}")
        if output["refusalDistance"] is not None:
            approaching = "YES" if output.get("isApproachingRefusal") else "NO"
            lines.append(f"Refusal Distance: {output['refusalDistance']:.4f} (Approaching: {approaching})")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("jailbreak-test")
def geometry_safety_jailbreak_test(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompts: str | None = typer.Option(None, "--prompts", help="Path to prompts file (JSON array or newline-separated)"),
    prompt: list[str] | None = typer.Option(None, "--prompt", help="Individual prompt(s) to test"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter to apply"),
) -> None:
    """Execute jailbreak entropy analysis to test model safety boundaries.

    Examples:
        mc geometry safety jailbreak-test --model ./model --prompts ./prompts.json
        mc geometry safety jailbreak-test --model ./model --prompt "test prompt"
    """
    context = _context(ctx)

    # Collect prompts from file or individual --prompt flags
    prompt_list: list[str] = []
    if prompts:
        # Will be loaded from file by the service
        prompt_input: list[str] | str = prompts
    elif prompt:
        prompt_list = list(prompt)
        prompt_input = prompt_list
    else:
        raise typer.BadParameter("Provide either --prompts file or --prompt values")

    service = get_geometry_safety_service()
    result = service.jailbreak_test(
        model_path=model,
        prompts=prompt_input,
        adapter_path=adapter,
    )

    output = {
        "modelPath": result.model_path,
        "adapterPath": result.adapter_path,
        "promptsTested": result.prompts_tested,
        "vulnerabilitiesFound": result.vulnerabilities_found,
        "overallAssessment": result.overall_assessment,
        "riskScore": result.risk_score,
        "processingTime": result.processing_time,
        "vulnerabilityDetails": [
            {
                "prompt": v.prompt[:100] + "..." if len(v.prompt) > 100 else v.prompt,
                "vulnerabilityType": v.vulnerability_type,
                "severity": v.severity,
                "baselineEntropy": v.baseline_entropy,
                "attackEntropy": v.attack_entropy,
                "deltaH": v.delta_h,
                "confidence": v.confidence,
                "attackVector": v.attack_vector,
                "mitigationHint": v.mitigation_hint,
            }
            for v in result.vulnerability_details
        ],
        "nextActions": [
            "mc geometry safety circuit-breaker for combined safety assessment",
            "mc thermo detect for detailed entropy analysis",
        ],
    }

    if context.output_format == "text":
        lines = [
            "JAILBREAK TEST RESULTS",
            f"Model: {result.model_path}",
        ]
        if result.adapter_path:
            lines.append(f"Adapter: {result.adapter_path}")
        lines.append(f"Prompts Tested: {result.prompts_tested}")
        lines.append(f"Vulnerabilities Found: {result.vulnerabilities_found}")
        lines.append(f"Overall Assessment: {result.overall_assessment.upper()}")
        lines.append(f"Risk Score: {result.risk_score:.2f}")
        lines.append(f"Processing Time: {result.processing_time:.2f}s")

        if result.vulnerability_details:
            lines.append("")
            lines.append("VULNERABILITY DETAILS:")
            for i, v in enumerate(result.vulnerability_details[:10], 1):  # Limit to 10 in text output
                lines.append(f"  {i}. [{v.severity.upper()}] {v.vulnerability_type} via {v.attack_vector}")
                lines.append(f"     Prompt: {v.prompt[:60]}...")
                lines.append(f"     Delta H: {v.delta_h:.3f}, Confidence: {v.confidence:.2f}")
                lines.append(f"     Hint: {v.mitigation_hint}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(output, context.output_format, context.pretty)


@app.command("probe-redteam")
def geometry_safety_probe_redteam(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Scan adapter metadata for threat indicators (static analysis).

    Examples:
        mc geometry safety probe-redteam --name my-adapter
        mc geometry safety probe-redteam --name my-adapter --tag skill1 --tag skill2
    """
    context = _context(ctx)
    service = SafetyProbeService()

    indicators = service.scan_adapter_metadata(
        name=name,
        description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.threat_indicators_payload(indicators)
    payload["nextActions"] = [
        "mc geometry safety probe-behavioral for runtime safety checks",
        "mc geometry safety circuit-breaker for combined assessment",
    ]

    if context.output_format == "text":
        lines = [
            "RED TEAM STATIC ANALYSIS",
            f"Adapter: {name}",
            f"Status: {payload['status'].upper()}",
            f"Threat Indicators: {payload['count']}",
            f"Max Severity: {payload['maxSeverity']:.2f}",
        ]
        if indicators:
            lines.append("")
            lines.append("DETECTED THREATS:")
            for ind in indicators:
                lines.append(f"  [{ind.severity:.2f}] {ind.location}: {ind.description}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-behavioral")
def geometry_safety_probe_behavioral(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    tier: str = typer.Option("standard", "--tier", help="Safety tier: quick, standard, full"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Run behavioral safety probes (requires inference hook for full analysis).

    Examples:
        mc geometry safety probe-behavioral --name my-adapter
        mc geometry safety probe-behavioral --name my-adapter --tier full
    """
    import asyncio
    from modelcypher.core.domain.safety.behavioral_probes import AdapterSafetyTier

    context = _context(ctx)
    service = SafetyProbeService()

    tier_map = {
        "quick": AdapterSafetyTier.QUICK,
        "standard": AdapterSafetyTier.STANDARD,
        "full": AdapterSafetyTier.FULL,
    }
    safety_tier = tier_map.get(tier.lower(), AdapterSafetyTier.STANDARD)

    result = asyncio.run(service.run_behavioral_probes(
        adapter_name=name,
        tier=safety_tier,
        adapter_description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    ))

    payload = SafetyProbeService.composite_result_payload(result)
    payload["nextActions"] = [
        "mc geometry safety probe-redteam for static analysis",
        "mc geometry safety circuit-breaker for combined assessment",
    ]

    if context.output_format == "text":
        lines = [
            "BEHAVIORAL SAFETY PROBE RESULTS",
            f"Adapter: {name}",
            f"Tier: {tier.upper()}",
            f"Recommended Status: {payload['recommendedStatus'].upper()}",
            f"Aggregate Risk: {payload['aggregateRiskScore']:.2f}",
            f"Probes Run: {payload['probeCount']}",
        ]
        if payload["anyTriggered"]:
            lines.append("")
            lines.append("TRIGGERED PROBES:")
            for r in result.probe_results:
                if r.triggered:
                    lines.append(f"  [{r.risk_score:.2f}] {r.probe_name}: {r.details}")
        if payload["allFindings"]:
            lines.append("")
            lines.append("FINDINGS:")
            for finding in payload["allFindings"][:10]:
                lines.append(f"  - {finding}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
