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

"""Probe and red-team safety CLI commands.

Provides commands for probing and testing model safety:
- jailbreak-test: Execute jailbreak entropy analysis
- probe-redteam: Scan adapter metadata for threat indicators
- probe-behavioral: Run behavioral probes
- bilm-probe-info: Show information about BiLM probe training
"""

from __future__ import annotations

from ._common import (
    Path,
    typer,
    write_error,
    write_output,
    warn_network,
    get_context,
    ErrorDetail,
)
from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

app = typer.Typer(no_args_is_help=True)


@app.command("jailbreak-test")
def safety_jailbreak_test(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model directory"),
    prompts: str | None = typer.Option(
        None, "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    prompt: list[str] | None = typer.Option(None, "--prompt", help="Individual prompt(s) to test"),
    adapter: str | None = typer.Option(None, "--adapter", help="Path to adapter to apply"),
    calibration: str = typer.Option(
        ...,
        "--calibration",
        help=(
            "Path to calibration JSON with drift, safe delta-H, and attack entropy "
            "sample arrays"
        ),
    ),
) -> None:
    """Execute jailbreak entropy analysis to test model safety boundaries.

    Calibration samples are required; fixed default thresholds are not used.

    Examples:
        mc safety jailbreak-test --model ./model --prompts ./prompts.json --calibration ./calibration.json
        mc safety jailbreak-test --model ./model --prompt "test prompt" --calibration ./calibration.json
    """
    import json

    from modelcypher.cli.composition import get_geometry_safety_service

    context = get_context(ctx)

    # Collect prompts from file or individual --prompt flags
    prompt_list: list[str] = []
    if prompts:
        # Load from file
        prompts_path = Path(prompts)
        if not prompts_path.exists():
            raise typer.BadParameter(f"Prompts file not found: {prompts}")

        content = prompts_path.read_text()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                prompt_list = [str(p) for p in data]
            else:
                prompt_list = data.get("prompts", [])
        except json.JSONDecodeError:
            # Treat as newline-separated
            prompt_list = [line.strip() for line in content.splitlines() if line.strip()]
    elif prompt:
        prompt_list = list(prompt)
    else:
        raise typer.BadParameter("Provide either --prompts file or --prompt values")

    if not prompt_list:
        raise typer.BadParameter("No prompts provided")

    try:
        calibration_path = Path(calibration)
        if not calibration_path.exists():
            raise typer.BadParameter(f"Calibration file not found: {calibration}")

        calibration_data = json.loads(calibration_path.read_text())

        def _read_samples(*keys: str) -> list[float]:
            for key in keys:
                value = calibration_data.get(key)
                if isinstance(value, list):
                    return [float(v) for v in value]
            return []

        drift_samples = _read_samples("driftSamples", "drift_samples")
        safe_delta_h_samples = _read_samples(
            "safeDeltaHSamples",
            "safe_delta_h_samples",
            "safeDeltaH",
        )
        attack_entropy_samples = _read_samples(
            "attackEntropySamples",
            "attack_entropy_samples",
            "attackEntropy",
        )

        if not drift_samples:
            raise typer.BadParameter("Calibration missing drift samples.")
        if not safe_delta_h_samples:
            raise typer.BadParameter("Calibration missing safe delta-H samples.")
        if not attack_entropy_samples:
            raise typer.BadParameter("Calibration missing attack entropy samples.")

        service = get_geometry_safety_service(
            drift_samples=drift_samples,
            safe_delta_h_samples=safe_delta_h_samples,
            attack_entropy_samples=attack_entropy_samples,
        )

        result = service.jailbreak_test(
            model_path=model,
            prompts=prompt_list,
            adapter_path=adapter,
        )

        output = {
            "modelPath": result.model_path,
            "adapterPath": result.adapter_path,
            "promptsTested": result.prompts_tested,
            "vulnerabilitiesFound": result.vulnerabilities_found,
            "meanThresholdExceedance": result.mean_threshold_exceedance,
            "processingTime": result.processing_time,
            "calibrationPath": str(calibration_path),
            "vulnerabilityDetails": [
                {
                    "prompt": v.prompt[:100] + "..." if len(v.prompt) > 100 else v.prompt,
                    "vulnerabilityType": v.vulnerability_type,
                    "baselineEntropy": v.baseline_entropy,
                    "attackEntropy": v.attack_entropy,
                    "deltaH": v.delta_h,
                    "thresholdExceedance": v.threshold_exceedance,
                    "attackVector": v.attack_vector,
                }
                for v in result.vulnerability_details
            ],
        }

        if context.output_format == "text":
            lines = [
                "JAILBREAK TEST RESULTS",
                f"Model: {result.model_path}",
            ]
            if result.adapter_path:
                lines.append(f"Adapter: {result.adapter_path}")
            lines.append(f"Calibration: {calibration_path}")
            lines.append(f"Prompts Tested: {result.prompts_tested}")
            lines.append(f"Vulnerabilities Found: {result.vulnerabilities_found}")
            lines.append(f"Mean Threshold Exceedance: {result.mean_threshold_exceedance:.2f}")
            lines.append(f"Processing Time: {result.processing_time:.2f}s")

            if result.vulnerability_details:
                lines.append("")
                lines.append("VULNERABILITIES:")
                for v in result.vulnerability_details[:5]:
                    lines.append(f"  [{v.vulnerability_type}] {v.prompt[:60]}...")
                    lines.append(f"    Delta-H: {v.delta_h:.4f}, Exceedance: {v.threshold_exceedance:.2f}")
                if len(result.vulnerability_details) > 5:
                    lines.append(f"  ... and {len(result.vulnerability_details) - 5} more")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(output, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-4006",
            title="Jailbreak test failed",
            detail=str(e),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


@app.command("probe-redteam")
def safety_probe_redteam(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Scan adapter metadata for threat indicators (static analysis).

    Examples:
        mc safety probe-redteam --name my-adapter
        mc safety probe-redteam --name my-adapter --tag skill1 --tag skill2
    """
    from modelcypher.cli.composition import get_safety_probe_service

    context = get_context(ctx)
    source, _ = EmbeddingDefaults.resolved_source()
    if source == "http":
        warn_network(
            context,
            f"Embedding provider uses HTTP endpoint from {EmbeddingDefaults.EMBEDDING_API_URL_ENV}.",
        )
    service = get_safety_probe_service()

    indicators = service.scan_adapter_metadata(
        name=name,
        description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.threat_indicators_payload(indicators)

    if context.output_format == "text":
        lines = [
            "RED TEAM STATIC ANALYSIS",
            f"Adapter: {name}",
            f"Threat Indicators: {payload['count']}",
            f"Max Mean Distance: {payload['maxMeanDistance']:.4f}",
        ]
        if indicators:
            lines.append("")
            lines.append("DETECTED OUTLIERS:")
            for ind in indicators:
                lines.append(
                    f"  [{ind.mean_distance:.4f}] {ind.field}: {ind.text}"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-behavioral")
def safety_probe_behavioral(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Adapter name"),
    description: str | None = typer.Option(None, "--description", help="Adapter description"),
    tags: list[str] | None = typer.Option(None, "--tag", help="Skill tags (can specify multiple)"),
    creator: str | None = typer.Option(None, "--creator", help="Creator identifier"),
    base_model: str | None = typer.Option(None, "--base-model", help="Base model ID"),
) -> None:
    """Run behavioral probes (requires inference hook for full analysis).

    Examples:
        mc safety probe-behavioral --name my-adapter
    """
    from modelcypher.cli.composition import get_safety_probe_service

    context = get_context(ctx)
    source, _ = EmbeddingDefaults.resolved_source()
    if source == "http":
        warn_network(
            context,
            f"Embedding provider uses HTTP endpoint from {EmbeddingDefaults.EMBEDDING_API_URL_ENV}.",
        )
    service = get_safety_probe_service()

    result = service.run_behavioral_probes(
        adapter_name=name,
        adapter_description=description,
        skill_tags=list(tags) if tags else None,
        creator=creator,
        base_model_id=base_model,
    )

    payload = SafetyProbeService.composite_result_payload(result)

    if context.output_format == "text":
        lines = [
            "BEHAVIORAL PROBE RESULTS",
            f"Adapter: {name}",
            f"Any Findings: {payload['anyFindings']}",
            f"Probes Run: {payload['probeCount']}",
        ]
        if payload["aggregateFindingCounts"]:
            counts_str = ", ".join(
                f"{k}: {v}" for k, v in payload["aggregateFindingCounts"].items()
            )
            lines.append(f"Aggregate Finding Counts: {counts_str}")
        if payload["anyFindings"]:
            lines.append("")
            lines.append("PROBES WITH FINDINGS:")
            for r in result.probe_results:
                if r.has_findings:
                    counts = r.finding_counts or {}
                    counts_str = ", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "none"
                    lines.append(f"  {r.probe_name}: {r.details} (counts: {counts_str})")
        if payload["allFindings"]:
            lines.append("")
            lines.append("FINDINGS:")
            for finding in payload["allFindings"][:10]:
                lines.append(f"  - {finding}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("bilm-probe-info")
def bilm_probe_info(
    ctx: typer.Context,
) -> None:
    """Show information about BiLM probe training.

    BiLM probes use bidirectional language model representations for
    token-level domain classification (e.g., detecting specific content types).

    Examples:
        mc analyze bilm-probe-info
    """
    context = get_context(ctx)

    payload = {
        "description": "Bidirectional LM Probe for token-level domain classification",
        "inputs": {
            "forward_positive": "Forward LM activations for positive samples [n_pos, hidden_dim]",
            "backward_positive": "Backward LM activations for positive samples [n_pos, hidden_dim]",
            "forward_negative": "Forward LM activations for negative samples [n_neg, hidden_dim]",
            "backward_negative": "Backward LM activations for negative samples [n_neg, hidden_dim]",
        },
        "outputs": {
            "train_accuracy": "Training accuracy",
            "train_f1": "Training F1 score",
            "val_accuracy": "Validation accuracy (if val_split > 0)",
            "val_f1": "Validation F1 score",
        },
        "hyperparameters": {
            "val_split": "Fraction for validation (default: 0.1)",
            "learning_rate": "Learning rate (default: 0.01)",
            "max_iterations": "Max training iterations (default: 1000)",
        },
        "usage": "Use BiLMProbeService.train() with collected activations from forward and backward LM passes.",
    }

    if context.output_format == "text":
        lines = [
            "BILM PROBE TRAINING",
            "",
            payload["description"],
            "",
            "REQUIRED INPUTS:",
        ]
        for name, desc in payload["inputs"].items():
            lines.append(f"  {name}: {desc}")
        lines.append("")
        lines.append("OUTPUTS:")
        for name, desc in payload["outputs"].items():
            lines.append(f"  {name}: {desc}")
        lines.append("")
        lines.append("HYPERPARAMETERS:")
        for name, desc in payload["hyperparameters"].items():
            lines.append(f"  {name}: {desc}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
