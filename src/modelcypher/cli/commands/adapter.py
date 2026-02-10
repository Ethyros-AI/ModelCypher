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

"""Adapter analysis CLI commands.

Commands:
    mc adapter analyze  - Compute geometry metrics for a LoRA adapter
"""

from __future__ import annotations

from typing import Any, Optional

import typer

from modelcypher.cli.commands._lora_baseline_artifact import (
    DEFAULT_BASELINE_ARTIFACT_PATH,
    DEFAULT_FOUR_CONDITION_RESULTS_PATH,
    calibrate_reference_baseline,
)
from modelcypher.cli.composition import get_adapter_analysis_service
from modelcypher.cli.output import write_error, write_output

app = typer.Typer(no_args_is_help=True)


@app.command()
def analyze(
    adapter_path: str = typer.Argument(
        ...,
        help="Path to adapter weights (safetensors file or directory containing adapters.safetensors)",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model", "-b",
        help="Path to base model (auto-detected from adapter_config.json if not specified)",
    ),
    output_format: str = typer.Option(
        "text",
        "--output", "-o",
        help="Output format: text, json",
    ),
    baseline_artifact: Optional[str] = typer.Option(
        None,
        "--baseline-artifact",
        help=(
            "Path to measured baseline artifact JSON. Defaults to "
            "results/real_adapter_analysis/summary.json when available."
        ),
    ),
) -> None:
    """Compute geometry metrics for a LoRA adapter.

    Analyzes the spectral properties of adapter weight deltas to measure:
    - Amplification CV (spectral selectivity): How selectively the adapter
      targets specific directions. Higher = more selective.
    - Weyl utilization: What fraction of the theoretical singular value
      shift bound is used. Lower = more conservative.

    Reports raw measurements and baseline-relative ratios (no categorical
    verdicts or hardcoded interpretation thresholds).

    Example:
        mc adapter analyze /path/to/adapter
        mc adapter analyze /path/to/adapter.safetensors -b /path/to/base_model
        mc adapter analyze /path/to/adapter --baseline-artifact ./results/real_adapter_analysis/summary.json
    """
    try:
        from modelcypher.core.use_cases.adapter_analysis_service import (
            AdapterAnalysisRequest,
        )

        service = get_adapter_analysis_service()
        request = AdapterAnalysisRequest(
            adapter_path=adapter_path,
            base_model=base_model,
            baseline_artifact=baseline_artifact,
        )
        result = service.analyze(request).to_dict()

        if output_format == "json":
            write_output(result, output_format="json", pretty=True)
        else:
            _print_text_output(result)

    except Exception as e:
        write_error({"code": "ADAPTER_ANALYSIS_ERROR", "message": str(e)})
        raise typer.Exit(1)


@app.command("calibrate-baseline")
def calibrate_baseline(
    four_condition_results: str = typer.Option(
        str(DEFAULT_FOUR_CONDITION_RESULTS_PATH),
        "--four-condition-results",
        help=(
            "Path to four-condition raw measurements JSON "
            "(typically results/four_condition/raw_measurements.json)."
        ),
    ),
    output_artifact: str = typer.Option(
        str(DEFAULT_BASELINE_ARTIFACT_PATH),
        "--output-artifact",
        "-o",
        help=(
            "Path to output baseline artifact JSON "
            "(typically results/real_adapter_analysis/summary.json)."
        ),
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Optional provenance label stored under findings.synthetic_random_baseline.source.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text, json",
    ),
) -> None:
    """Calibrate synthetic random baseline from four-condition measurements.

    Promotes measured experiment output into production artifact format used by
    `mc adapter analyze` and safety diagnostics.

    Example:
        mc adapter calibrate-baseline
        mc adapter calibrate-baseline \\
            --four-condition-results ./results/four_condition/raw_measurements.json \\
            --output-artifact ./results/real_adapter_analysis/summary.json
    """
    try:
        payload = calibrate_reference_baseline(
            four_condition_results=four_condition_results,
            output_artifact=output_artifact,
            source_label=source,
        )

        if output_format == "json":
            write_output(payload, output_format="json", pretty=True)
            return

        baseline = payload["reference_baseline"]
        print("\nBaseline Calibration Complete")
        print(f"Output Artifact: {payload['output_artifact']}")
        print(f"Input Measurements: {payload['source_four_condition_results']}")
        print("Measured Synthetic Random Baseline:")
        print(f"  amplification_cv: {baseline['amplification_cv']:.6f}")
        print(f"  weyl_utilization: {baseline['weyl_utilization']:.6f}")
        if "sample_count" in baseline:
            print(f"  sample_count: {baseline['sample_count']}")
        if "amplification_cv_range" in baseline:
            cv_range = baseline["amplification_cv_range"]
            print(f"  amplification_cv_range: [{cv_range[0]:.6f}, {cv_range[1]:.6f}]")
        if "weyl_utilization_range" in baseline:
            weyl_range = baseline["weyl_utilization_range"]
            print(
                "  weyl_utilization_range: "
                f"[{weyl_range[0]:.6f}, {weyl_range[1]:.6f}]"
            )
        print(f"  source: {baseline['source']}")
    except Exception as e:
        write_error({"code": "ADAPTER_BASELINE_CALIBRATION_ERROR", "message": str(e)})
        raise typer.Exit(1)


def _print_text_output(result: dict[str, Any]) -> None:
    """Print human-readable output."""
    print(f"\n{'='*60}")
    print(f"Adapter: {result['adapter_name']}")
    print(f"Base Model: {result['base_model']}")
    print(f"Rank: {result['lora_rank']}, Scale: {result['lora_scale']}")
    print(f"{'='*60}")

    metrics = result["metrics"]

    print(f"\nGeometry Metrics ({result['n_layers']} layers):")
    print(
        f"  Amplification CV:    {metrics['amplification_cv']['mean']:.4f} "
        f"(range: {metrics['amplification_cv']['min']:.4f} - "
        f"{metrics['amplification_cv']['max']:.4f})"
    )
    print(
        f"  Weyl Utilization:    {metrics['weyl_utilization']['mean']:.4f} "
        f"(range: {metrics['weyl_utilization']['min']:.4f} - "
        f"{metrics['weyl_utilization']['max']:.4f})"
    )
    print(f"  Total Frobenius:     {metrics['delta_frobenius_norm']['total']:.4f}")
    print(f"  Max Spectral Norm:   {metrics['delta_spectral_norm']['max']:.4f}")

    comparison = result.get("reference_comparison")
    baseline = result.get("reference_baseline")
    if comparison is not None:
        print("\nBaseline Ratios:")
        print(
            "  Amplification CV vs random: "
            f"{comparison['amplification_cv_vs_random_baseline']:.4f}x"
        )
        print(
            "  Weyl utilization vs random: "
            f"{comparison['weyl_utilization_vs_random_baseline']:.4f}x"
        )

    if baseline is not None:
        print("\nReference Baseline (Measured Artifact):")
        print(f"  Type:    {baseline['type']}")
        print(
            "  Random:  "
            f"CV={baseline['amplification_cv']:.6f}, "
            f"Weyl={baseline['weyl_utilization']:.6f}"
        )
        print(f"  Source:  {baseline['source']}")
        if "experiment_date" in baseline:
            print(f"  Date:    {baseline['experiment_date']}")
        print(f"  File:    {baseline['artifact_path']}")
