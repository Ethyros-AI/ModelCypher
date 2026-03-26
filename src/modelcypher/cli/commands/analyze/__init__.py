from __future__ import annotations

import typer

from .behavioral import (
    safety_adapter_probe,
    safety_behavioral_signature,
    safety_cognitive_reflection_test,
)
from .benchmark import (
    curriculum_profile,
    knowledge_type_analysis,
    lora_svd_diagnostic,
    run_benchmark,
    sparse_region_analysis,
)
from .geodesic_compare import geodesic_compare
from .geodesic_profile import geodesic_profile
from .geodesic_trajectory import geodesic_trajectory
from .geometric import (
    concept_volume_analysis,
    safety_attention_collapse,
    safety_attention_sink,
    safety_chain_profile,
    safety_dimension_profile,
    safety_entropy_trajectory,
    safety_expansion_ratio,
    safety_jacobian_trace,
    safety_reasoning_flow,
    safety_spectral_trajectory,
    safety_verification_depth_profile,
)
from .monitoring import (
    crm_build,
    crm_compare,
    entropy_baseline_verify,
    entropy_pattern_analysis,
    safety_circuit_breaker,
    safety_persona,
    uncertainty_modes,
)
from .probes import (
    bilm_probe_info,
    safety_calibrate,
    safety_jailbreak_test,
    safety_probe_behavioral,
    safety_probe_redteam,
)

from .workflows import (
    analyze_capture,
    analyze_compare,
    analyze_family,
    analyze_report,
)

WORKFLOW_PANEL = "Canonical Workflows"
EXPERT_PANEL = "Expert Instruments"

app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Workflow-first model observation and analysis. "
        "Canonical workflows: capture, family, compare, report, and probe. "
        "Expert metric commands remain available for direct inspection."
    ),
)
probe_app = typer.Typer(
    no_args_is_help=True,
    help="Targeted probe, red-team, and boundary-testing analysis workflows.",
)
app.add_typer(probe_app, name="probe", rich_help_panel=WORKFLOW_PANEL)

# Canonical workflow commands
app.command("capture", rich_help_panel=WORKFLOW_PANEL)(analyze_capture)
app.command("family", rich_help_panel=WORKFLOW_PANEL)(analyze_family)
app.command("compare", rich_help_panel=WORKFLOW_PANEL)(analyze_compare)
app.command("report", rich_help_panel=WORKFLOW_PANEL)(analyze_report)

# Probe workflow canonical subcommands
probe_app.command("calibrate")(safety_calibrate)
probe_app.command("jailbreak")(safety_jailbreak_test)
probe_app.command("redteam")(safety_probe_redteam)
probe_app.command("behavioral")(safety_probe_behavioral)
probe_app.command("bilm-info")(bilm_probe_info)

# Direct expert and compatibility surfaces
app.command("attention-collapse", rich_help_panel=EXPERT_PANEL)(safety_attention_collapse)
app.command("attention-sink", rich_help_panel=EXPERT_PANEL)(safety_attention_sink)
app.command("geodesic-compare", rich_help_panel=EXPERT_PANEL)(geodesic_compare)
app.command("geodesic-profile", rich_help_panel=EXPERT_PANEL)(geodesic_profile)
app.command("geodesic-trajectory", rich_help_panel=EXPERT_PANEL)(geodesic_trajectory)
app.command("concept-volume", rich_help_panel=EXPERT_PANEL)(concept_volume_analysis)
app.command("dimension-profile", rich_help_panel=EXPERT_PANEL)(safety_dimension_profile)
app.command("entropy-trajectory", rich_help_panel=EXPERT_PANEL)(safety_entropy_trajectory)
app.command("expansion-ratio", rich_help_panel=EXPERT_PANEL)(safety_expansion_ratio)
app.command("reasoning-flow", rich_help_panel=EXPERT_PANEL)(safety_reasoning_flow)
app.command("spectral-trajectory", rich_help_panel=EXPERT_PANEL)(safety_spectral_trajectory)
app.command("jacobian-trace", rich_help_panel=EXPERT_PANEL)(safety_jacobian_trace)
app.command("verification-depth-profile", rich_help_panel=EXPERT_PANEL)(
    safety_verification_depth_profile
)
app.command("chain-profile", rich_help_panel=EXPERT_PANEL)(safety_chain_profile)
app.command("adapter-probe", rich_help_panel=EXPERT_PANEL)(safety_adapter_probe)
app.command("behavioral-signature", rich_help_panel=EXPERT_PANEL)(
    safety_behavioral_signature
)
app.command("cognitive-reflection-test", rich_help_panel=EXPERT_PANEL)(
    safety_cognitive_reflection_test
)
app.command("calibrate-safety", rich_help_panel=EXPERT_PANEL)(safety_calibrate)
app.command("jailbreak-test", rich_help_panel=EXPERT_PANEL)(safety_jailbreak_test)
app.command("probe-redteam", rich_help_panel=EXPERT_PANEL)(safety_probe_redteam)
app.command("probe-behavioral", rich_help_panel=EXPERT_PANEL)(safety_probe_behavioral)
app.command("bilm-probe-info", rich_help_panel=EXPERT_PANEL)(bilm_probe_info)
app.command("benchmark", rich_help_panel=EXPERT_PANEL)(run_benchmark)
app.command("lora-svd", rich_help_panel=EXPERT_PANEL)(lora_svd_diagnostic)
app.command("sparse-region", rich_help_panel=EXPERT_PANEL)(sparse_region_analysis)
app.command("knowledge-type", rich_help_panel=EXPERT_PANEL)(knowledge_type_analysis)
app.command("curriculum-profile", rich_help_panel=EXPERT_PANEL)(curriculum_profile)
app.command("circuit-breaker", rich_help_panel=EXPERT_PANEL)(safety_circuit_breaker)
app.command("persona", rich_help_panel=EXPERT_PANEL)(safety_persona)
app.command("uncertainty-modes", rich_help_panel=EXPERT_PANEL)(uncertainty_modes)
app.command("entropy-pattern", rich_help_panel=EXPERT_PANEL)(entropy_pattern_analysis)
app.command("entropy-baseline-verify", rich_help_panel=EXPERT_PANEL)(
    entropy_baseline_verify
)
app.command("crm-build", rich_help_panel=EXPERT_PANEL)(crm_build)
app.command("crm-compare", rich_help_panel=EXPERT_PANEL)(crm_compare)

__all__ = ["app", "probe_app"]
