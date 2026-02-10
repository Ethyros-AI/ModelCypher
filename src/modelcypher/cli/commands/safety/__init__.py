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

"""Safety analysis CLI commands.

Provides commands for safety analysis, behavioral probing, and geometric analysis.

Modules:
    geometric: Dimension profiling, entropy trajectory, expansion ratio, reasoning flow
    behavioral: Adapter probing, behavioral signatures, cognitive reflection tests
    probes: Jailbreak testing, red-team probes, behavioral probes
    benchmark: Benchmarking, LoRA diagnostics, curriculum profiling
    monitoring: Circuit breaker, persona drift, entropy monitoring, CRM

Commands:
    mc analyze concept-volume --model <path> --concepts <file> --layer <n>
    mc safety adapter-probe --adapter <path>
    mc safety behavioral-signature --model <path>
    mc safety dimension-profile --model <path>
    mc safety entropy-trajectory --model <path>
    mc safety expansion-ratio --model <path> --prompt <text>
    mc safety reasoning-flow --model <path> --prompt <text>
    mc safety spectral-trajectory --model <path>
    mc safety jacobian-trace --model <path> --prompt <text>
    mc safety verification-depth-profile --model <path>
    mc safety cognitive-reflection-test --model <path>
    mc safety calibrate-safety --model <path> --prompts <file> --output-file <file>
    mc safety jailbreak-test --model <path> --prompts <file>
    mc safety probe-redteam --name <name>
    mc safety probe-behavioral --name <name>
    mc safety bilm-probe-info
    mc safety benchmark <model> --suite <suite>
    mc safety reasoning-geometry-validation --model LFM2-350M --benchmark gsm8k
    mc safety lora-svd <adapter> --base <model>
    mc safety sparse-region --list-domains
    mc safety knowledge-type <model> --statement <text> --counterfactual <text> --layer <n>
    mc safety curriculum-profile <model> --problems <file>
    mc safety circuit-breaker --job <id>
    mc safety persona --job <id>
    mc safety uncertainty-modes
    mc safety entropy-pattern --samples <file>
    mc safety entropy-baseline-verify --baseline <file> --deltas <file>
    mc safety crm-build <model> --output <path>
    mc safety crm-compare <source> <target>
"""

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
    reasoning_geometry_validation,
    run_benchmark,
    sparse_region_analysis,
)

# Import command functions from each module
from .geometric import (
    concept_volume_analysis,
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

# Create the main app
app = typer.Typer(no_args_is_help=True)

# Register all commands directly on the main app
# Geometric commands
app.command("concept-volume")(concept_volume_analysis)
app.command("dimension-profile")(safety_dimension_profile)
app.command("entropy-trajectory")(safety_entropy_trajectory)
app.command("expansion-ratio")(safety_expansion_ratio)
app.command("reasoning-flow")(safety_reasoning_flow)
app.command("spectral-trajectory")(safety_spectral_trajectory)
app.command("jacobian-trace")(safety_jacobian_trace)
app.command("verification-depth-profile")(safety_verification_depth_profile)

# Behavioral commands
app.command("adapter-probe")(safety_adapter_probe)
app.command("behavioral-signature")(safety_behavioral_signature)
app.command("cognitive-reflection-test")(safety_cognitive_reflection_test)

# Probe commands
app.command("calibrate-safety")(safety_calibrate)
app.command("jailbreak-test")(safety_jailbreak_test)
app.command("probe-redteam")(safety_probe_redteam)
app.command("probe-behavioral")(safety_probe_behavioral)
app.command("bilm-probe-info")(bilm_probe_info)

# Benchmark commands
app.command("benchmark")(run_benchmark)
app.command("reasoning-geometry-validation")(reasoning_geometry_validation)
app.command("lora-svd")(lora_svd_diagnostic)
app.command("sparse-region")(sparse_region_analysis)
app.command("knowledge-type")(knowledge_type_analysis)
app.command("curriculum-profile")(curriculum_profile)

# Monitoring commands
app.command("circuit-breaker")(safety_circuit_breaker)
app.command("persona")(safety_persona)
app.command("uncertainty-modes")(uncertainty_modes)
app.command("entropy-pattern")(entropy_pattern_analysis)
app.command("entropy-baseline-verify")(entropy_baseline_verify)
app.command("crm-build")(crm_build)
app.command("crm-compare")(crm_compare)

__all__ = ["app"]
