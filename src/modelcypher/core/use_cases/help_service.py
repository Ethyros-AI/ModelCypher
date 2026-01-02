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

"""Help service for contextual help, completions, and schema.

Provides help ask, shell completions, and JSON schema functionality
for CLI discoverability and documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HelpResponse:
    """Response to a help question."""

    question: str
    answer: str
    related_commands: list[str]
    examples: list[str]
    docs_url: str | None


class HelpService:
    """Service for help, completions, and schema.

    Provides contextual help for CLI commands, shell completion scripts,
    and JSON schemas for command outputs.
    """

    def __init__(self) -> None:
        """Initialize help service."""
        self._command_help: dict[str, dict[str, Any]] = self._build_command_help()

    def ask(self, question: str) -> HelpResponse:
        """Return contextual help for a question.

        Args:
            question: User's question about ModelCypher

        Returns:
            HelpResponse with answer and related information
        """
        question_lower = question.lower()

        # Match question to relevant commands
        related_commands = []
        examples = []

        if "train" in question_lower:
            related_commands = ["mc train start", "mc train preflight", "mc train status"]
            examples = [
                "mc train start --model qwen-0.5b --dataset data.jsonl --epochs 3",
                "mc train preflight --model qwen-0.5b --dataset data.jsonl",
            ]
            answer = (
                "Training uses the `mc train` command group with geometry-aware monitoring. "
                "Start with `mc train start`, preflight with `mc train preflight`, "
                "and track progress via `mc train status`."
            )
        elif "merge" in question_lower:
            related_commands = ["mc merge pipeline", "mc merge validate", "mc merge diagnose"]
            examples = [
                "mc merge pipeline --source ./model-a --target ./model-b --output-dir ./merged --transplant-domains math",
                "mc merge validate --merged ./merged",
            ]
            answer = (
                "Merge workflows use geometry-first validation. "
                "Run `mc merge pipeline` to combine models, then `mc merge validate` or "
                "`mc merge diagnose` to inspect alignment."
            )
        elif "model" in question_lower:
            related_commands = ["mc model list", "mc model fetch", "mc model probe"]
            examples = [
                "mc model list --output json",
                "mc model fetch Qwen/Qwen2.5-0.5B-Instruct --auto-register --alias qwen",
            ]
            answer = (
                "Model management uses the `mc model` command group. "
                "List models with `mc model list`, download with `mc model fetch`, "
                "and inspect with `mc model probe`."
            )
        elif "geometry" in question_lower:
            related_commands = [
                "mc geometry validate",
                "mc geometry training status",
                "mc geometry primes list",
            ]
            examples = [
                "mc geometry validate --output json",
                "mc geometry training status --job <job-id>",
            ]
            answer = (
                "Geometry commands analyze training dynamics and model alignment. "
                "Use `mc geometry validate` for math validation, "
                "`mc geometry training status` for live metrics."
            )
        else:
            related_commands = ["mc inventory", "mc geometry validate", "mc train --help"]
            examples = ["mc inventory --output json", "mc geometry validate --output json"]
            answer = (
                "ModelCypher focuses on geometry-first diagnostics and merge validation. "
                "Start with `mc inventory` to see resources, "
                "or use `mc --help` for command overview."
            )

        return HelpResponse(
            question=question,
            answer=answer,
            related_commands=related_commands,
            examples=examples,
            docs_url="https://github.com/modelcypher/modelcypher/docs",
        )

    def explain(self, command: str) -> dict[str, Any]:
        """Explain a command's side effects and requirements.

        Args:
            command: Command name to explain

        Returns:
            Dictionary with command metadata (service calls, affected resources, etc.)
        """
        command_lower = command.lower().strip()

        # Default fallback
        payload = {
            "command": command,
            "description": "General ModelCypher command",
            "serviceCalls": [],
            "affectedResources": [],
            "requiredPermissions": [],
            "warnings": [],
            "estimatedDuration": None,
        }

        if "train start" in command_lower:
            payload.update(
                {
                    "description": "Initialize and execute a LoRA fine-tuning job",
                    "serviceCalls": ["TrainingService.start", "LocalTrainingEngine.start"],
                    "affectedResources": ["VRAM", "Disk (checkpoints)", "CPU"],
                    "requiredPermissions": ["Filesystem Write", "GPU Access"],
                    "warnings": ["High power consumption", "Thermal throttling possible"],
                    "estimatedDuration": "Minutes to Hours",
                }
            )
        elif "model fetch" in command_lower:
            payload.update(
                {
                    "description": "Download a model from remote repository",
                    "serviceCalls": ["ModelService.fetch", "HuggingFaceHub"],
                    "affectedResources": ["Bandwidth", "Disk Space"],
                    "requiredPermissions": ["Network Access", "Filesystem Write"],
                    "warnings": ["Large download size"],
                    "estimatedDuration": "Seconds to Minutes",
                }
            )
        elif "inventory" in command_lower:
            payload.update(
                {
                    "description": "Retrieve comprehensive system and resource inventory",
                    "serviceCalls": ["InventoryService.inventory", "SystemService.status"],
                    "affectedResources": [],
                    "requiredPermissions": ["Read Only"],
                    "warnings": [],
                    "estimatedDuration": "Fast",
                }
            )
        elif "geometry validate" in command_lower:
            payload.update(
                {
                    "description": "Validate mathematical invariants and geometric projections",
                    "serviceCalls": ["GeometryService.validate"],
                    "affectedResources": ["CPU", "Memory"],
                    "requiredPermissions": ["Read Only"],
                    "warnings": ["Computationally intensive"],
                    "estimatedDuration": "Seconds",
                }
            )

        return payload

    def completions(self, shell: str) -> str:
        """Generate shell completion script.

        Args:
            shell: Shell type (bash, zsh, fish)

        Returns:
            Shell completion script as string

        Raises:
            ValueError: If shell type is not supported
        """
        supported_shells = {"bash", "zsh", "fish"}
        shell_lower = shell.lower()

        if shell_lower not in supported_shells:
            raise ValueError(f"Unsupported shell: {shell}. Supported: {supported_shells}")

        if shell_lower == "bash":
            return self._bash_completions()
        elif shell_lower == "zsh":
            return self._zsh_completions()
        else:
            return self._fish_completions()

    def schema(self, command: str) -> dict[str, Any]:
        """Return JSON schema for command output.

        Args:
            command: Command name (e.g., "geometry validate", "model list")

        Returns:
            JSON Schema document for the command's output

        Raises:
            ValueError: If command is not found
        """
        # Normalize command name
        command_key = command.lower().replace(" ", "_").replace("-", "_")

        schemas = self._get_schemas()
        if command_key not in schemas:
            # Try without underscores
            command_key = command.lower().replace(" ", "").replace("-", "")

        if command_key not in schemas:
            raise ValueError(f"Schema not found for command: {command}")

        return schemas[command_key]

    def _build_command_help(self) -> dict[str, dict[str, Any]]:
        """Build command help database."""
        return {
            "train_start": {
                "description": "Start a new training job",
                "usage": "mc train start --model <id> --dataset <path>",
                "required": ["--model", "--dataset"],
            },
            "geometry_validate": {
                "description": "Validate geometric invariants and diagnostics",
                "usage": "mc geometry validate --output json",
                "required": [],
            },
            "geometry_primes_probe_model": {
                "description": "Probe a model for semantic prime activations",
                "usage": "mc geometry primes probe-model <model_path>",
                "required": ["<model_path>"],
            },
            "geometry_primes_compare": {
                "description": "Compare semantic prime activations between two models",
                "usage": "mc geometry primes compare <activations_a.json> <activations_b.json>",
                "required": ["<activations_a.json>", "<activations_b.json>"],
            },
            "model_list": {
                "description": "List registered models",
                "usage": "mc model list --output json",
                "required": [],
            },
            "inventory": {
                "description": "Get complete system state",
                "usage": "mc inventory --output json",
                "required": [],
            },
        }

    def _get_schemas(self) -> dict[str, dict[str, Any]]:
        """Get JSON schemas for command outputs."""
        schemas = {
            "train_start": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "jobId": {"type": "string"},
                    "batchSize": {"type": "integer"},
                },
                "required": ["jobId", "batchSize"],
            },
            "trainstart": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "jobId": {"type": "string"},
                    "batchSize": {"type": "integer"},
                },
                "required": ["jobId", "batchSize"],
            },
            "geometry_validate": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "gromovWasserstein": {"type": ["number", "null"]},
                    "traversalCoherence": {"type": ["number", "null"]},
                    "pathSignature": {"type": ["array", "null"]},
                },
                "required": [],
            },
            "geometryvalidate": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "gromovWasserstein": {"type": ["number", "null"]},
                    "traversalCoherence": {"type": ["number", "null"]},
                    "pathSignature": {"type": ["array", "null"]},
                },
                "required": [],
            },
            "geometry_primes_probe_model": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "_schema": {"type": "string"},
                    "model_path": {"type": "string"},
                    "layer": {"type": "integer"},
                    "primes_probed": {"type": "integer"},
                    "total_primes": {"type": "integer"},
                    "overall_coherence": {"type": "number"},
                    "overall_coherence_raw": {"type": "number"},
                    "category_coherence": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": [],
            },
            "geometryprimesprobemodel": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "_schema": {"type": "string"},
                    "model_path": {"type": "string"},
                    "layer": {"type": "integer"},
                    "primes_probed": {"type": "integer"},
                    "total_primes": {"type": "integer"},
                    "overall_coherence": {"type": "number"},
                    "overall_coherence_raw": {"type": "number"},
                    "category_coherence": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": [],
            },
            "geometry_primes_compare": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "_schema": {"type": "string"},
                    "model_a": {"type": "string"},
                    "model_b": {"type": "string"},
                    "common_primes": {"type": "integer"},
                    "cka_similarity": {"type": "number"},
                    "cka_raw": {"type": "number"},
                    "most_similar_primes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "most_divergent_primes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
            "geometryprimescompare": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "_schema": {"type": "string"},
                    "model_a": {"type": "string"},
                    "model_b": {"type": "string"},
                    "common_primes": {"type": "integer"},
                    "cka_similarity": {"type": "number"},
                    "cka_raw": {"type": "number"},
                    "most_similar_primes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "most_divergent_primes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [],
            },
            "model_list": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "alias": {"type": "string"},
                        "architecture": {"type": "string"},
                        "format": {"type": "string"},
                        "path": {"type": "string"},
                        "sizeBytes": {"type": "integer"},
                    },
                    "required": ["id", "path"],
                },
            },
            "modellist": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "alias": {"type": "string"},
                        "architecture": {"type": "string"},
                        "format": {"type": "string"},
                        "path": {"type": "string"},
                        "sizeBytes": {"type": "integer"},
                    },
                    "required": ["id", "path"],
                },
            },
            "inventory": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "models": {"type": "array"},
                    "checkpoints": {"type": "array"},
                    "jobs": {"type": "array"},
                    "workspace": {"type": "object"},
                    "mlxVersion": {"type": ["string", "null"]},
                },
                "required": ["models", "checkpoints", "jobs"],
            },
        }

        geometry_schema_ids: dict[str, str | None] = {
            "geometry_atlas_dimensionality": "mc.geometry.atlas.dimensionality.v1",
            "geometry_atlas_dimensionality_study": "mc.geometry.atlas.dimensionality_study.v1",
            "geometry_baseline_compare": "mc.geometry.baseline.compare.v1",
            "geometry_baseline_extract": "mc.geometry.baseline.extract.v1",
            "geometry_baseline_list": "mc.geometry.baseline.list.v1",
            "geometry_baseline_validate": "mc.geometry.baseline.validate.v1",
            "geometry_concept_compare": "mc.geometry.concept.compare.v1",
            "geometry_concept_detect": "mc.geometry.concept.detect.v1",
            "geometry_crm_build": None,
            "geometry_crm_compare": None,
            "geometry_crm_delta_mask": "mc.geometry.crm.delta_mask.v1",
            "geometry_crm_sequence_inventory": None,
            "geometry_cross_cultural_analyze": "mc.geometry.cross_cultural.analyze.v1",
            "geometry_emotion_analyze": None,
            "geometry_emotion_inventory": None,
            "geometry_emotion_opposition": None,
            "geometry_geom_adapter_decomposition": None,
            "geometry_geom_adapter_sparsity": None,
            "geometry_interference_null_space": "mc.geometry.interference.null_space.v1",
            "geometry_interference_predict": "mc.geometry.merge_analysis.v1",
            "geometry_interference_safety_polytope": "mc.geometry.interference.safety_polytope.v1",
            "geometry_interference_volume": "mc.geometry.interference.volume.v1",
            "geometry_invariant_atlas_inventory": "mc.geometry.atlas.inventory.v1",
            "geometry_invariant_collapse_risk": None,
            "geometry_invariant_map_layers": None,
            "geometry_manifold_cluster": None,
            "geometry_manifold_dimension": None,
            "geometry_manifold_query": None,
            "geometry_merge_entropy_guide": "mc.merge.entropy.guide.v1",
            "geometry_merge_entropy_profile": "mc.merge.entropy.profile.v1",
            "geometry_merge_entropy_validate": "mc.merge.entropy.validate.v1",
            "geometry_metrics_gromov_wasserstein": None,
            "geometry_metrics_intrinsic_dimension": None,
            "geometry_metrics_topological_fingerprint": None,
            "geometry_moral_analyze": "mc.geometry.moral.analyze.v1",
            "geometry_moral_anchors": "mc.geometry.moral.anchors.v1",
            "geometry_moral_probe_model": "mc.geometry.moral.probe_model.v1",
            "geometry_path_compare": None,
            "geometry_path_detect": None,
            "geometry_persona_drift": None,
            "geometry_persona_extract": None,
            "geometry_persona_traits": None,
            "geometry_primes_compare": "mc.geometry.primes.compare.v1",
            "geometry_primes_list": "mc.geometry.primes.list.v1",
            "geometry_primes_probe_model": "mc.geometry.primes.probe.v1",
            "geometry_refinement_analyze": None,
            "geometry_refinement_summary": None,
            "geometry_refusal_detect": None,
            "geometry_refusal_pairs": None,
            "geometry_research_build_eval_dataset": "mc.geometry.research.build_eval_dataset.v1",
            "geometry_research_concept_density": "mc.geometry.research.concept_density.v1",
            "geometry_research_graft_boundary": "mc.geometry.research.graft_boundary.v1",
            "geometry_research_knowledge_diff": "mc.geometry.research.knowledge_diff.v1",
            "geometry_research_sparse_regions": "mc.geometry.research.sparse_regions.v1",
            "geometry_research_zero_shot_transfer": "mc.geometry.research.zero_shot_transfer.v1",
            "geometry_safety_circuit_breaker": None,
            "geometry_safety_jailbreak_test": None,
            "geometry_safety_persona": None,
            "geometry_safety_probe_behavioral": None,
            "geometry_safety_probe_redteam": None,
            "geometry_social_analyze": "mc.geometry.social.analyze.v1",
            "geometry_social_anchors": "mc.geometry.social.anchors.v1",
            "geometry_social_probe_model": "mc.geometry.social.probe_model.v1",
            "geometry_sparse_domains": None,
            "geometry_sparse_locate": None,
            "geometry_sparse_neurons": None,
            "geometry_spatial_analyze": "mc.geometry.spatial.full_analysis.v1",
            "geometry_spatial_anchors": "mc.geometry.spatial.anchors.v1",
            "geometry_spatial_cross_grounding_feasibility": "mc.geometry.spatial.cross_grounding_feasibility.v1",
            "geometry_spatial_cross_grounding_transfer": "mc.geometry.spatial.cross_grounding_transfer.v1",
            "geometry_spatial_density": "mc.geometry.spatial.density.v1",
            "geometry_spatial_euclidean": "mc.geometry.spatial.euclidean.v1",
            "geometry_spatial_gravity": "mc.geometry.spatial.gravity.v1",
            "geometry_spatial_probe_model": "mc.geometry.spatial.probe_model.v1",
            "geometry_temporal_analyze": "mc.geometry.temporal.analyze.v1",
            "geometry_temporal_anchors": "mc.geometry.temporal.anchors.v1",
            "geometry_temporal_probe_model": "mc.geometry.temporal.probe_model.v1",
            "geometry_training_history": None,
            "geometry_training_levels": None,
            "geometry_training_status": None,
            "geometry_transfer_compare": None,
            "geometry_transfer_profile": None,
            "geometry_transfer_project": None,
            "geometry_waypoint_alpha_profile": "mc.geometry.waypoint.alpha_profile.v1",
            "geometry_waypoint_audit": "mc.geometry.waypoint.audit.v1",
            "geometry_waypoint_profile": "mc.geometry.waypoint.profile.v1",
            "geometry_waypoint_validate": "mc.geometry.waypoint.validate.v1",
        }

        for command_key, schema_id in geometry_schema_ids.items():
            if command_key in schemas:
                continue
            schema: dict[str, Any] = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "_schema": {"type": "string"},
                },
                "required": [],
            }
            if schema_id:
                schema["properties"]["_schema"] = {"const": schema_id}
            schemas[command_key] = schema

        return schemas

    def _bash_completions(self) -> str:
        """Generate bash completion script."""
        return """# ModelCypher bash completions
_mc_completions() {
    local cur prev commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    commands="model merge system geometry adapter entropy thermo safety agent stability dashboard storage infer agent-eval research help inventory explain train job checkpoint eval compare validate estimate calibration"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        model)
            COMPREPLY=( $(compgen -W "list register merge delete fetch search probe validate-merge analyze-alignment" -- ${cur}) )
            ;;
        geometry)
            COMPREPLY=( $(compgen -W "validate path training safety adapter primes stitch" -- ${cur}) )
            ;;
        train)
            COMPREPLY=( $(compgen -W "start preflight status pause resume cancel export logs" -- ${cur}) )
            ;;
        job)
            COMPREPLY=( $(compgen -W "list show attach delete" -- ${cur}) )
            ;;
        checkpoint)
            COMPREPLY=( $(compgen -W "list export delete" -- ${cur}) )
            ;;
        eval)
            COMPREPLY=( $(compgen -W "list show run" -- ${cur}) )
            ;;
        compare)
            COMPREPLY=( $(compgen -W "list show run checkpoints baseline score" -- ${cur}) )
            ;;
        validate)
            COMPREPLY=( $(compgen -W "train" -- ${cur}) )
            ;;
        estimate)
            COMPREPLY=( $(compgen -W "train" -- ${cur}) )
            ;;
        calibration)
            COMPREPLY=( $(compgen -W "run status apply" -- ${cur}) )
            ;;
        *)
            ;;
    esac
}
complete -F _mc_completions mc
"""

    def _zsh_completions(self) -> str:
        """Generate zsh completion script."""
        return """#compdef mc
# ModelCypher zsh completions

_mc() {
    local -a commands
    commands=(
        'model:Model management'
        'system:System information'
        'geometry:Geometry commands'
        'adapter:Adapter commands'
        'entropy:Entropy analysis'
        'thermo:Thermodynamic analysis'
        'safety:Safety analysis'
        'agent:Agent tools'
        'stability:Stability testing'
        'dashboard:Dashboard metrics'
        'storage:Storage management'
        'infer:Inference tools'
        'agent-eval:Agent evaluation'
        'research:Research tools'
        'help:Help commands'
        'inventory:System inventory'
        'explain:Command explanations'
        'train:Training jobs'
        'job:Training job management'
        'checkpoint:Checkpoint management'
        'eval:Evaluation runs'
        'compare:Evaluation comparisons'
        'validate:Validation utilities'
        'estimate:Estimation utilities'
        'calibration:Calibration runs'
    )

    _describe 'command' commands
}

_mc "$@"
"""

    def _fish_completions(self) -> str:
        """Generate fish completion script."""
        return """# ModelCypher fish completions

complete -c mc -n "__fish_use_subcommand" -a model -d "Model management"
complete -c mc -n "__fish_use_subcommand" -a system -d "System information"
complete -c mc -n "__fish_use_subcommand" -a geometry -d "Geometry commands"
complete -c mc -n "__fish_use_subcommand" -a adapter -d "Adapter commands"
complete -c mc -n "__fish_use_subcommand" -a entropy -d "Entropy analysis"
complete -c mc -n "__fish_use_subcommand" -a thermo -d "Thermodynamic analysis"
complete -c mc -n "__fish_use_subcommand" -a safety -d "Safety analysis"
complete -c mc -n "__fish_use_subcommand" -a agent -d "Agent tools"
complete -c mc -n "__fish_use_subcommand" -a stability -d "Stability testing"
complete -c mc -n "__fish_use_subcommand" -a agent-eval -d "Agent evaluation"
complete -c mc -n "__fish_use_subcommand" -a dashboard -d "Dashboard metrics"
complete -c mc -n "__fish_use_subcommand" -a storage -d "Storage management"
complete -c mc -n "__fish_use_subcommand" -a infer -d "Inference tools"
complete -c mc -n "__fish_use_subcommand" -a research -d "Research tools"
complete -c mc -n "__fish_use_subcommand" -a help -d "Help commands"
complete -c mc -n "__fish_use_subcommand" -a inventory -d "System inventory"
complete -c mc -n "__fish_use_subcommand" -a explain -d "Command explanations"
complete -c mc -n "__fish_use_subcommand" -a train -d "Training jobs"
complete -c mc -n "__fish_use_subcommand" -a job -d "Training job management"
complete -c mc -n "__fish_use_subcommand" -a checkpoint -d "Checkpoint management"
complete -c mc -n "__fish_use_subcommand" -a eval -d "Evaluation runs"
complete -c mc -n "__fish_use_subcommand" -a compare -d "Evaluation comparisons"
complete -c mc -n "__fish_use_subcommand" -a validate -d "Validation utilities"
complete -c mc -n "__fish_use_subcommand" -a estimate -d "Estimation utilities"
complete -c mc -n "__fish_use_subcommand" -a calibration -d "Calibration runs"
"""
