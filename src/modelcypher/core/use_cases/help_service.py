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
            related_commands = ["mc train start", "mc train status", "mc train pause"]
            examples = [
                "mc train start --model qwen-0.5b --dataset data.jsonl --epochs 3",
                "mc train status <job-id>",
            ]
            answer = (
                "Training in ModelCypher uses the `mc train` command group. "
                "Start training with `mc train start`, monitor with `mc train status`, "
                "and control with `mc train pause/resume/cancel`."
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
        elif "dataset" in question_lower:
            related_commands = [
                "mc dataset validate",
                "mc dataset preview",
                "mc dataset convert",
            ]
            examples = [
                "mc dataset validate data.jsonl --output json",
                "mc dataset preview data.jsonl --lines 5",
            ]
            answer = (
                "Dataset commands help prepare training data. "
                "Validate with `mc dataset validate`, preview with `mc dataset preview`, "
                "and convert formats with `mc dataset convert`."
            )
        else:
            related_commands = ["mc inventory", "mc --help"]
            examples = ["mc inventory --output json", "mc train --help"]
            answer = (
                "ModelCypher is a CLI for on-device ML training. "
                "Start with `mc inventory` to see available resources, "
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
            command: Command name (e.g., "train start", "model list")

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
        return {
            "train_start": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "jobId": {"type": "string"},
                    "batchSize": {"type": "integer"},
                },
                "required": ["jobId"],
            },
            "trainstart": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "jobId": {"type": "string"},
                    "batchSize": {"type": "integer"},
                },
                "required": ["jobId"],
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
                    "datasets": {"type": "array"},
                    "checkpoints": {"type": "array"},
                    "jobs": {"type": "array"},
                    "workspace": {"type": "object"},
                    "mlxVersion": {"type": ["string", "null"]},
                },
                "required": ["models", "datasets", "jobs"],
            },
        }

    def _bash_completions(self) -> str:
        """Generate bash completion script."""
        return """# ModelCypher bash completions
_mc_completions() {
    local cur prev commands
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    commands="train job checkpoint model system dataset eval compare doc geometry adapter thermo calibration rag stability agent-eval dashboard help inventory explain"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        return 0
    fi

    case "${prev}" in
        train)
            COMPREPLY=( $(compgen -W "start status pause resume cancel export logs preflight" -- ${cur}) )
            ;;
        model)
            COMPREPLY=( $(compgen -W "list register merge delete fetch search probe validate-merge analyze-alignment" -- ${cur}) )
            ;;
        geometry)
            COMPREPLY=( $(compgen -W "validate path training safety adapter primes stitch" -- ${cur}) )
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
        'train:Training lifecycle commands'
        'job:Job management'
        'checkpoint:Checkpoint management'
        'model:Model management'
        'system:System information'
        'dataset:Dataset management'
        'eval:Evaluation results'
        'compare:Comparison history'
        'geometry:Geometry commands'
        'adapter:Adapter commands'
        'thermo:Thermodynamic analysis'
        'calibration:Calibration commands'
        'rag:RAG commands'
        'stability:Stability testing'
        'agent-eval:Agent evaluation'
        'dashboard:Dashboard metrics'
        'help:Help commands'
        'inventory:System inventory'
    )

    _describe 'command' commands
}

_mc "$@"
"""

    def _fish_completions(self) -> str:
        """Generate fish completion script."""
        return """# ModelCypher fish completions

complete -c mc -n "__fish_use_subcommand" -a train -d "Training lifecycle"
complete -c mc -n "__fish_use_subcommand" -a job -d "Job management"
complete -c mc -n "__fish_use_subcommand" -a checkpoint -d "Checkpoint management"
complete -c mc -n "__fish_use_subcommand" -a model -d "Model management"
complete -c mc -n "__fish_use_subcommand" -a system -d "System information"
complete -c mc -n "__fish_use_subcommand" -a dataset -d "Dataset management"
complete -c mc -n "__fish_use_subcommand" -a eval -d "Evaluation results"
complete -c mc -n "__fish_use_subcommand" -a compare -d "Comparison history"
complete -c mc -n "__fish_use_subcommand" -a geometry -d "Geometry commands"
complete -c mc -n "__fish_use_subcommand" -a adapter -d "Adapter commands"
complete -c mc -n "__fish_use_subcommand" -a thermo -d "Thermodynamic analysis"
complete -c mc -n "__fish_use_subcommand" -a calibration -d "Calibration commands"
complete -c mc -n "__fish_use_subcommand" -a rag -d "RAG commands"
complete -c mc -n "__fish_use_subcommand" -a stability -d "Stability testing"
complete -c mc -n "__fish_use_subcommand" -a agent-eval -d "Agent evaluation"
complete -c mc -n "__fish_use_subcommand" -a dashboard -d "Dashboard metrics"
complete -c mc -n "__fish_use_subcommand" -a help -d "Help commands"
complete -c mc -n "__fish_use_subcommand" -a inventory -d "System inventory"

# train subcommands
complete -c mc -n "__fish_seen_subcommand_from train" -a start -d "Start training"
complete -c mc -n "__fish_seen_subcommand_from train" -a status -d "Job status"
complete -c mc -n "__fish_seen_subcommand_from train" -a pause -d "Pause job"
complete -c mc -n "__fish_seen_subcommand_from train" -a resume -d "Resume job"
complete -c mc -n "__fish_seen_subcommand_from train" -a cancel -d "Cancel job"
"""
