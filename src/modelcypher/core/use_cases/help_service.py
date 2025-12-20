"""Help service for contextual help, completions, and schema.

Provides help ask, shell completions, and JSON schema functionality
for CLI discoverability and documentation.
"""

from __future__ import annotations

import json
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
            related_commands = ["tc train start", "tc train status", "tc train pause"]
            examples = [
                "tc train start --model qwen-0.5b --dataset data.jsonl --epochs 3",
                "tc train status <job-id>",
            ]
            answer = (
                "Training in ModelCypher uses the `tc train` command group. "
                "Start training with `tc train start`, monitor with `tc train status`, "
                "and control with `tc train pause/resume/cancel`."
            )
        elif "model" in question_lower:
            related_commands = ["tc model list", "tc model fetch", "tc model probe"]
            examples = [
                "tc model list --output json",
                "tc model fetch Qwen/Qwen2.5-0.5B-Instruct --auto-register --alias qwen",
            ]
            answer = (
                "Model management uses the `tc model` command group. "
                "List models with `tc model list`, download with `tc model fetch`, "
                "and inspect with `tc model probe`."
            )
        elif "geometry" in question_lower:
            related_commands = [
                "tc geometry validate",
                "tc geometry training status",
                "tc geometry primes list",
            ]
            examples = [
                "tc geometry validate --output json",
                "tc geometry training status --job <job-id>",
            ]
            answer = (
                "Geometry commands analyze training dynamics and model alignment. "
                "Use `tc geometry validate` for math validation, "
                "`tc geometry training status` for live metrics."
            )
        elif "dataset" in question_lower:
            related_commands = [
                "tc dataset validate",
                "tc dataset preview",
                "tc dataset convert",
            ]
            examples = [
                "tc dataset validate data.jsonl --output json",
                "tc dataset preview data.jsonl --lines 5",
            ]
            answer = (
                "Dataset commands help prepare training data. "
                "Validate with `tc dataset validate`, preview with `tc dataset preview`, "
                "and convert formats with `tc dataset convert`."
            )
        else:
            related_commands = ["tc inventory", "tc --help"]
            examples = ["tc inventory --output json", "tc train --help"]
            answer = (
                "ModelCypher is a CLI for on-device ML training. "
                "Start with `tc inventory` to see available resources, "
                "or use `tc --help` for command overview."
            )

        return HelpResponse(
            question=question,
            answer=answer,
            related_commands=related_commands,
            examples=examples,
            docs_url="https://github.com/modelcypher/modelcypher/docs",
        )

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
            raise ValueError(
                f"Unsupported shell: {shell}. Supported: {supported_shells}"
            )

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
                "usage": "tc train start --model <id> --dataset <path>",
                "required": ["--model", "--dataset"],
            },
            "model_list": {
                "description": "List registered models",
                "usage": "tc model list --output json",
                "required": [],
            },
            "inventory": {
                "description": "Get complete system state",
                "usage": "tc inventory --output json",
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
        return '''# ModelCypher bash completions
_tc_completions() {
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
complete -F _tc_completions tc
'''

    def _zsh_completions(self) -> str:
        """Generate zsh completion script."""
        return '''#compdef tc
# ModelCypher zsh completions

_tc() {
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

_tc "$@"
'''

    def _fish_completions(self) -> str:
        """Generate fish completion script."""
        return '''# ModelCypher fish completions

complete -c tc -n "__fish_use_subcommand" -a train -d "Training lifecycle"
complete -c tc -n "__fish_use_subcommand" -a job -d "Job management"
complete -c tc -n "__fish_use_subcommand" -a checkpoint -d "Checkpoint management"
complete -c tc -n "__fish_use_subcommand" -a model -d "Model management"
complete -c tc -n "__fish_use_subcommand" -a system -d "System information"
complete -c tc -n "__fish_use_subcommand" -a dataset -d "Dataset management"
complete -c tc -n "__fish_use_subcommand" -a eval -d "Evaluation results"
complete -c tc -n "__fish_use_subcommand" -a compare -d "Comparison history"
complete -c tc -n "__fish_use_subcommand" -a geometry -d "Geometry commands"
complete -c tc -n "__fish_use_subcommand" -a adapter -d "Adapter commands"
complete -c tc -n "__fish_use_subcommand" -a thermo -d "Thermodynamic analysis"
complete -c tc -n "__fish_use_subcommand" -a calibration -d "Calibration commands"
complete -c tc -n "__fish_use_subcommand" -a rag -d "RAG commands"
complete -c tc -n "__fish_use_subcommand" -a stability -d "Stability testing"
complete -c tc -n "__fish_use_subcommand" -a agent-eval -d "Agent evaluation"
complete -c tc -n "__fish_use_subcommand" -a dashboard -d "Dashboard metrics"
complete -c tc -n "__fish_use_subcommand" -a help -d "Help commands"
complete -c tc -n "__fish_use_subcommand" -a inventory -d "System inventory"

# train subcommands
complete -c tc -n "__fish_seen_subcommand_from train" -a start -d "Start training"
complete -c tc -n "__fish_seen_subcommand_from train" -a status -d "Job status"
complete -c tc -n "__fish_seen_subcommand_from train" -a pause -d "Pause job"
complete -c tc -n "__fish_seen_subcommand_from train" -a resume -d "Resume job"
complete -c tc -n "__fish_seen_subcommand_from train" -a cancel -d "Cancel job"
'''
