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

"""MCP tools for multi-donor transplant program management."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.mcp.tools.common import ServiceContext


def register_program_tools(context: "ServiceContext") -> None:
    """Register program management tools with the MCP server."""
    mcp = context.mcp
    tool_set = context.tool_set

    MUTATING_ANNOTATIONS = {"category": "model_merge"}
    READ_ONLY_ANNOTATIONS = {"category": "read_only"}

    if "mc_program_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_program_run(
            configPath: str,
            parallel: bool = False,
            maxWorkers: int = 2,
            dryRun: bool = False,
            baseFilter: str | None = None,
        ) -> dict:
            """Execute a multi-donor transplant program.

            Sequentially transplants knowledge from multiple donors into base model(s).
            Supports checkpointing for resumability and parallel execution across bases.

            Args:
                configPath: Path to program YAML/JSON config file
                parallel: Process base models in parallel
                maxWorkers: Max parallel workers (if parallel=True)
                dryRun: Validate program without execution
                baseFilter: Only process specific base model (by ID)

            Returns:
                Program execution result with per-donor metrics
            """
            from modelcypher.core.use_cases.multi_donor_merge import (
                MultiDonorMergeService,
                TransplantProgram,
            )

            config_path = Path(configPath).expanduser().resolve()
            if not config_path.exists():
                return {
                    "_schema": "mc.program.run.v1",
                    "status": "error",
                    "message": f"Config file not found: {config_path}",
                }

            try:
                program = TransplantProgram.from_yaml(config_path)
            except ValueError as e:
                return {
                    "_schema": "mc.program.run.v1",
                    "status": "error",
                    "message": f"Invalid program config: {e}",
                }

            # Filter to specific base if requested
            if baseFilter:
                matching_bases = tuple(b for b in program.bases if b.id == baseFilter)
                if not matching_bases:
                    return {
                        "_schema": "mc.program.run.v1",
                        "status": "error",
                        "message": f"Base '{baseFilter}' not found. Available: {', '.join(b.id for b in program.bases)}",
                    }
                program = TransplantProgram(
                    name=program.name,
                    description=program.description,
                    bases=matching_bases,
                    donors=program.donors,
                    evaluation=program.evaluation,
                    output_dir=program.output_dir,
                )

            if dryRun:
                return {
                    "_schema": "mc.program.run.v1",
                    "status": "valid",
                    "program": program.name,
                    "bases": len(program.bases),
                    "donors": len(program.donors),
                    "baseIds": [b.id for b in program.bases],
                    "donorIds": [d.id for d in program.donors],
                }

            service = MultiDonorMergeService()
            result = service.execute_program(
                program=program,
                parallel=parallel,
                max_workers=maxWorkers,
            )

            return {
                "_schema": "mc.program.run.v1",
                **result.to_dict(),
            }

    if "mc_program_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_program_status(programId: str) -> dict:
            """Get status of a running or completed program.

            Args:
                programId: Program ID to check status

            Returns:
                Program status including progress per base model
            """
            from modelcypher.core.use_cases.multi_donor_merge import MultiDonorMergeService

            service = MultiDonorMergeService()

            try:
                status = service.get_program_status(programId)
                return {
                    "_schema": "mc.program.status.v1",
                    **status.to_dict(),
                }
            except FileNotFoundError:
                return {
                    "_schema": "mc.program.status.v1",
                    "status": "error",
                    "message": f"Program '{programId}' not found",
                }

    if "mc_program_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_program_list() -> dict:
            """List all programs (running, completed, failed).

            Returns:
                List of program statuses
            """
            from modelcypher.core.use_cases.multi_donor_merge import MultiDonorMergeService

            service = MultiDonorMergeService()
            programs = service.list_programs()

            return {
                "_schema": "mc.program.list.v1",
                "programs": [
                    {
                        "programId": p.program_id,
                        "programName": p.program_name,
                        "status": p.status,
                        "startedAt": p.started_at.isoformat(),
                        "updatedAt": p.updated_at.isoformat(),
                    }
                    for p in programs
                ],
                "count": len(programs),
            }

    if "mc_program_show" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_program_show(configPath: str) -> dict:
            """Show details of a program configuration.

            Args:
                configPath: Path to program YAML/JSON config file

            Returns:
                Parsed program configuration
            """
            from modelcypher.core.use_cases.multi_donor_merge import TransplantProgram

            config_path = Path(configPath).expanduser().resolve()
            if not config_path.exists():
                return {
                    "_schema": "mc.program.show.v1",
                    "status": "error",
                    "message": f"Config file not found: {config_path}",
                }

            try:
                program = TransplantProgram.from_yaml(config_path)
                return {
                    "_schema": "mc.program.show.v1",
                    "name": program.name,
                    "description": program.description,
                    "bases": [
                        {
                            "id": b.id,
                            "source": b.source,
                            "alias": b.effective_alias,
                        }
                        for b in program.bases
                    ],
                    "donors": [
                        {
                            "id": d.id,
                            "source": d.source,
                            "domains": list(d.domains),
                            "priority": d.priority,
                            "layers": list(d.layers) if d.layers else None,
                        }
                        for d in program.donors
                    ],
                    "evaluation": program.evaluation.to_dict(),
                    "outputDir": program.output_dir,
                }
            except ValueError as e:
                return {
                    "_schema": "mc.program.show.v1",
                    "status": "error",
                    "message": f"Invalid program config: {e}",
                }
