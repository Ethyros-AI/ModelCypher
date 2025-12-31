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

"""Evaluation MCP tools.

Contains tools for:
- Running evaluations on models
- Listing evaluation runs
- Showing evaluation results
"""

from __future__ import annotations

from modelcypher.core.use_cases.evaluation_service import EvalConfig

from .common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)


def register_evaluation_tools(ctx: ServiceContext) -> None:
    """Register evaluation MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_eval_run" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_run(
            model: str,
            dataset: str,
            metrics: list[str] | None = None,
            batchSize: int = 4,
            maxSamples: int | None = None,
        ) -> dict:
            """Run evaluation on a model."""
            model_path = require_existing_directory(model)
            dataset_path = require_existing_path(dataset)

            config = EvalConfig(
                metrics=metrics,
                batch_size=batchSize,
                max_samples=maxSamples,
            )

            result = ctx.evaluation_service.run(model_path, dataset_path, config)

            return {
                "_schema": "mc.eval.run.v1",
                "evalId": result.eval_id,
                "averageLoss": result.average_loss,
                "perplexity": result.perplexity,
                "sampleCount": result.sample_count,
            }

    if "mc_eval_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_list(limit: int = 50) -> dict:
            return ctx.evaluation_service.list_evaluations(limit)

    if "mc_eval_show" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_show(evalId: str) -> dict:
            return ctx.evaluation_service.results(evalId)
