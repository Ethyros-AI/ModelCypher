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

"""Inference MCP tools.

Contains tools for:
- Basic inference
- Inference with adapter and security scanning
- Batch inference
- Suite inference
"""

from __future__ import annotations

from .common import (
    MUTATING_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)


def register_inference_tools(ctx: ServiceContext) -> None:
    """Register inference MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_infer" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer(
            model: str,
            prompt: str,
        ) -> dict:
            """Run basic inference with a model and prompt."""
            model_path = require_existing_directory(model)
            result = ctx.inference_engine.infer(model_path, prompt)
            return {
                "_schema": "mc.infer.v1",
                "modelId": result["modelId"],
                "prompt": result["prompt"],
                "response": result["response"],
                "tokenCount": result["tokenCount"],
                "tokensPerSecondTPS": result["tokensPerSecond"],
                "timeToFirstTokenSeconds": result["timeToFirstToken"],
                "totalDurationSeconds": result["totalDuration"],
            }

    if "mc_infer_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_run(
            model: str,
            prompt: str,
            adapter: str | None = None,
            securityScan: bool = False,
        ) -> dict:
            """Execute inference with optional adapter and security scanning."""
            model_path = require_existing_directory(model)
            adapter_path = require_existing_directory(adapter) if adapter else None

            result = ctx.inference_engine.run(
                model=model_path,
                prompt=prompt,
                adapter=adapter_path,
                security_scan=securityScan,
            )

            payload = {
                "_schema": "mc.infer.run.v1",
                "model": result.model,
                "prompt": result.prompt,
                "response": result.response,
                "tokenCount": result.token_count,
                "tokensPerSecond": result.tokens_per_second,
                "timeToFirstToken": result.time_to_first_token,
                "totalDuration": result.total_duration,
                "stopReason": result.stop_reason,
                "adapter": result.adapter,
            }

            if result.security:
                payload["security"] = {
                    "anomalyCount": result.security.anomaly_count,
                    "maxAnomalyScore": result.security.max_anomaly_score,
                    "avgDelta": result.security.avg_delta,
                    "disagreementRate": result.security.disagreement_rate,
                }

            return payload

    if "mc_infer_batch" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_batch(
            model: str,
            promptsFile: str,
        ) -> dict:
            """Execute batched inference from a prompts file."""
            model_path = require_existing_directory(model)
            prompts_path = require_existing_path(promptsFile)
            result = ctx.inference_engine.run_batch(model_path, prompts_path)
            return {
                "_schema": "mc.infer.batch.v1",
                "modelId": result.model_id,
                "promptsFile": result.prompts_file,
                "totalPrompts": result.total_prompts,
                "successful": result.successful,
                "failed": result.failed,
                "totalTokens": result.total_tokens,
                "totalDuration": result.total_duration,
                "averageTokensPerSecond": result.average_tokens_per_second,
                "results": result.results[:10],
            }

    if "mc_infer_suite" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_suite(
            model: str,
            suiteFile: str,
            adapter: str | None = None,
            securityScan: bool = False,
        ) -> dict:
            """Execute batched inference over a suite of prompts."""
            model_path = require_existing_directory(model)
            suite_path = require_existing_path(suiteFile)
            adapter_path = require_existing_directory(adapter) if adapter else None

            result = ctx.inference_engine.suite(
                model=model_path,
                suite_file=suite_path,
                adapter=adapter_path,
                security_scan=securityScan,
            )

            # Convert cases to dict format
            cases_payload = []
            for case in result.cases:
                case_dict = {
                    "name": case.name,
                    "prompt": case.prompt,
                    "response": case.response,
                    "tokenCount": case.token_count,
                    "duration": case.duration,
                    "passed": case.passed,
                    "expected": case.expected,
                }
                if case.error:
                    case_dict["error"] = case.error
                cases_payload.append(case_dict)

            return {
                "_schema": "mc.infer.suite.v1",
                "model": result.model,
                "adapter": result.adapter,
                "suite": result.suite,
                "totalCases": result.total_cases,
                "passed": result.passed,
                "failed": result.failed,
                "totalDuration": result.total_duration,
                "summary": result.summary,
                "cases": cases_payload[:10],
            }
