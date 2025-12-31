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

"""Model management MCP tools.

Contains tools for:
- Model listing and registration
- Model deletion
- Model search (HuggingFace)
- Model probing and architecture analysis
- Model merge validation and execution
- Model alignment analysis
- Model fetching from HuggingFace
"""

from __future__ import annotations

from pathlib import Path

from modelcypher.core.domain.model_search import (
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.mcp.security import ConfirmationError, create_confirmation_response

from .common import (
    DESTRUCTIVE_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
    NETWORK_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_model_tools(ctx: ServiceContext) -> None:
    """Register model management MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_model_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_list() -> dict:
            """List all registered local models."""
            models = ctx.model_service.list_models()
            entries = [
                {
                    "id": model.id,
                    "alias": model.alias,
                    "path": model.path,
                    "architecture": model.architecture,
                    "format": model.format,
                    "sizeBytes": model.size_bytes,
                }
                for model in models
            ]
            return {
                "_schema": "mc.model.list.v1",
                "models": entries,
                "count": len(entries),
            }

    if "mc_model_register" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_model_register(path: str, alias: str | None = None) -> dict:
            """Register a local model."""
            model_path = require_existing_directory(path)
            model = ctx.model_service.register_model(model_path, alias=alias)
            return {
                "_schema": "mc.model.register.v1",
                "modelId": model.id,
                "path": model.path,
                "alias": model.alias,
                "status": "registered",
            }

    if "mc_model_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_model_delete(modelId: str, confirmationToken: str | None = None) -> dict:
            """Delete a model. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                ctx.confirmation_manager.require_confirmation(
                    operation="delete_model",
                    tool_name="mc_model_delete",
                    parameters={"modelId": modelId},
                    description=f"Delete model '{modelId}' from local registry",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete model '{modelId}' from local registry",
                    timeout_seconds=ctx.security_config.confirmation_timeout_seconds,
                )
            ctx.model_service.delete_model(modelId)
            return {
                "_schema": "mc.model.delete.v1",
                "modelId": modelId,
                "status": "deleted",
            }

    if "mc_model_search" in tool_set:

        @mcp.tool(annotations=NETWORK_ANNOTATIONS)
        def mc_model_search(
            query: str | None = None,
            author: str | None = None,
            library: str = "mlx",
            quant: str | None = None,
            sort: str = "downloads",
            limit: int = 20,
            cursor: str | None = None,
        ) -> dict:
            """Search for models on HuggingFace."""
            if limit <= 0:
                raise ValueError("limit must be a positive integer")

            library_key = library.lower()
            if library_key == "mlx":
                library_filter = ModelSearchLibraryFilter.mlx
            elif library_key == "safetensors":
                library_filter = ModelSearchLibraryFilter.safetensors
            elif library_key == "pytorch":
                library_filter = ModelSearchLibraryFilter.pytorch
            elif library_key == "any":
                library_filter = ModelSearchLibraryFilter.any
            else:
                raise ValueError("Invalid library filter. Use: mlx, safetensors, pytorch, any.")

            quant_filter: ModelSearchQuantization | None
            if quant is None:
                quant_filter = None
            else:
                quant_key = quant.lower()
                if quant_key == "4bit":
                    quant_filter = ModelSearchQuantization.four_bit
                elif quant_key == "8bit":
                    quant_filter = ModelSearchQuantization.eight_bit
                elif quant_key == "any":
                    quant_filter = ModelSearchQuantization.any
                else:
                    raise ValueError("Invalid quant filter. Use: 4bit, 8bit, any.")

            sort_key = sort.lower()
            if sort_key == "downloads":
                sort_option = ModelSearchSortOption.downloads
            elif sort_key == "likes":
                sort_option = ModelSearchSortOption.likes
            elif sort_key == "lastmodified":
                sort_option = ModelSearchSortOption.last_modified
            elif sort_key == "trending":
                sort_option = ModelSearchSortOption.trending
            else:
                raise ValueError(
                    "Invalid sort option. Use: downloads, likes, lastModified, trending."
                )

            filters = ModelSearchFilters(
                query=query,
                architecture=None,
                max_size_gb=None,
                author=author,
                library=library_filter,
                quantization=quant_filter,
                sort_by=sort_option,
                limit=min(limit, 100),
            )

            try:
                page = ctx.model_search_service.search(filters, cursor)
            except ModelSearchError as exc:
                raise ValueError(f"Search failed: {exc}") from exc

            models = [
                {
                    "id": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "author": model.author,
                    "pipelineTag": model.pipeline_tag,
                    "tags": model.tags,
                    "isGated": model.is_gated,
                    "isPrivate": model.is_private,
                    "isRecommended": model.is_recommended,
                    "estimatedSizeGB": model.estimated_size_gb,
                    "memoryFitStatus": model.memory_fit_status.value
                    if model.memory_fit_status
                    else None,
                }
                for model in page.models
            ]
            return {
                "_schema": "mc.model.search.v1",
                "count": len(models),
                "hasMore": page.has_more,
                "nextCursor": page.next_cursor,
                "models": models,
            }

    if "mc_model_probe" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_probe(modelPath: str) -> dict:
            """Probe a model for architecture details."""
            model_path = require_existing_directory(modelPath)
            result = ctx.model_probe_service.probe(model_path)
            return {
                "_schema": "mc.model.probe.v1",
                "architecture": result.architecture,
                "parameterCount": result.parameter_count,
                "vocabSize": result.vocab_size,
                "hiddenSize": result.hidden_size,
                "numAttentionHeads": result.num_attention_heads,
                "quantization": result.quantization,
                "layerCount": len(result.layers),
                "layers": [
                    {
                        "name": layer.name,
                        "type": layer.type,
                        "parameters": layer.parameters,
                        "shape": layer.shape,
                    }
                    for layer in result.layers[:20]
                ],
            }

    if "mc_model_validate_merge" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_validate_merge(source: str, target: str) -> dict:
            """Validate merge effort between two models."""
            source_path = require_existing_directory(source)
            target_path = require_existing_directory(target)
            result = ctx.model_probe_service.validate_merge(source_path, target_path)
            return {
                "_schema": "mc.model.validate_merge.v1",
                "lowEffort": result.low_effort,
                "architectureMatch": result.architecture_match,
                "vocabMatch": result.vocab_match,
                "dimensionMatch": result.dimension_match,
                "warnings": result.warnings,
            }

    if "mc_model_merge" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_model_merge(
            source: str,
            target: str,
            output: str,
            transplantDomains: list[str],
            transplantLayers: list[int] | None = None,
            transplantBoundaryK: int | None = None,
            transplantGeodesicK: int | None = None,
            idempotencyKey: str | None = None,
        ) -> dict:
            """Merge two models using null-space constrained transplant.

            Transplant formula: W' = W_target + P_null(A_boundary) @ (W_source - W_target)
            Guarantee: A_boundary @ W' = A_boundary @ W_target (boundary preserved)

            Pipeline: VOCAB → PROBE → TRANSPLANT → VALIDATE

            Args:
                transplantDomains: Core domains to transplant (e.g., ["mathematical", "logical"])
                transplantLayers: Optional layer indices to transplant.
            """
            from modelcypher.core.use_cases.unified_geometric_merge import UnifiedGeometricMerger

            if idempotencyKey:
                previous = ctx.get_idempotency("model_merge", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.model.merge.v1",
                        "status": "duplicate",
                        "message": "Merge already completed with this idempotency key",
                        "outputPath": previous,
                    }

            require_existing_directory(source)
            require_existing_directory(target)
            output_path = Path(output).expanduser().resolve()

            if not transplantDomains:
                return {
                    "_schema": "mc.model.merge.v1",
                    "status": "error",
                    "message": "transplantDomains is required (e.g., ['mathematical', 'logical'])",
                }

            layer_list = None
            if transplantLayers:
                try:
                    layer_list = [int(layer) for layer in transplantLayers]
                except (TypeError, ValueError):
                    return {
                        "_schema": "mc.model.merge.v1",
                        "status": "error",
                        "message": "transplantLayers must be a list of integers",
                    }

            merger = UnifiedGeometricMerger(
                model_loader=ctx.registry.model_loader,
            )
            result = merger.merge(
                source_path=source,
                target_path=target,
                output_dir=str(output_path),
                transplant_domains=transplantDomains,
                transplant_layers=layer_list,
                transplant_boundary_k=transplantBoundaryK,
                transplant_geodesic_k_neighbors=transplantGeodesicK,
            )

            if idempotencyKey:
                ctx.set_idempotency("model_merge", idempotencyKey, str(output_path))

            return {
                "_schema": "mc.model.merge.v1",
                "status": "completed",
                "outputPath": result.output_path,
                "layerCount": result.layer_count,
                "weightCount": result.weight_count,
                "meanConfidence": result.mean_confidence,
                "vocabAligned": result.vocab_aligned,
                "metrics": {
                    "meanProcrustesError": result.mean_procrustes_error,
                },
                "transplantMetrics": {
                    "layersTransplanted": result.transplant_metrics.get("layers_transplanted"),
                    "weightsTransplanted": result.transplant_metrics.get("weights_transplanted"),
                    "meanPreservedFraction": result.transplant_metrics.get("mean_preserved_fraction"),
                    "meanProjectionLoss": result.transplant_metrics.get("mean_projection_loss"),
                    "meanBoundaryRelativeDiff": result.transplant_metrics.get(
                        "mean_boundary_relative_diff"
                    ),
                    "maxBoundaryRelativeDiff": result.transplant_metrics.get(
                        "max_boundary_relative_diff"
                    ),
                    "meanNullDim": result.transplant_metrics.get("mean_null_dim"),
                },
            }

    if "mc_model_analyze_alignment" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_analyze_alignment(modelA: str, modelB: str) -> dict:
            """Analyze alignment drift between two models."""
            path_a = require_existing_directory(modelA)
            path_b = require_existing_directory(modelB)
            result = ctx.model_probe_service.analyze_alignment(path_a, path_b)
            return {
                "_schema": "mc.model.analyze_alignment.v1",
                "driftMagnitude": result.drift_magnitude,
                "driftStd": result.drift_std,
                "driftMin": result.drift_min,
                "driftMax": result.drift_max,
                "driftP50": result.drift_p50,
                "driftP90": result.drift_p90,
                "commonLayerCount": result.common_layer_count,
                "comparableLayerCount": result.comparable_layer_count,
                "missingLayerCount": result.missing_layer_count,
                "layerDrifts": [
                    {
                        "layerName": drift.layer_name,
                        "driftMagnitude": drift.drift_magnitude,
                        "driftZScore": drift.drift_z_score,
                        "comparable": drift.comparable,
                    }
                    for drift in result.layer_drifts[:20]
                ],
            }

    if "mc_model_fetch" in tool_set:

        @mcp.tool(annotations=NETWORK_ANNOTATIONS)
        def mc_model_fetch(
            modelId: str,
            revision: str = "main",
            idempotencyKey: str | None = None,
        ) -> dict:
            """Fetch a model from HuggingFace."""
            if idempotencyKey:
                previous = ctx.get_idempotency("model_fetch", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.model.fetch.v1",
                        "wasExecuted": False,
                        "modelId": None,
                        "path": None,
                        "status": None,
                        "previousPath": previous,
                        "message": "Model already downloaded with this idempotency key",
                    }

            result = ctx.model_service.fetch_model(modelId, revision, False, None, None)
            local_path = result["localPath"]
            if idempotencyKey:
                ctx.set_idempotency("model_fetch", idempotencyKey, local_path)
            return {
                "_schema": "mc.model.fetch.v1",
                "wasExecuted": True,
                "modelId": modelId,
                "path": local_path,
                "status": "completed",
                "previousPath": None,
                "message": None,
            }
