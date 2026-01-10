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

"""Geometry CRM (Concept Response Matrix) MCP tools.

Contains tools for:
- CRM building
- CRM comparison
- Sequence invariant inventory
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)


def register_geometry_crm_tools(ctx: ServiceContext) -> None:
    """Register geometry CRM tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_crm_build" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_crm_build(
            modelPath: str,
            outputPath: str,
            adapter: str | None = None,
        ) -> dict:
            """Build a concept response matrix (CRM) for a model."""
            model_path = require_existing_directory(modelPath)
            output_path = str(Path(outputPath).expanduser().resolve())
            summary = ctx.geometry_crm_service.build(
                model_path=model_path,
                output_path=output_path,
                adapter=adapter,
            )
            return {
                "_schema": "mc.geometry.crm.build.v1",
                "modelPath": summary.model_path,
                "outputPath": summary.output_path,
                "layerCount": summary.layer_count,
                "hiddenDim": summary.hidden_dim,
                "anchorCount": summary.anchor_count,
                "primeCount": summary.prime_count,
                "gateCount": summary.gate_count,
                "sequenceInvariantCount": summary.sequence_invariant_count,
            }

    if "mc_geometry_crm_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_crm_compare(
            sourcePath: str,
            targetPath: str,
        ) -> dict:
            """Compare two CRMs and compute CKA-based correspondence."""
            source_path = require_existing_path(sourcePath)
            target_path = require_existing_path(targetPath)
            summary = ctx.geometry_crm_service.compare(source_path, target_path)
            payload = {
                "_schema": "mc.geometry.crm.compare.v1",
                "sourcePath": summary.source_path,
                "targetPath": summary.target_path,
                "commonAnchorCount": summary.common_anchor_count,
                "meanCKA": summary.mean_cka,
                "aligned": summary.aligned,
                "layerCorrespondence": summary.layer_correspondence,
            }
            if summary.cka_matrix is not None:
                payload["ckaMatrix"] = summary.cka_matrix
            return payload

    if "mc_geometry_crm_probe_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_crm_probe_inventory(source: str | None = None) -> dict:
            """List available probes for CRM anchoring.

            Args:
                source: Optional filter by source (e.g., sequence_invariant, semantic_prime)
            """
            from modelcypher.core.domain.agents import (
                AtlasSource,
                UnifiedAtlasInventory,
            )

            if source:
                try:
                    atlas_source = AtlasSource(source)
                    probes = UnifiedAtlasInventory.probes_by_source({atlas_source})
                except ValueError:
                    return {
                        "_schema": "mc.error.v1",
                        "error": f"Unknown source: {source}",
                        "validSources": [s.value for s in AtlasSource],
                    }
            else:
                probes = UnifiedAtlasInventory.all_probes()

            counts = UnifiedAtlasInventory.probe_count()
            return {
                "_schema": "mc.geometry.crm.probe_inventory.v1",
                "totalProbes": len(probes),
                "sourceCounts": {src.value: count for src, count in counts.items()},
                "probes": [
                    {
                        "id": p.id,
                        "source": p.source.value,
                        "domain": p.domain.value,
                        "name": p.name,
                        "description": p.description,
                        "weight": p.cross_domain_weight,
                    }
                    for p in probes
                ],
            }
