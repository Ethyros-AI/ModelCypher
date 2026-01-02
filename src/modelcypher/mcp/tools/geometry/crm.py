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
            includePrimes: bool = True,
            includeGates: bool = True,
            includePolyglot: bool = True,
            includeSequenceInvariants: bool = True,
            sequenceFamilies: list[str] | None = None,
            anchorPrefixes: list[str] | None = None,
        ) -> dict:
            """Build a concept response matrix (CRM) for a model."""
            from modelcypher.core.domain.agents.sequence_invariant_atlas import SequenceFamily
            from modelcypher.core.use_cases.concept_response_matrix_service import CRMBuildConfig

            model_path = require_existing_directory(modelPath)
            output_path = str(Path(outputPath).expanduser().resolve())
            parsed_families: frozenset[SequenceFamily] | None = None
            if sequenceFamilies:
                family_set: set[SequenceFamily] = set()
                for name in sequenceFamilies:
                    try:
                        family_set.add(SequenceFamily(name.strip().lower()))
                    except ValueError:
                        pass
                if family_set:
                    parsed_families = frozenset(family_set)
            config = CRMBuildConfig(
                include_primes=includePrimes,
                include_gates=includeGates,
                include_polyglot=includePolyglot,
                include_sequence_invariants=includeSequenceInvariants,
                sequence_families=parsed_families,
                anchor_prefixes=anchorPrefixes,
            )
            summary = ctx.geometry_crm_service.build(
                model_path=model_path,
                output_path=output_path,
                config=config,
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
            includeMatrix: bool = False,
        ) -> dict:
            """Compare two CRMs and compute CKA-based correspondence."""
            source_path = require_existing_path(sourcePath)
            target_path = require_existing_path(targetPath)
            summary = ctx.geometry_crm_service.compare(
                source_path, target_path, include_matrix=includeMatrix
            )
            payload = {
                "_schema": "mc.geometry.crm.compare.v1",
                "sourcePath": summary.source_path,
                "targetPath": summary.target_path,
                "commonAnchorCount": summary.common_anchor_count,
                "overallAlignment": summary.overall_alignment,
                "layerCorrespondence": summary.layer_correspondence,
            }
            if summary.cka_matrix is not None:
                payload["ckaMatrix"] = summary.cka_matrix
            return payload

    if "mc_geometry_crm_sequence_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_crm_sequence_inventory(family: str | None = None) -> dict:
            """List available sequence invariant probes for CRM anchoring."""
            from modelcypher.core.domain.agents.sequence_invariant_atlas import (
                SequenceFamily,
                SequenceInvariantInventory,
            )

            family_filter: set[SequenceFamily] | None = None
            if family:
                try:
                    family_filter = {SequenceFamily(family.strip().lower())}
                except ValueError:
                    return {
                        "_schema": "mc.error.v1",
                        "error": f"Unknown family '{family}'",
                        "validFamilies": [f.value for f in SequenceFamily],
                    }
            probes = SequenceInvariantInventory.probes_for_families(family_filter)
            counts = SequenceInvariantInventory.probe_count_by_family()
            return {
                "_schema": "mc.geometry.crm.sequence_inventory.v1",
                "totalProbes": len(probes),
                "familyCounts": {fam.value: count for fam, count in counts.items()},
                "probes": [
                    {
                        "id": p.id,
                        "family": p.family.value,
                        "domain": p.domain.value,
                        "name": p.name,
                        "description": p.description,
                        "weight": p.cross_domain_weight,
                    }
                    for p in probes
                ],
            }
