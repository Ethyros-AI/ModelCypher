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

"""Register default probe inventories for geometry modules.

This is the sanctioned bridge: outer layers load agent inventories and
register them into the geometry registry, keeping geometry independent.
"""

from __future__ import annotations

from modelcypher.core.domain.geometry import atlas_registry


def register_default_atlas_registry() -> None:
    """Register default inventories from the agents domain."""
    from modelcypher.core.domain.agents.computational_gate_atlas import (
        ComputationalGateInventory,
    )
    from modelcypher.core.domain.agents.metaphor_invariant_atlas import (
        MetaphorInvariantInventory,
    )
    from modelcypher.core.domain.agents.moral_atlas import MoralConceptInventory
    from modelcypher.core.domain.agents.sequence_invariant_atlas import (
        SequenceInvariantInventory,
        TriangulationScorer,
    )
    from modelcypher.core.domain.agents.social_atlas import SocialConceptInventory
    from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory
    from modelcypher.core.domain.agents.temporal_atlas import TemporalConceptInventory
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    atlas_registry.register_atlas_probes(UnifiedAtlasInventory.all_probes())
    atlas_registry.register_sequence_invariants(SequenceInvariantInventory.all_probes())
    atlas_registry.register_sequence_triangulation_scorer(TriangulationScorer.compute_score)
    atlas_registry.register_gate_inventory(ComputationalGateInventory.all_gates())
    atlas_registry.register_spatial_concepts(SpatialConceptInventory.all_concepts())
    atlas_registry.register_social_concepts(SocialConceptInventory.all_concepts())
    atlas_registry.register_temporal_concepts(TemporalConceptInventory.all_concepts())
    atlas_registry.register_moral_concepts(MoralConceptInventory.all_concepts())
    atlas_registry.register_metaphor_invariants(MetaphorInvariantInventory.ALL_PROBES)
