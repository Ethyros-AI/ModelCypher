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
register them into the geometry registry at
modelcypher.core.domain.geometry.atlas_registry.
"""

from __future__ import annotations

from modelcypher.core.domain.geometry import atlas_registry


def register_default_atlas_inventories() -> None:
    """Register default inventories from the agents domain.

    Optional atlas modules are loaded gracefully - missing modules are skipped.
    """
    # Core unified atlas (required)
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    atlas_registry.register_atlas_probes(UnifiedAtlasInventory.all_probes())

    # Optional specialized atlases - load if available
    try:
        from modelcypher.core.domain.agents.computational_gate_atlas import (
            ComputationalGateInventory,
        )

        atlas_registry.register_gate_inventory(ComputationalGateInventory.all_gates())
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.sequence_invariant_atlas import (
            SequenceInvariantInventory,
            TriangulationScorer,
        )

        atlas_registry.register_sequence_invariants(SequenceInvariantInventory.all_probes())
        atlas_registry.register_sequence_triangulation_scorer(TriangulationScorer.compute_score)
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory

        atlas_registry.register_spatial_concepts(SpatialConceptInventory.all_concepts())
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.social_atlas import SocialConceptInventory

        atlas_registry.register_social_concepts(SocialConceptInventory.all_concepts())
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.temporal_atlas import TemporalConceptInventory

        atlas_registry.register_temporal_concepts(TemporalConceptInventory.all_concepts())
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.moral_atlas import MoralConceptInventory

        atlas_registry.register_moral_concepts(MoralConceptInventory.all_concepts())
    except ImportError:
        pass

    try:
        from modelcypher.core.domain.agents.metaphor_invariant_atlas import (
            MetaphorInvariantInventory,
        )

        atlas_registry.register_metaphor_invariants(MetaphorInvariantInventory.ALL_PROBES)
    except ImportError:
        pass
