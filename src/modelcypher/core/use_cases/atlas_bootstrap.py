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

All probes are now loaded from JSON files in data/probes/.
Moral concepts are loaded from the moral_concepts module.
"""

from __future__ import annotations

from modelcypher.core.domain.geometry import atlas_registry


def register_default_atlas_inventories() -> None:
    """Register default inventories from the agents domain.

    All probes are loaded from the unified JSON-based probe system.
    Specialized inventories (moral, gates) are registered separately.
    """
    from modelcypher.core.domain.atlas.gate_inventory import ComputationalGateInventory
    from modelcypher.core.domain.atlas.moral_concepts import MoralConceptInventory
    from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory

    atlas_registry.register_atlas_probes(UnifiedAtlasInventory.all_probes())
    atlas_registry.register_moral_concepts(MoralConceptInventory.all_concepts())
    atlas_registry.register_gate_inventory(ComputationalGateInventory.all_gates())
