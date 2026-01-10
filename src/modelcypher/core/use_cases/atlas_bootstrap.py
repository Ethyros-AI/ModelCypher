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
"""

from __future__ import annotations

from modelcypher.core.domain.geometry import atlas_registry


def register_default_atlas_inventories() -> None:
    """Register default inventories from the agents domain.

    All probes are loaded from the unified JSON-based probe system.
    """
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    atlas_registry.register_atlas_probes(UnifiedAtlasInventory.all_probes())
