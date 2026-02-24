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

"""
Atlas Package.

Probe infrastructure for geometric analysis.

Atlas probes are loaded from JSON files in data/probes/.
Use UnifiedAtlasInventory.all_probes() to get all probes.
"""

# Base atlas infrastructure
from .atlas_base import (
    AtlasConcept,
    BaseAtlas,
    BaseAtlasSignature,
)

# Probe loader for JSON-based probes
from .probe_loader import (
    get_probe_count_by_domain,
    load_all_probes,
    load_probes_from_file,
)

# Unified atlas system (JSON-based probes)
from .unified_atlas import (
    AFFECTIVE_DOMAINS,
    ALL_ATLAS_SOURCES,
    COMPUTATIONAL_DOMAINS,
    DEFAULT_ATLAS_SOURCES,
    LINGUISTIC_DOMAINS,
    MATHEMATICAL_DOMAINS,
    MORAL_DOMAINS,
    PHILOSOPHICAL_DOMAINS,
    SAFETY_DOMAINS,
    SPATIOTEMPORAL_DOMAINS,
    AtlasDomain,
    AtlasProbe,
    AtlasSource,
    MultiAtlasTriangulationScore,
    MultiAtlasTriangulationScorer,
    UnifiedAtlasInventory,
    get_probe_ids,
)
