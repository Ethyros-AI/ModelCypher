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

"""MoE domain primitives: topology, routing analysis, and expert selection."""

from modelcypher.core.domain.moe.expert_selection import (
    ExpertTarget,
    ExpertTargetSelection,
    select_expert_targets,
)
from modelcypher.core.domain.moe.routing_analysis import (
    ExpertRoutingStats,
    RoutingProfile,
    build_routing_profile,
)
from modelcypher.core.domain.moe.topology import MoETopology

__all__ = [
    "ExpertRoutingStats",
    "ExpertTarget",
    "ExpertTargetSelection",
    "MoETopology",
    "RoutingProfile",
    "build_routing_profile",
    "select_expert_targets",
]
