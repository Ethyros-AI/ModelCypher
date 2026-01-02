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

"""Stage functions for geometric merge orchestration."""

from .alignment import stage_compute_alignment
from .analyze import stage_analyze_geometry
from .correspondence import stage_layer_correspondence
from .interference import stage_analyze_interference
from .probe import stage_probe_fingerprint
from .shared_structure import stage_find_shared_structure
from .validate import stage_validate

__all__ = [
    "stage_probe_fingerprint",
    "stage_layer_correspondence",
    "stage_analyze_geometry",
    "stage_find_shared_structure",
    "stage_compute_alignment",
    "stage_analyze_interference",
    "stage_validate",
]
