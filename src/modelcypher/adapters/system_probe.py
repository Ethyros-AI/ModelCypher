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

from __future__ import annotations

from modelcypher.backends.mlx_probe import get_mlx_probe_error, probe_mlx_available
from modelcypher.ports.system_probe import SystemProbePort


class MLXSystemProbe(SystemProbePort):
    """Adapter for MLX runtime probes."""

    def mlx_available(self, explicit: bool = False) -> bool:
        return probe_mlx_available(explicit=explicit)

    def mlx_probe_error(self) -> str | None:
        return get_mlx_probe_error()
