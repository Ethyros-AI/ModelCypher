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

from modelcypher.backends import detect_default_backend_type, probe_backends
from modelcypher.ports.system_probe import BackendProbe, SystemProbePort


class BackendSystemProbe(SystemProbePort):
    """Adapter for backend runtime probes."""

    def probe_backends(self, explicit: bool = False) -> list[BackendProbe]:
        return [
            BackendProbe(
                key=descriptor.key,
                display_name=descriptor.display_name,
                available=descriptor.available,
                error=descriptor.error,
                system_info=descriptor.system_info,
            )
            for descriptor in probe_backends(explicit=explicit)
        ]

    def default_backend_key(self) -> str:
        return detect_default_backend_type()
