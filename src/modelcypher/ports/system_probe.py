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

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendProbe:
    """Probe result for a single backend runtime."""

    key: str
    display_name: str
    available: bool
    error: str | None = None
    system_info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SystemProbePort(Protocol):
    """Port for probing platform-specific runtime capabilities."""

    def probe_backends(self, explicit: bool = False) -> list[BackendProbe]:
        """Return probe results for all known backends."""

    def default_backend_key(self) -> str:
        """Return the preferred backend key for this machine."""
