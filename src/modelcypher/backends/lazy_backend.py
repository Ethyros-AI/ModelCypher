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

from typing import Callable

from modelcypher.ports.backend import Backend


class LazyBackend:
    """Lazily instantiate the backend on first use."""

    def __init__(self, factory: Callable[[], Backend]) -> None:
        self._factory = factory
        self._backend: Backend | None = None

    def _get_backend(self) -> Backend:
        if self._backend is None:
            self._backend = self._factory()
        return self._backend

    def __getattr__(self, name: str):
        return getattr(self._get_backend(), name)
