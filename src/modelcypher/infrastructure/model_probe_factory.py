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

"""Factory for creating ModelProbePort implementations.

Creates a Backend-based model probe that works with any framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.model_probe import ModelProbePort


def get_model_probe() -> "ModelProbePort":
    """Get the model probe for the current runtime.

    Returns:
        A BackendModelProbe instance that uses the Backend protocol.
    """
    from modelcypher.adapters.model_probe import BackendModelProbe
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()
    return BackendModelProbe(backend=backend)


__all__ = ["get_model_probe"]
