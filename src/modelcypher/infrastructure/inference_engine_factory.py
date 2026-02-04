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

"""Factory for creating InferenceEngine implementations.

Delegates backend selection to the backends layer so infrastructure
does not depend on specific runtime names or frameworks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.inference import HiddenStateEngine


def get_inference_engine() -> "HiddenStateEngine":
    """Get the unified inference engine with default backend.

    Returns:
        InferenceEngine instance using the default backend.
    """
    from modelcypher.adapters.inference_engine import InferenceEngine

    return InferenceEngine()


__all__ = ["get_inference_engine"]
