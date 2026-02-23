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
Interpretability Tools.

Feature Steering: Modify model behavior via activation intervention.
- Contrastive directions (Fréchet mean difference)
- Null-space constrained steering (AlphaSteer)
- Refusal direction subtraction

All tools are backend-agnostic and geodesic-principled.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = {
    "feature_steering",
}

_ATTR_TO_MODULE = {
    "FeatureSteering": ("feature_steering", "FeatureSteering"),
    "SteeringVector": ("feature_steering", "SteeringVector"),
    "SteeringConfig": ("feature_steering", "SteeringConfig"),
    "SteeringResult": ("feature_steering", "SteeringResult"),
}


def __getattr__(name: str):
    """Lazy load submodules and commonly used attributes."""
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTR_TO_MODULE:
        module_name, attr_name = _ATTR_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available submodules and attributes."""
    return list(_SUBMODULES) + list(_ATTR_TO_MODULE.keys())
