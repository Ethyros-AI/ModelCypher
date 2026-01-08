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
Domain layer - core business logic and models.

This module uses lazy imports to avoid loading all subpackages at import time.
Subpackages are loaded on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# =============================================================================
# Subdomain package registry for lazy loading
# =============================================================================

_SUBPACKAGES = {
    "adapters",
    "agents",
    "entropy",
    "geometry",
    "inference",
    "merging",
    "research",
    "safety",
    "thermo",
    "training",
    "validation",
}

# =============================================================================
# Root-level module registry
# =============================================================================

_ROOT_MODULES = {
    "chat_template",
    "model_search",
    "models",
    "settings",
    "storage_usage",
}

def __getattr__(name: str):
    """Lazy load subpackages and root-level modules."""
    # Check subpackages first
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)

    # Check root-level modules
    if name in _ROOT_MODULES:
        return importlib.import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available subpackages and root-level modules."""
    return list(_SUBPACKAGES) + list(_ROOT_MODULES)


# TYPE_CHECKING for static analysis only
if TYPE_CHECKING:
    from . import (  # noqa: F401
        adapters,
        agents,
        entropy,
        geometry,
        inference,
        merging,
        research,
        safety,
        thermo,
        training,
        validation,
    )
