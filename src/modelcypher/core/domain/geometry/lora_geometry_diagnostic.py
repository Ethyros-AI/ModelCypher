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

"""LoRA geometry diagnostics placeholder.

The implementation relies on NumPy and is intentionally located in
`modelcypher.adapters.geometry.lora_geometry_diagnostic` to avoid CPU fallbacks
in the domain layer.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - explicit import guidance
    raise RuntimeError(
        "LoRA geometry diagnostics live in "
        "`modelcypher.adapters.geometry.lora_geometry_diagnostic`. "
        "Import from adapters to use this functionality."
    )


__all__ = []
