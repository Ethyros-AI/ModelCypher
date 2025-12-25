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

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    # Handle backend array types (MLX, JAX, numpy, torch) via duck typing
    if hasattr(value, "tolist"):
        return value.tolist()
    # Handle scalar types with .item() method (numpy, MLX, JAX scalars)
    if hasattr(value, "item") and hasattr(value, "ndim") and getattr(value, "ndim", 1) == 0:
        return value.item()
    return value


def dump_json(data: Any, pretty: bool = False) -> str:
    return json.dumps(
        data,
        default=json_default,
        ensure_ascii=True,
        indent=2 if pretty else None,
        sort_keys=True,
    )
