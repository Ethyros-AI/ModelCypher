from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def dump_json(data: Any, pretty: bool = False) -> str:
    return json.dumps(
        data,
        default=json_default,
        ensure_ascii=True,
        indent=2 if pretty else None,
        sort_keys=True,
    )
