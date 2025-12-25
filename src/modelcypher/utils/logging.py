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
import logging
from dataclasses import asdict, is_dataclass
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str, quiet: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    formatter = (
        JSONFormatter() if quiet else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_extra(**kwargs: Any) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if is_dataclass(value):
            sanitized[key] = asdict(value)
        else:
            sanitized[key] = value
    return {"extra": sanitized}
