from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_json(name: str) -> Any:
    data = resources.files("modelcypher.data").joinpath(name).read_text(encoding="utf-8")
    return json.loads(data)
