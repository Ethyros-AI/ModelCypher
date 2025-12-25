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
import sys
from typing import Any

import yaml

from modelcypher.utils.json import dump_json


def write_output(data: Any, output_format: str, pretty: bool = False) -> None:
    if output_format == "json":
        sys.stdout.write(dump_json(data, pretty=pretty))
        sys.stdout.write("\n")
        return
    if output_format == "yaml":
        normalized = json.loads(dump_json(data, pretty=False))
        yaml.safe_dump(normalized, sys.stdout, sort_keys=True)
        return
    sys.stdout.write(str(data))
    sys.stdout.write("\n")


def write_error(error: dict, output_format: str, pretty: bool = False) -> None:
    payload = {"error": error}
    if output_format == "json":
        sys.stdout.write(dump_json(payload, pretty=pretty))
        sys.stdout.write("\n")
        return
    if output_format == "yaml":
        normalized = json.loads(dump_json(payload, pretty=False))
        yaml.safe_dump(normalized, sys.stdout, sort_keys=True)
        return
    sys.stderr.write(f"Error: {error.get('title')}\n{error.get('detail')}\n")
