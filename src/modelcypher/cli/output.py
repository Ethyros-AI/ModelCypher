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

import sys
from typing import Any

from modelcypher.utils.json import dump_json


def write_output(data: Any) -> None:
    sys.stdout.write(dump_json(data, pretty=False))
    sys.stdout.write("\n")


def write_error(error: dict) -> None:
    payload = {"error": error}
    sys.stdout.write(dump_json(payload, pretty=False))
    sys.stdout.write("\n")
