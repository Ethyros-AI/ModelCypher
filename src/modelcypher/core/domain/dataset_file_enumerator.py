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

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    data: bytes
    line_number: int


class DatasetEnumerationError(ValueError):
    pass


class DatasetLineTooLargeError(DatasetEnumerationError):
    def __init__(self, line_number: int, length: int, limit: int) -> None:
        super().__init__(
            f"Line {line_number} exceeds safety limit ({length} bytes, limit {limit} bytes)"
        )
        self.line_number = line_number
        self.length = length
        self.limit = limit


class DatasetFileEnumerator:
    def __init__(self, max_line_bytes: int = 2 * 1024 * 1024) -> None:
        self.max_line_bytes = max_line_bytes

    def enumerate_lines(self, path: str, process) -> int:
        emitted = 0
        file_path = Path(path)
        with file_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip(b"\r\n")
                if len(line) > self.max_line_bytes:
                    raise DatasetLineTooLargeError(
                        line_number=line_number,
                        length=len(line),
                        limit=self.max_line_bytes,
                    )
                record = DatasetRecord(data=line, line_number=line_number)
                should_continue = process(record)
                emitted += 1
                if should_continue is False:
                    break
        return emitted
