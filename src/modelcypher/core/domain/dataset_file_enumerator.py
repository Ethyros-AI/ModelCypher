from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    data: bytes
    line_number: int


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
                    raise ValueError(
                        f"Line {line_number} exceeds safety limit ({len(line)} bytes, limit {self.max_line_bytes} bytes)"
                    )
                record = DatasetRecord(data=line, line_number=line_number)
                should_continue = process(record)
                emitted += 1
                if should_continue is False:
                    break
        return emitted
