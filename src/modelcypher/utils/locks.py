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

import time
from pathlib import Path


class FileLockError(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def acquire(self, timeout: float = 0.0) -> None:
        import fcntl

        start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a", encoding="utf-8")
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if timeout <= 0 or (time.time() - start) >= timeout:
                    raise FileLockError(f"Lock already held: {self.path}")
                time.sleep(0.1)

    def release(self) -> None:
        if not self.handle:
            return
        import fcntl

        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def is_locked(self) -> bool:
        """Check if the lock file is currently held by another process."""
        import fcntl

        if not self.path.exists():
            return False
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
