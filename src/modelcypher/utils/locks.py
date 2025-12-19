from __future__ import annotations

import os
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

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
