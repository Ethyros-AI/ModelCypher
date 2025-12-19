from __future__ import annotations

from pathlib import Path


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = expand_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
