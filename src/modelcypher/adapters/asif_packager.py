from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from modelcypher.utils.paths import expand_path


class ASIFPackager:
    def pack(
        self,
        source: str,
        destination: str,
        headroom_percent: int = 15,
        minimum_free_gib: int = 2,
        filesystem: str = "apfs",
        volume_name: str = "DATASET",
        overwrite: bool = False,
    ) -> dict:
        src = expand_path(source)
        dest = expand_path(destination)
        if dest.exists() and not overwrite:
            raise RuntimeError(f"Destination exists: {dest}")

        source_bytes = self._size_bytes(src)
        headroom_bytes = int(source_bytes * headroom_percent / 100)
        minimum_free_bytes = minimum_free_gib * 1024**3
        total_bytes = source_bytes + headroom_bytes + minimum_free_bytes

        if dest.exists():
            dest.unlink()

        subprocess.run(
            ["diskutil", "image", "create", "blank", "--size", str(total_bytes), "--format", "ASIF", str(dest)],
            check=True,
            capture_output=True,
        )

        attach = subprocess.run(
            ["hdiutil", "attach", str(dest), "-nobrowse", "-mountpoint", "/tmp/modelcypher-asif"],
            check=True,
            capture_output=True,
        )

        try:
            if filesystem != "none":
                subprocess.run(
                    [
                        "diskutil",
                        "erasevolume",
                        filesystem,
                        volume_name,
                        "/tmp/modelcypher-asif",
                    ],
                    check=True,
                    capture_output=True,
                )
            subprocess.run(["cp", "-R", str(src), "/tmp/modelcypher-asif"], check=True, capture_output=True)
        finally:
            subprocess.run(["hdiutil", "detach", "/tmp/modelcypher-asif"], check=False, capture_output=True)

        sha256 = self._hash_file(dest)
        return {
            "image": str(dest),
            "volumeName": volume_name,
            "imageBytes": dest.stat().st_size,
            "sourceBytes": source_bytes,
            "headroomBytes": headroom_bytes,
            "sha256": sha256,
        }

    @staticmethod
    def _size_bytes(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
