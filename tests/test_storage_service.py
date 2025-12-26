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
from datetime import datetime
from pathlib import Path

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import CheckpointRecord, ModelInfo
from modelcypher.core.use_cases.storage_service import BYTES_PER_GB, StorageService


@dataclass
class _DiskUsage:
    total: int
    used: int
    free: int


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_storage_usage_computes_sizes(tmp_path, monkeypatch) -> None:
    home = tmp_path / "mc_home"
    hf_home = tmp_path / "hf_cache"
    monkeypatch.setenv("MODELCYPHER_HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(hf_home))

    store = FileSystemStore()

    model_dir = tmp_path / "model_dir"
    _write_bytes(model_dir / "weights.bin", 2048)
    store.register_model(
        ModelInfo(
            id="demo",
            alias="demo",
            architecture="llama",
            format="safetensors",
            path=str(model_dir),
            size_bytes=2048,
            parameter_count=None,
            is_default_chat=False,
            created_at=datetime.utcnow(),
        )
    )

    checkpoint_path = tmp_path / "checkpoint.safetensors"
    _write_bytes(checkpoint_path, 512)
    store.add_checkpoint(
        CheckpointRecord(
            job_id="job-1",
            step=1,
            loss=0.5,
            timestamp=datetime.utcnow(),
            file_path=str(checkpoint_path),
        )
    )

    _write_bytes(home / "caches" / "cache.bin", 256)
    _write_bytes(hf_home / "models" / "blob", 64)
    _write_bytes(home / "rag" / "index.bin", 128)

    def disk_usage_provider(_path: str) -> _DiskUsage:
        total = 10 * BYTES_PER_GB
        free = 5 * BYTES_PER_GB
        return _DiskUsage(total=total, used=total - free, free=free)

    service = StorageService(
        model_store=store,
        job_store=store,
        base_dir=store.paths.base,
        logs_dir=store.paths.logs,
        disk_usage_provider=disk_usage_provider,
        cache_ttl_seconds=0.0,
    )
    usage = service.storage_usage()

    assert usage.total_gb == 10.0
    assert abs(usage.models_gb - (2048 / BYTES_PER_GB)) < 1e-9
    assert abs(usage.checkpoints_gb - (512 / BYTES_PER_GB)) < 1e-9
    expected_other = (256 + 64) / BYTES_PER_GB
    assert abs(usage.other_gb - expected_other) < 1e-9


def test_storage_cleanup_clears_targets(tmp_path, monkeypatch) -> None:
    home = tmp_path / "mc_home"
    hf_home = tmp_path / "hf_cache"
    monkeypatch.setenv("MODELCYPHER_HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(hf_home))

    _write_bytes(home / "caches" / "cache.bin", 100)
    _write_bytes(hf_home / "models" / "blob", 200)
    _write_bytes(home / "rag" / "index.bin", 300)

    store = FileSystemStore()
    service = StorageService(
        model_store=store,
        job_store=store,
        base_dir=store.paths.base,
        logs_dir=store.paths.logs,
        cache_ttl_seconds=0.0,
    )
    cleared = service.cleanup(["caches"])

    assert "caches" in cleared
    assert list((home / "caches").iterdir()) == []
    assert list(hf_home.iterdir()) == []
