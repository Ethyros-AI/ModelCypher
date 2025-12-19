from __future__ import annotations

from datetime import datetime

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import ModelInfo


def test_atomic_write_cleans_temp_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    store = FileSystemStore()
    store.register_model(
        ModelInfo(
            id="demo",
            alias="demo",
            architecture="llama",
            format="safetensors",
            path=str(tmp_path / "model"),
            size_bytes=0,
            parameter_count=None,
            is_default_chat=False,
            created_at=datetime.utcnow(),
        )
    )

    tmp_files = list(store.paths.base.glob(".models.json.*.tmp"))
    assert tmp_files == []
    assert store.paths.base.joinpath("models.json.lock").exists()
