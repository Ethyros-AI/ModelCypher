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

"""Tests for ModelProfileStore persistence behavior."""

from __future__ import annotations

import json
from pathlib import Path

from modelcypher.core.domain.geometry.model_profile import (
    ModelProfile,
    ModelProfileStore,
)


def _create_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 64}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    return model_dir


def test_profile_store_writes_global_and_sidecar(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    model_dir = _create_model_dir(tmp_path)

    store = ModelProfileStore()
    profile, identity = store.ensure(str(model_dir))
    store.save(profile, identity)

    global_path = store.profile_path(identity.model_id)
    sidecar_path = store.sidecar_path(str(model_dir))

    assert global_path.exists()
    assert sidecar_path.exists()

    loaded = ModelProfile.load(global_path)
    assert loaded.model_id == identity.model_id
    assert loaded.config_hash == identity.config_hash
    assert loaded.weights_hash == identity.weights_hash


def test_profile_store_loads_sidecar_when_global_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    model_dir = _create_model_dir(tmp_path)

    store = ModelProfileStore()
    profile, identity = store.ensure(str(model_dir))
    store.save(profile, identity)

    global_path = store.profile_path(identity.model_id)
    global_path.unlink()

    fresh_store = ModelProfileStore()
    loaded, loaded_identity = fresh_store.load(str(model_dir))

    assert loaded is not None
    assert loaded_identity.model_id == identity.model_id
    assert loaded.model_id == identity.model_id
    assert loaded.config_hash == identity.config_hash
    assert loaded.weights_hash == identity.weights_hash
