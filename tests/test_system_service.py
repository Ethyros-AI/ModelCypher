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

from pathlib import Path
from types import SimpleNamespace

from modelcypher.core.use_cases.system_service import SystemService


class _DummyStore:
    def __init__(self) -> None:
        self.paths = SimpleNamespace(base=Path("."))


def test_system_probe_cuda_payload() -> None:
    service = SystemService(_DummyStore())
    payload = service.probe("cuda")
    assert payload["target"] == "cuda"
    assert "cuda" in payload
    assert "available" in payload["cuda"]
    assert "version" in payload["cuda"]


def test_system_probe_jax_payload() -> None:
    service = SystemService(_DummyStore())
    payload = service.probe("jax")
    assert payload["target"] == "jax"
    assert "jax" in payload
    assert "available" in payload["jax"]
    assert "version" in payload["jax"]


def test_system_readiness_includes_backends() -> None:
    service = SystemService(_DummyStore())
    payload = service.readiness()
    assert "mlxVersion" in payload
    assert "cudaVersion" in payload
    assert "jaxVersion" in payload
    assert "preferredBackend" in payload
