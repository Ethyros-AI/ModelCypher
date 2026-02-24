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

import modelcypher.cli.composition as composition


@dataclass
class _DummyRegistry:
    model_probe: object
    model_loader: object


def test_composition_probe_and_thermo_getters_smoke(monkeypatch) -> None:
    registry = _DummyRegistry(model_probe=object(), model_loader=object())

    composition._get_factory.cache_clear()
    monkeypatch.setattr(composition, "_get_registry", lambda: registry)

    probe_service = composition.get_model_probe_service()
    thermo_service = composition.get_thermo_service()

    assert probe_service is not None
    assert thermo_service is not None
