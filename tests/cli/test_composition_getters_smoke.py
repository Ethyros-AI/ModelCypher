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
import modelcypher.core.use_cases.observation_service as observation_module
from modelcypher.core.use_cases.observation_bundle_report_service import (
    ObservationBundleReportService,
)


@dataclass
class _DummyRegistry:
    backend: object
    activation_provider: object
    model_probe: object
    model_loader: object


def test_composition_probe_and_thermo_getters_smoke(monkeypatch) -> None:
    registry = _DummyRegistry(
        backend=object(),
        activation_provider=object(),
        model_probe=object(),
        model_loader=object(),
    )

    composition._get_factory.cache_clear()
    monkeypatch.setattr(composition, "_get_registry", lambda: registry)

    probe_service = composition.get_model_probe_service()
    thermo_service = composition.get_thermo_service()

    assert probe_service is not None
    assert thermo_service is not None


def test_composition_observation_getter_injects_sublayer_collector(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel_collector = object()
    registry = _DummyRegistry(
        backend=object(),
        activation_provider=object(),
        model_probe=object(),
        model_loader=object(),
    )

    class _StubObservationService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    composition._get_factory.cache_clear()
    monkeypatch.setattr(composition, "_get_registry", lambda: registry)
    monkeypatch.setattr(
        "modelcypher.backends.sublayer_collector.collect_sublayer_activations",
        sentinel_collector,
    )
    monkeypatch.setattr(
        observation_module,
        "ObservationService",
        _StubObservationService,
    )

    service = composition.get_observation_service()

    assert isinstance(service, _StubObservationService)
    assert captured["backend"] is registry.backend
    assert captured["activation_provider"] is registry.activation_provider
    assert captured["model_loader"] is registry.model_loader
    assert captured["sublayer_collector"] is sentinel_collector


def test_composition_observation_bundle_report_getter_smoke() -> None:
    service = composition.get_observation_bundle_report_service()
    assert isinstance(service, ObservationBundleReportService)
