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

from types import SimpleNamespace

import pytest

from modelcypher.experimental.self_improve.improver import AutonomousSelfImprover
from modelcypher.experimental.self_improve.types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
)


class _ToyModel:
    def __init__(self, n_layers: int = 6) -> None:
        self.model = SimpleNamespace(layers=[object() for _ in range(n_layers)])


def _disconnected_analysis(capability: Capability) -> CapabilityAnalysis:
    return CapabilityAnalysis(
        capability=capability,
        status=CapabilityStatus.DISCONNECTED,
        accuracy_raw=0.0,
        accuracy_primed=1.0,
        kappa_raw=float("nan"),
        kappa_primed=float("nan"),
        best_prime="say",
    )


def test_phase2b_records_steering_action(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = Capability.from_lists(
        name="arithmetic",
        prompts=["1+1="],
        problems=[("1+1=", "2")],
    )
    improver = AutonomousSelfImprover(_ToyModel(), tokenizer=object())
    observed: dict[str, object] = {}

    def fake_scan(
        cap: Capability,
        accuracy_threshold: float,
        primes: object | None = None,
    ) -> CapabilityAnalysis:
        del primes
        observed["threshold"] = accuracy_threshold
        return _disconnected_analysis(cap)

    def fake_collect(
        capability: Capability,
        best_prime: str,
        target_layer: int,
    ) -> tuple[str, str]:
        observed["collect"] = (capability.name, best_prime, target_layer)
        return ("positive", "negative")

    monkeypatch.setattr(improver.scanner, "scan", fake_scan)
    monkeypatch.setattr(
        improver.scanner,
        "collect_contrastive_activations",
        fake_collect,
    )

    class FakeFeatureSteering:
        def __init__(self, _model: object, backend: object) -> None:
            observed["backend"] = backend

        def _get_layers(self) -> list[object]:
            return [object() for _ in range(6)]

        def extract_contrastive_direction(
            self,
            positive_activations: object,
            negative_activations: object,
            layer: int,
            label: str,
        ) -> SimpleNamespace:
            observed["extract"] = (
                positive_activations,
                negative_activations,
                layer,
                label,
            )
            return SimpleNamespace(direction="raw_direction", strength_range=(-0.5, 0.5))

        def project_to_null_space(
            self,
            steering_direction: object,
            prior_activations: object,
        ) -> tuple[str, float]:
            observed["project"] = (steering_direction, prior_activations)
            return ("projected_direction", 0.25)

    import modelcypher.experimental.interpretability.feature_steering as steering_mod

    backend_sentinel = object()
    monkeypatch.setattr(
        "modelcypher.core.domain._backend.get_default_backend",
        lambda: backend_sentinel,
    )
    monkeypatch.setattr(steering_mod, "FeatureSteering", FakeFeatureSteering)

    log = improver.improve([capability], accuracy_threshold=1.0)

    assert observed["threshold"] == 1.0
    assert observed["collect"] == ("arithmetic", "say", 3)
    assert observed["extract"] == ("positive", "negative", 3, "bridge_arithmetic")
    assert observed["project"] == ("raw_direction", "negative")

    steering_actions = [action for action in log.actions if action.action_type == "steering"]
    assert len(steering_actions) == 1

    action = steering_actions[0]
    assert action.capability == "arithmetic"
    assert action.details["prime"] == "say"
    assert action.details["target_layer"] == 3
    assert action.details["projection_loss"] == pytest.approx(0.25)
    assert action.details["strength_range"] == [-0.5, 0.5]


def test_phase2b_falls_back_to_prime_on_steering_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = Capability.from_lists(
        name="arithmetic",
        prompts=["1+1="],
        problems=[("1+1=", "2")],
    )
    improver = AutonomousSelfImprover(_ToyModel(), tokenizer=object())

    monkeypatch.setattr(
        improver.scanner,
        "scan",
        lambda cap, accuracy_threshold, primes=None: _disconnected_analysis(cap),
    )
    monkeypatch.setattr(
        improver.scanner,
        "collect_contrastive_activations",
        lambda capability, best_prime, target_layer: ("positive", "negative"),
    )

    class FailingFeatureSteering:
        def __init__(self, _model: object, _backend: object) -> None:
            pass

        def _get_layers(self) -> list[object]:
            return [object() for _ in range(4)]

        def extract_contrastive_direction(
            self,
            positive_activations: object,
            negative_activations: object,
            layer: int,
            label: str,
        ) -> SimpleNamespace:
            del positive_activations, negative_activations, layer, label
            raise RuntimeError("steering failed")

    import modelcypher.experimental.interpretability.feature_steering as steering_mod

    monkeypatch.setattr(steering_mod, "FeatureSteering", FailingFeatureSteering)

    log = improver.improve([capability], accuracy_threshold=1.0)

    assert all(action.action_type != "steering" for action in log.actions)
    fallback_actions = [action for action in log.actions if action.action_type == "apply_prime"]
    assert len(fallback_actions) == 1
    assert fallback_actions[0].details["prime"] == "say"
