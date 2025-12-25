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

from modelcypher.core.domain.geometry.persona_vector_monitor import (
    Configuration,
    PersonaTraitDefinition,
    PersonaVectorMonitor,
)


def test_persona_vector_monitor_extract_and_drift() -> None:
    trait = PersonaTraitDefinition(
        id="truthful",
        name="Truthful",
        description="Honesty trait",
        positive_prompts=["Tell the truth"],
        negative_prompts=["Make it up"],
    )
    config = Configuration(
        persona_traits=[trait], correlation_threshold=0.1, normalize_vectors=True
    )

    positive = [[1.0, 0.0], [1.0, 0.0]]
    negative = [[0.0, 1.0], [0.0, 1.0]]
    vector = PersonaVectorMonitor.extract_vector(
        positive_activations=positive,
        negative_activations=negative,
        trait=trait,
        configuration=config,
        layer_index=4,
        model_id="model-1",
    )
    assert vector is not None

    bundle = PersonaVectorMonitor.extract_bundle(
        activations_per_trait={"truthful": (positive, negative)},
        configuration=config,
        layer_index=4,
        model_id="model-1",
    )
    assert bundle.vectors

    positions = PersonaVectorMonitor.measure_all_positions([1.0, -1.0], bundle, None)
    baseline = PersonaVectorMonitor.create_baseline(
        positions, model_id="model-1", is_pretrained_baseline=True
    )
    positions_with_baseline = PersonaVectorMonitor.measure_all_positions(
        [0.0, 1.0], bundle, baseline
    )
    drift = PersonaVectorMonitor.compute_drift_metrics(positions_with_baseline, step=10)
    assert drift.step == 10
