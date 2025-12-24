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

import pytest

from modelcypher.core.domain.geometry.concept_response_matrix import (
    AnchorActivation,
    AnchorMetadata,
    ConceptResponseMatrix,
)
from modelcypher.core.domain.geometry.shared_subspace_projector import (
    AlignmentMethod,
    Config,
    SharedSubspaceProjector,
)


def _make_crm(model_id: str, activations: dict[str, list[float]]) -> ConceptResponseMatrix:
    metadata = AnchorMetadata(
        total_count=len(activations),
        semantic_prime_count=len(activations),
        computational_gate_count=0,
        anchor_ids=sorted(activations.keys()),
    )
    crm = ConceptResponseMatrix(
        model_identifier=model_id,
        layer_count=1,
        hidden_dim=len(next(iter(activations.values()))),
        anchor_metadata=metadata,
    )
    crm.activations = {
        0: {
            anchor_id: AnchorActivation(anchor_id, 0, vector)
            for anchor_id, vector in activations.items()
        }
    }
    return crm


def test_discover_requires_min_samples() -> None:
    source = _make_crm("source", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    target = _make_crm("target", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=3)

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is None


def test_discover_procrustes_identity() -> None:
    source = _make_crm("source", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    target = _make_crm("target", {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0]})
    config = Config(alignment_method=AlignmentMethod.procrustes, min_samples=1, variance_threshold=0.9)

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is not None
    assert result.method == AlignmentMethod.procrustes
    assert result.alignment_error == pytest.approx(0.0, abs=1e-6)
    assert result.shared_dimension >= 1


def test_discover_shared_svd() -> None:
    source = _make_crm(
        "source",
        {"prime:a": [1.0, 0.0], "prime:b": [0.0, 1.0], "prime:c": [1.0, 1.0]},
    )
    target = _make_crm(
        "target",
        {"prime:a": [2.0, 0.0], "prime:b": [0.0, 2.0], "prime:c": [2.0, 2.0]},
    )
    config = Config(alignment_method=AlignmentMethod.shared_svd, min_samples=1, variance_threshold=0.8)

    result = SharedSubspaceProjector.discover(source, target, layer=0, config=config)
    assert result is not None
    assert result.shared_dimension >= 1
    assert len(result.alignment_strengths) == result.shared_dimension
    assert result.shared_variance_ratio > 0.0


def test_anchor_weighting_biases_shared_subspace() -> None:
    source = _make_crm(
        "source",
        {
            "prime:a": [1.0],
            "prime:b": [2.0],
            "gate:a": [-1.0],
            "gate:b": [-2.0],
        },
    )
    target = _make_crm(
        "target",
        {
            "prime:a": [1.0],
            "prime:b": [2.0],
            "gate:a": [2.0],
            "gate:b": [-2.0],
        },
    )
    unweighted = SharedSubspaceProjector.discover(
        source,
        target,
        layer=0,
        config=Config(
            alignment_method=AlignmentMethod.cca,
            min_samples=1,
            cca_regularization=0.0,
        ),
    )
    weighted = SharedSubspaceProjector.discover(
        source,
        target,
        layer=0,
        config=Config(
            alignment_method=AlignmentMethod.cca,
            min_samples=1,
            cca_regularization=0.0,
            anchor_weights={"prime:": 2.0, "gate:": 0.0},
        ),
    )
    assert unweighted is not None
    assert weighted is not None
    assert weighted.alignment_strengths[0] > unweighted.alignment_strengths[0]
    assert weighted.alignment_strengths[0] > 0.6
