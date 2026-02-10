# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass

import pytest

import modelcypher.core.domain.geometry.direct_lora_synthesis as dls
from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    TransferConfidenceComponents,
    TransferPoint,
)


def _transfer_point(any_backend, *, coordinates: list[float]) -> TransferPoint:
    b = any_backend
    profile = AnchorDistanceProfile(
        concept_id="c1",
        anchor_ids=["a1", "a2"],
        distances=b.array([1.0, 2.0]),
        weights=b.array([0.6, 0.4]),
        source_curvature=None,
        source_volume=None,
    )
    return TransferPoint(
        concept_id="c1",
        source_profile=profile,
        coordinates=b.array(coordinates),
        projected_volume=None,
        stress=0.1,
        anchor_stress={"a1": 0.05},
        curvature_mismatch=0.0,
        confidence_components=TransferConfidenceComponents(
            stress_factor=0.9,
            anchor_factor=0.8,
            curvature_factor=1.0,
        ),
    )


def test_determine_rank_edge_cases(any_backend) -> None:
    b = any_backend
    generator = dls.GeometricLoRAGenerator()

    assert generator._determine_rank(b.array([]), b) == 1
    assert generator._determine_rank(b.array([0.0, 0.0]), b) == 1
    assert generator._determine_rank(b.array([4.0, 2.0, 1e-7]), b) >= 1


def test_compute_layer_lora_success_and_properties(any_backend) -> None:
    b = any_backend
    generator = dls.GeometricLoRAGenerator()
    tp = _transfer_point(b, coordinates=[1.2, -0.8, 0.4])
    weight = b.array(
        [
            [0.3, -0.1],
            [0.1, 0.2],
            [-0.2, 0.4],
        ]
    )
    anchor_activations = {
        "a1": b.array([1.0, 0.0]),
        "a2": b.array([0.0, 1.0]),
    }

    layer = generator._compute_layer_lora(
        transfer_point=tp,
        current_weight=weight,
        layer_idx=4,
        proj_name="q_proj",
        anchor_activations=anchor_activations,
    )

    assert layer is not None
    assert layer.layer_idx == 4
    assert layer.projection_name == "q_proj"
    assert layer.in_features == 2
    assert layer.out_features == 3
    assert layer.rank >= 1
    assert layer.geometric_loss >= 0.0
    assert int(layer.delta_W.shape[0]) == 3
    assert int(layer.delta_W.shape[1]) == 2
    assert layer.effective_rank >= 1.0

    as_dict = layer.to_dict()
    assert as_dict["layer"] == 4
    assert as_dict["projection"] == "q_proj"


def test_compute_layer_lora_raises_on_missing_or_bad_shapes(any_backend) -> None:
    b = any_backend
    generator = dls.GeometricLoRAGenerator()
    weight = b.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])

    with pytest.raises(ValueError, match="No anchor activations"):
        generator._compute_layer_lora(
            transfer_point=_transfer_point(b, coordinates=[1.0, 1.0, 1.0]),
            current_weight=weight,
            layer_idx=0,
            proj_name="q_proj",
            anchor_activations={},
        )

    with pytest.raises(ValueError, match="Activation dimension"):
        generator._compute_layer_lora(
            transfer_point=_transfer_point(b, coordinates=[1.0, 1.0, 1.0]),
            current_weight=weight,
            layer_idx=0,
            proj_name="q_proj",
            anchor_activations={
                "a1": b.array([1.0, 0.0, 0.0]),
                "a2": b.array([0.0, 1.0, 0.0]),
            },
        )

    with pytest.raises(ValueError, match="Target position dimension"):
        generator._compute_layer_lora(
            transfer_point=_transfer_point(b, coordinates=[1.0, 1.0]),
            current_weight=weight,
            layer_idx=0,
            proj_name="q_proj",
            anchor_activations={
                "a1": b.array([1.0, 0.0]),
                "a2": b.array([0.0, 1.0]),
            },
        )

    with pytest.raises(ValueError, match="Near-zero input norm"):
        generator._compute_layer_lora(
            transfer_point=_transfer_point(b, coordinates=[1.0, 1.0, 1.0]),
            current_weight=weight,
            layer_idx=0,
            proj_name="q_proj",
            anchor_activations={
                "a1": b.array([0.0, 0.0]),
                "a2": b.array([0.0, 0.0]),
            },
        )


def test_generate_and_convenience_functions(monkeypatch, any_backend, tmp_path) -> None:
    b = any_backend
    tp = _transfer_point(b, coordinates=[1.0, 0.0, -1.0])
    weights = {
        0: {
            "q_proj": b.array([[1.0, 0.0], [0.0, 1.0], [0.2, -0.2]]),
            "v_proj": b.array([[0.5, 0.1], [-0.1, 0.3], [0.0, 0.2]]),
        }
    }
    anchor_acts = {"a1": b.array([1.0, 0.0]), "a2": b.array([0.0, 1.0])}

    lora = dls.generate_geometric_lora(
        transfer_point=tp,
        model_weights=weights,
        anchor_activations=anchor_acts,
        target_layers=[0],
        target_projections=["q_proj", "v_proj"],
    )
    assert lora.num_layers == 1
    assert lora.total_rank >= 2
    assert lora.num_parameters > 0
    assert lora.get_weights_for_layer(0)
    assert "conceptId" in lora.to_dict()
    assert lora.to_safetensors_dict()

    # Empty generation path.
    empty = dls.GeometricLoRAGenerator().generate(
        transfer_point=tp,
        model_weights={},
        anchor_activations=anchor_acts,
    )
    assert empty.weights == []
    assert empty.total_rank == 0
    assert empty.mean_geometric_loss == 1.0

    saved = {}
    monkeypatch.setattr(
        dls,
        "get_default_backend",
        lambda: type(
            "_FakeBackend",
            (),
            {"save_safetensors": staticmethod(lambda path, tensors, metadata=None: saved.update({"path": path, "metadata": metadata, "tensors": tensors}))},
        )(),
    )
    out_path = tmp_path / "geo.safetensors"
    dls.save_geometric_lora(lora, str(out_path), include_metadata=True)
    assert saved["path"] == str(out_path)
    assert saved["metadata"]["concept_id"] == "c1"
    assert saved["tensors"]

