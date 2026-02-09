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

import modelcypher.core.domain.geometry.spatial_3d as spatial_3d


@dataclass
class _Anchor:
    id: str
    name: str
    prompt: str
    expected_x: float
    expected_y: float
    expected_z: float
    category: str
    axis: str


def _anchors() -> list[_Anchor]:
    return [
        _Anchor("a1", "right_hand", "", 1.0, 0.0, 0.0, "lateral", "x_lateral"),
        _Anchor("a2", "left_hand", "", -1.0, 0.0, 0.0, "lateral", "x_lateral"),
        _Anchor("a3", "ceiling", "", 0.0, 1.0, 0.0, "vertical", "y_vertical"),
        _Anchor("a4", "floor", "", 0.0, -1.0, 0.0, "vertical", "y_vertical"),
        _Anchor("a5", "foreground", "", 0.0, 0.0, 1.0, "depth", "z_depth"),
        _Anchor("a6", "background", "", 0.0, 0.0, -1.0, "depth", "z_depth"),
        _Anchor("a7", "anvil", "", 0.0, -0.8, 0.2, "mass", "y_vertical"),
        _Anchor("a8", "balloon", "", 0.0, 0.8, -0.2, "furniture", "y_vertical"),
    ]


def test_backend_numeric_helpers(any_backend) -> None:
    b = any_backend
    arr = b.array([1.0, float("nan"), float("inf"), -float("inf")], dtype="float32")

    isnan = spatial_3d._backend_isnan(b, arr)
    isinf = spatial_3d._backend_isinf(b, arr)
    cleaned = spatial_3d._backend_nan_to_num(b, arr, nan_val=0.0, posinf_val=9.0, neginf_val=-9.0)

    assert b.tolist(isnan) == [False, True, False, False]
    assert b.tolist(isinf) == [False, False, True, True]
    assert b.tolist(cleaned) == pytest.approx([1.0, 0.0, 9.0, -9.0], abs=1e-6)

    corr = spatial_3d._backend_corrcoef(b, b.array([1.0, 2.0, 3.0]), b.array([2.0, 4.0, 6.0]))
    corr_flat = spatial_3d._backend_corrcoef(b, b.array([1.0, 1.0, 1.0]), b.array([2.0, 3.0, 4.0]))
    assert corr == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert corr_flat == 0.0

    assert spatial_3d._backend_vector_norm(b, b.array([3.0, 4.0])) == pytest.approx(5.0)
    assert spatial_3d._backend_vector_dot(b, b.array([1.0, 2.0]), b.array([3.0, 4.0])) == pytest.approx(11.0)
    assert spatial_3d._backend_var(b, b.array([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 3.0)
    assert spatial_3d._backend_std(b, b.array([1.0, 2.0, 3.0])) == pytest.approx((2.0 / 3.0) ** 0.5)
    assert b.tolist(spatial_3d._backend_clip(b, b.array([-2.0, 0.5, 3.0]), -1.0, 1.0)) == pytest.approx(
        [-1.0, 0.5, 1.0]
    )

    assert spatial_3d._scalar_isnan(float("nan")) is True
    assert spatial_3d._scalar_isinf(float("inf")) is True


def test_get_spatial_anchors_by_axis_and_orthogonality(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(spatial_3d, "get_spatial_concepts", lambda: _anchors())

    vertical = spatial_3d.get_spatial_anchors_by_axis("y_vertical")
    lateral = spatial_3d.get_spatial_anchors_by_axis("x_lateral")
    depth = spatial_3d.get_spatial_anchors_by_axis("z_depth")
    all_anchors = spatial_3d.get_spatial_anchors_by_axis("unknown")

    assert any(a.name == "ceiling" for a in vertical)
    assert all(a.category == "lateral" for a in lateral)
    assert all(a.category == "depth" for a in depth)
    assert len(all_anchors) == len(_anchors())

    anchors_for_axes = _anchors()[:6]
    activations = b.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    ortho = spatial_3d._compute_axis_orthogonality(b, activations, anchors_for_axes)
    assert ortho["x_y_orthogonality"] == pytest.approx(1.0, abs=1e-6)
    assert ortho["y_z_orthogonality"] == pytest.approx(1.0, abs=1e-6)
    assert ortho["x_z_orthogonality"] == pytest.approx(1.0, abs=1e-6)


def test_spatial_stereoscopy_and_occlusion(any_backend) -> None:
    b = any_backend
    stereoscopy = spatial_3d.SpatialStereoscopy(backend=b)

    too_few = stereoscopy.analyze_scene(
        viewpoint_activations={"front": b.array([0.0, 0.0, 0.0])},
        scene_prompts=[spatial_3d.STEREOSCOPIC_SCENES[0]],
    )
    assert too_few.parallax_correlation == 0.0
    assert too_few.depth_axis_detected is False

    prompts = [p for p in spatial_3d.STEREOSCOPIC_SCENES if p.scene_id == "cube"][:3]
    scene_acts = {
        "front": b.array([0.0, 0.0, 0.0]),
        "left": b.array([-0.5, 0.0, 0.2]),
        "right": b.array([0.5, 0.0, 0.2]),
    }
    stereo = stereoscopy.analyze_scene(scene_acts, prompts)
    assert stereo.scene_id == "cube"
    assert len(stereo.measured_parallax) == 3
    assert 0.0 <= stereo.perspective_consistency <= 1.0

    occlusion = spatial_3d.OcclusionProber(backend=b)
    probe = spatial_3d.OCCLUSION_PROBES[0]
    occ = occlusion.analyze(
        a_front_activation=b.array([0.0, 0.0, 0.0]),
        b_front_activation=b.array([0.4, 0.0, -0.2]),
        probe=probe,
    )
    assert occ.scene_id == probe.scene_id
    assert occ.z_shift_detected is True
    assert occ.occlusion_understood is True


def test_gravity_density_and_full_analyzer(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(spatial_3d, "get_spatial_concepts", lambda: _anchors())

    anchor_activations = {
        "ceiling": b.array([0.0, 2.0, 0.0]),
        "floor": b.array([0.0, -2.0, 0.0]),
        "anvil": b.array([0.0, -1.5, 0.2]),
        "balloon": b.array([0.0, 1.2, -0.2]),
        "right_hand": b.array([1.0, 0.0, 0.0]),
        "left_hand": b.array([-1.0, 0.0, 0.0]),
        "foreground": b.array([0.0, 0.0, 1.0]),
        "background": b.array([0.0, 0.0, -1.0]),
    }

    gravity = spatial_3d.GravityGradientAnalyzer(backend=b).analyze(anchor_activations)
    assert gravity.gravity_axis_detected is True
    assert gravity.gravity_direction is not None
    assert -1.0 <= gravity.mass_correlation <= 1.0
    assert gravity.sink_anchors
    assert gravity.float_anchors

    density = spatial_3d.VolumetricDensityProber(backend=b).analyze(anchor_activations)
    assert density.anchor_densities
    assert -1.0 <= density.density_mass_correlation <= 1.0
    assert 0.0 <= density.inverse_square_compliance <= 1.0

    analyzer = spatial_3d.Spatial3DAnalyzer(backend=b)
    report = analyzer.full_analysis(
        anchor_activations=anchor_activations,
        stereoscopy_activations={
            "cube": {
                "front": b.array([0.0, 0.0, 0.0]),
                "left": b.array([-0.4, 0.0, 0.2]),
                "right": b.array([0.4, 0.0, 0.2]),
            }
        },
        occlusion_activations={
            "box_ball": (b.array([0.0, 0.0, 0.0]), b.array([0.3, 0.0, -0.1]))
        },
    )

    assert report.axis_orthogonality
    assert report.stereoscopy_results
    assert report.occlusion_results
    assert 0.0 <= report.world_model_score <= 1.0
    assert "world_model_score" in report.to_dict()

