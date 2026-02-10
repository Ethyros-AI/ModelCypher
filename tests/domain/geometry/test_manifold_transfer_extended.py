from __future__ import annotations

from types import SimpleNamespace

import pytest

import modelcypher.core.domain.geometry.manifold_transfer as transfer_mod
from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    LocalCurvature,
)
from modelcypher.core.domain.geometry.riemannian_density import ConceptVolume


def _local_curvature(point, mean_sectional: float) -> LocalCurvature:
    return LocalCurvature(
        point=point,
        mean_sectional=mean_sectional,
        variance_sectional=0.0,
        min_sectional=mean_sectional,
        max_sectional=mean_sectional,
        principal_directions=None,
        principal_curvatures=None,
        sign=CurvatureSign.FLAT,
        scalar_curvature=mean_sectional,
    )


def _concept_volume(any_backend, concept_id: str = "c") -> ConceptVolume:
    b = any_backend
    centroid = b.array([0.0, 0.0])
    return ConceptVolume(
        concept_id=concept_id,
        centroid=centroid,
        covariance=b.array([[1.0, 0.0], [0.0, 1.0]]),
        geodesic_radius=1.0,
        local_curvature=_local_curvature(centroid, 0.1),
        num_samples=4,
    )


def test_required_anchor_count_and_space_form_scale(any_backend, monkeypatch) -> None:
    b = any_backend

    assert transfer_mod._required_anchor_count([], b) == 0

    anchors = [b.array([0.0, 0.0]), b.array([1.0, 0.0]), b.array([2.0, 0.0])]
    required = transfer_mod._required_anchor_count(anchors, b)
    assert required >= 1

    monkeypatch.setattr(
        transfer_mod,
        "geodesic_svd",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert transfer_mod._required_anchor_count(anchors, b) == len(anchors)

    reference = b.array([1.0, 2.0])
    assert transfer_mod._space_form_scale(1.0, 0.0, b, reference) == pytest.approx(1.0, abs=1e-8)
    assert transfer_mod._space_form_scale(0.0, 1.0, b, reference) == pytest.approx(1.0, abs=1e-8)

    pos = transfer_mod._space_form_scale(0.25, 1.0, b, reference)
    neg = transfer_mod._space_form_scale(-0.25, 1.0, b, reference)
    assert pos > 0.0
    assert neg > 0.0
    assert neg > pos


def test_transfer_point_to_dict_and_report_count(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(transfer_mod, "get_default_backend", lambda: b)

    profile = transfer_mod.AnchorDistanceProfile(
        concept_id="concept",
        anchor_ids=["a", "b"],
        distances=b.array([1.0, 2.0]),
        weights=b.array([0.5, 0.5]),
        source_curvature=None,
        source_volume=None,
    )
    point = transfer_mod.TransferPoint(
        concept_id="concept",
        source_profile=profile,
        coordinates=b.array([0.1, -0.2]),
        projected_volume=None,
        stress=0.2,
        curvature_mismatch=0.05,
        confidence_components=transfer_mod.TransferConfidenceComponents(0.2, 0.9, 0.05),
    )

    payload = point.to_dict()
    assert payload["conceptId"] == "concept"
    assert payload["coordinates"] == pytest.approx([0.1, -0.2], abs=1e-6)
    assert payload["numAnchors"] == 2
    assert payload["meanSourceDistance"] == pytest.approx(1.5, abs=1e-6)

    report = transfer_mod.TransferReport(
        transfers=[point],
        source_model_id="s",
        target_model_id="t",
        mean_stress=0.2,
        max_stress=0.2,
        min_stress=0.2,
        median_stress=0.2,
        std_stress=0.0,
        source_mean_curvature=None,
        target_mean_curvature=None,
    )
    assert report.transfer_count == 1


def test_project_transfer_batch_project_volume_and_confidence(any_backend, monkeypatch) -> None:
    b = any_backend
    monkeypatch.setattr(transfer_mod, "get_default_backend", lambda: b)
    monkeypatch.setattr(transfer_mod, "regularization_epsilon", lambda *_args, **_kwargs: 1e6)

    projector = transfer_mod.CrossManifoldProjector()

    concept = b.array([[1.0, 0.0], [0.8, 0.2], [0.9, 0.1]])
    source_anchors = {
        "a": b.array([[0.0, 0.0], [0.1, 0.1]]),
        "b": b.array([[1.0, 1.0], [0.9, 0.9]]),
        "c": b.array([[0.0, 1.0], [0.1, 0.9]]),
    }
    target_anchors = {
        "a": b.array([[0.2, 0.0], [0.3, 0.1]]),
        "b": b.array([[1.1, 0.9], [0.8, 1.0]]),
        "c": b.array([[0.1, 0.8], [0.2, 1.0]]),
    }

    profile = projector.compute_distance_profile(concept, "c1", source_anchors)
    if profile.source_curvature is None:
        profile.source_curvature = _local_curvature(b.array([0.0, 0.0]), 0.1)
    if profile.source_volume is None:
        profile.source_volume = _concept_volume(b, "c1")

    class _TargetProfile:
        global_mean = 0.3

        @staticmethod
        def curvature_at_point(point):
            return _local_curvature(point, 0.25)

    target_profile = _TargetProfile()

    transfer = projector.project(profile, target_anchors, target_profile)
    assert transfer.concept_id == "c1"
    assert transfer.stress >= 0.0
    assert transfer.anchor_stress
    assert transfer.confidence_components.anchor_factor > 0.0
    assert transfer.projected_volume is not None
    assert transfer.projected_volume.concept_id.endswith("_transferred")
    assert transfer.curvature_mismatch >= 0.0

    no_curv_volume = projector._project_volume(
        _concept_volume(b, "x"),
        b.array([0.0, 0.0]),
        source_curvature=None,
        target_curvature=None,
    )
    assert no_curv_volume.geodesic_radius == pytest.approx(1.0, abs=1e-6)

    with_curv_volume = projector._project_volume(
        _concept_volume(b, "y"),
        b.array([0.0, 0.0]),
        source_curvature=_local_curvature(b.array([0.0, 0.0]), 0.2),
        target_curvature=_local_curvature(b.array([0.0, 0.0]), -0.2),
    )
    assert with_curv_volume.geodesic_radius > 0.0

    components = projector._compute_confidence_components(
        normalized_stress=0.4,
        num_anchors=2,
        curvature_mismatch=0.1,
        required_anchors=0,
    )
    assert components.stress_factor == pytest.approx(0.4, abs=1e-6)
    assert components.anchor_factor == pytest.approx(0.0, abs=1e-6)
    assert components.curvature_factor == pytest.approx(0.1, abs=1e-6)

    bad_profile = transfer_mod.AnchorDistanceProfile(
        concept_id="bad",
        anchor_ids=profile.anchor_ids,
        distances=profile.distances,
        weights=profile.weights,
        source_curvature=None,
        source_volume=None,
    )

    def fake_project(profile_arg, *_args, **_kwargs):
        if profile_arg.concept_id == "bad":
            raise RuntimeError("forced failure")
        return transfer

    monkeypatch.setattr(projector, "project", fake_project)
    report = projector.transfer_batch(
        [profile, bad_profile],
        target_anchors,
        target_manifold_profile=target_profile,
        source_model_id="source-a",
        target_model_id="target-b",
    )

    assert report.transfer_count == 1
    assert report.source_model_id == "source-a"
    assert report.target_model_id == "target-b"
    assert report.mean_stress == pytest.approx(transfer.stress, abs=1e-6)
    assert report.median_stress == pytest.approx(transfer.stress, abs=1e-6)
    assert report.source_mean_curvature is not None
    assert report.target_mean_curvature == pytest.approx(0.3, abs=1e-6)


def test_project_concept_delegates(monkeypatch, any_backend) -> None:
    b = any_backend

    calls: dict[str, object] = {}

    def fake_compute(self, concept_activations, concept_id, source_anchor_activations):
        calls["concept"] = concept_activations
        calls["concept_id"] = concept_id
        calls["source"] = source_anchor_activations
        return "profile"

    def fake_project(self, profile, target_anchor_activations):
        calls["profile"] = profile
        calls["target"] = target_anchor_activations
        return "transfer-result"

    monkeypatch.setattr(transfer_mod.CrossManifoldProjector, "compute_distance_profile", fake_compute)
    monkeypatch.setattr(transfer_mod.CrossManifoldProjector, "project", fake_project)

    concept = b.array([[1.0, 0.0]])
    source = {"a": b.array([[0.0, 0.0]])}
    target = {"a": b.array([[0.1, 0.0]])}

    out = transfer_mod.project_concept(concept, "trait", source, target)

    assert out == "transfer-result"
    assert calls["concept_id"] == "trait"
    assert calls["profile"] == "profile"
    assert calls["source"] is source
    assert calls["target"] is target
