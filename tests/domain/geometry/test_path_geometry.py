# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import dataclasses
from uuid import UUID

import pytest

import modelcypher.core.domain.geometry.path_geometry as mod


# ---------------------------------------------------------------------------
# Data model tests (pure Python, no backend needed)
# ---------------------------------------------------------------------------


class TestPathNode:
    def test_instantiation(self) -> None:
        node = mod.PathNode(gate_id="gate_1", token_index=0, entropy=1.5)
        assert node.gate_id == "gate_1"
        assert node.token_index == 0
        assert node.entropy == 1.5
        assert node.embedding is None

    def test_with_embedding(self) -> None:
        node = mod.PathNode(gate_id="g", token_index=1, entropy=0.5, embedding=[1.0, 2.0])
        assert node.embedding == [1.0, 2.0]

    def test_frozen(self) -> None:
        node = mod.PathNode(gate_id="g", token_index=0, entropy=0.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            node.gate_id = "other"


class TestPathSignature:
    def test_instantiation(self) -> None:
        nodes = [mod.PathNode("a", 0, 1.0), mod.PathNode("b", 1, 2.0)]
        sig = mod.PathSignature(model_id="m1", prompt_id="p1", nodes=nodes)
        assert sig.model_id == "m1"
        assert sig.prompt_id == "p1"
        assert len(sig.nodes) == 2
        assert isinstance(sig.id, UUID)

    def test_gate_sequence(self) -> None:
        nodes = [mod.PathNode("x", 0, 0.1), mod.PathNode("y", 1, 0.2), mod.PathNode("z", 2, 0.3)]
        sig = mod.PathSignature(model_id="m", prompt_id="p", nodes=nodes)
        assert sig.gate_sequence == ["x", "y", "z"]

    def test_gate_sequence_empty(self) -> None:
        sig = mod.PathSignature(model_id="m", prompt_id="p", nodes=[])
        assert sig.gate_sequence == []

    def test_frozen(self) -> None:
        sig = mod.PathSignature(model_id="m", prompt_id="p", nodes=[])
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            sig.model_id = "other"


class TestAlignmentOp:
    def test_all_values(self) -> None:
        assert mod.AlignmentOp.match == "match"
        assert mod.AlignmentOp.insert == "insert"
        assert mod.AlignmentOp.delete == "delete"
        assert mod.AlignmentOp.substitute == "substitute"

    def test_str_subclass(self) -> None:
        assert isinstance(mod.AlignmentOp.match, str)

    def test_count(self) -> None:
        assert len(mod.AlignmentOp) == 4


class TestAlignmentStep:
    def test_instantiation(self) -> None:
        node = mod.PathNode("g", 0, 1.0)
        step = mod.AlignmentStep(op=mod.AlignmentOp.match, node_a=node, node_b=node, cost=0.0)
        assert step.op == mod.AlignmentOp.match
        assert step.cost == 0.0

    def test_none_nodes(self) -> None:
        step = mod.AlignmentStep(op=mod.AlignmentOp.insert, node_a=None, node_b=None, cost=1.0)
        assert step.node_a is None
        assert step.node_b is None

    def test_frozen(self) -> None:
        step = mod.AlignmentStep(op=mod.AlignmentOp.delete, node_a=None, node_b=None, cost=0.5)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            step.cost = 0.0


class TestPathComparison:
    def test_instantiation(self) -> None:
        result = mod.PathComparison(total_distance=2.5, normalized_distance=0.5, alignment=[])
        assert result.total_distance == 2.5
        assert result.normalized_distance == 0.5
        assert result.alignment == []

    def test_frozen(self) -> None:
        result = mod.PathComparison(total_distance=1.0, normalized_distance=0.1, alignment=[])
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.total_distance = 0.0


class TestFrechetResult:
    def test_instantiation(self) -> None:
        result = mod.FrechetResult(distance=3.14, optimal_coupling=[(0, 0), (1, 1)])
        assert result.distance == 3.14
        assert len(result.optimal_coupling) == 2

    def test_frozen(self) -> None:
        result = mod.FrechetResult(distance=0.0, optimal_coupling=[])
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.distance = 1.0


class TestDTWResult:
    def test_instantiation(self) -> None:
        result = mod.DTWResult(
            total_cost=5.0, normalized_cost=1.0, warping_path=[(0, 0)], compression_ratio=0.8
        )
        assert result.total_cost == 5.0
        assert result.compression_ratio == 0.8

    def test_frozen(self) -> None:
        result = mod.DTWResult(total_cost=0.0, normalized_cost=0.0, warping_path=[], compression_ratio=1.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.total_cost = 1.0


class TestTruncatedSignature:
    def test_instantiation(self) -> None:
        sig = mod.TruncatedSignature(
            level1=[1.0, 2.0], level2=[[0.1, 0.2], [0.3, 0.4]], signed_area=0.5, signature_norm=2.5
        )
        assert sig.level1 == [1.0, 2.0]
        assert sig.signed_area == 0.5
        assert sig.signature_norm == 2.5

    def test_frozen(self) -> None:
        sig = mod.TruncatedSignature(level1=[], level2=[], signed_area=0.0, signature_norm=0.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            sig.signed_area = 1.0


class TestEntropyPathAnalysis:
    def test_instantiation(self) -> None:
        result = mod.EntropyPathAnalysis(
            total_entropy=10.0,
            mean_entropy=2.0,
            entropy_variance=0.5,
            max_entropy=4.0,
            max_entropy_index=2,
            mean_gradient=0.1,
        )
        assert result.total_entropy == 10.0
        assert result.max_entropy_index == 2

    def test_frozen(self) -> None:
        result = mod.EntropyPathAnalysis(
            total_entropy=0.0,
            mean_entropy=0.0,
            entropy_variance=0.0,
            max_entropy=0.0,
            max_entropy_index=0,
            mean_gradient=0.0,
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.mean_entropy = 1.0


class TestLocalGeometry:
    def test_instantiation(self) -> None:
        result = mod.LocalGeometry(
            curvatures=[0.1, 0.2],
            mean_curvature=0.15,
            max_curvature=0.2,
            total_curvature=0.3,
            torsions=[0.01],
            mean_torsion=0.01,
        )
        assert result.curvatures == [0.1, 0.2]
        assert result.mean_torsion == 0.01

    def test_frozen(self) -> None:
        result = mod.LocalGeometry(
            curvatures=[], mean_curvature=0.0, max_curvature=0.0, total_curvature=0.0, torsions=[], mean_torsion=0.0
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.mean_curvature = 1.0


class TestComprehensiveComparison:
    def test_instantiation(self) -> None:
        lev = mod.PathComparison(total_distance=1.0, normalized_distance=0.5, alignment=[])
        frechet = mod.FrechetResult(distance=2.0, optimal_coupling=[(0, 0)])
        dtw = mod.DTWResult(total_cost=3.0, normalized_cost=1.0, warping_path=[], compression_ratio=1.0)
        comp = mod.ComprehensiveComparison(
            levenshtein=lev, frechet=frechet, dtw=dtw, signature_similarity=0.9
        )
        assert comp.levenshtein.total_distance == 1.0
        assert comp.frechet.distance == 2.0
        assert comp.dtw.total_cost == 3.0
        assert comp.signature_similarity == 0.9

    def test_frozen(self) -> None:
        lev = mod.PathComparison(total_distance=0.0, normalized_distance=0.0, alignment=[])
        frechet = mod.FrechetResult(distance=0.0, optimal_coupling=[])
        dtw = mod.DTWResult(total_cost=0.0, normalized_cost=0.0, warping_path=[], compression_ratio=1.0)
        comp = mod.ComprehensiveComparison(levenshtein=lev, frechet=frechet, dtw=dtw, signature_similarity=0.0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            comp.signature_similarity = 1.0


# ---------------------------------------------------------------------------
# PathGeometry computational tests (backend-dependent)
# ---------------------------------------------------------------------------


def _make_path(gate_ids: list[str], entropies: list[float] | None = None) -> mod.PathSignature:
    if entropies is None:
        entropies = [1.0] * len(gate_ids)
    nodes = [mod.PathNode(gid, i, e) for i, (gid, e) in enumerate(zip(gate_ids, entropies))]
    return mod.PathSignature(model_id="test", prompt_id="test", nodes=nodes)


def _simple_embeddings() -> dict[str, list[float]]:
    return {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
    }


class TestPathGeometryCompare:
    def test_identical_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b", "c"])
        result = mod.PathGeometry.compare(path, path, _simple_embeddings())
        assert isinstance(result, mod.PathComparison)
        assert result.total_distance == pytest.approx(0.0, abs=1e-6)
        assert result.normalized_distance == pytest.approx(0.0, abs=1e-6)

    def test_different_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path_a = _make_path(["a", "b"])
        path_b = _make_path(["b", "c"])
        result = mod.PathGeometry.compare(path_a, path_b, _simple_embeddings())
        assert result.total_distance > 0.0

    def test_empty_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path([])
        result = mod.PathGeometry.compare(path, path, _simple_embeddings())
        assert result.total_distance == pytest.approx(0.0, abs=1e-6)


class TestPathGeometryFrechet:
    def test_identical_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b"])
        result = mod.PathGeometry.frechet_distance(path, path, _simple_embeddings())
        assert isinstance(result, mod.FrechetResult)
        assert result.distance == pytest.approx(0.0, abs=1e-6)

    def test_different_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path_a = _make_path(["a", "a"])
        path_b = _make_path(["c", "c"])
        result = mod.PathGeometry.frechet_distance(path_a, path_b, _simple_embeddings())
        assert result.distance > 0.0


class TestPathGeometryDTW:
    def test_identical_paths(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b", "c"])
        result = mod.PathGeometry.dynamic_time_warping(path, path, _simple_embeddings())
        assert isinstance(result, mod.DTWResult)
        assert result.total_cost == pytest.approx(0.0, abs=1e-6)
        assert result.compression_ratio == pytest.approx(1.0, abs=1e-6)


class TestPathGeometryEntropyAnalysis:
    def test_basic(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b", "c"], entropies=[1.0, 3.0, 2.0])
        result = mod.PathGeometry.analyze_entropy_path(path)
        assert isinstance(result, mod.EntropyPathAnalysis)
        assert result.total_entropy == pytest.approx(6.0, abs=1e-6)
        assert result.mean_entropy == pytest.approx(2.0, abs=1e-6)
        assert result.max_entropy == pytest.approx(3.0, abs=1e-6)
        assert result.max_entropy_index == 1

    def test_single_node(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a"], entropies=[5.0])
        result = mod.PathGeometry.analyze_entropy_path(path)
        assert result.total_entropy == pytest.approx(5.0, abs=1e-6)
        assert result.entropy_variance == pytest.approx(0.0, abs=1e-6)


class TestPathGeometrySignature:
    def test_compute_and_similarity(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b", "c"])
        sig = mod.PathGeometry.compute_signature(path, _simple_embeddings())
        assert isinstance(sig, mod.TruncatedSignature)
        self_sim = mod.PathGeometry.signature_similarity(sig, sig)
        assert self_sim == pytest.approx(1.0, abs=1e-4)

    def test_different_signatures(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path_a = _make_path(["a", "b"])
        path_b = _make_path(["c", "a"])
        sig_a = mod.PathGeometry.compute_signature(path_a, _simple_embeddings())
        sig_b = mod.PathGeometry.compute_signature(path_b, _simple_embeddings())
        sim = mod.PathGeometry.signature_similarity(sig_a, sig_b)
        assert -1.0 <= sim <= 1.0


class TestPathGeometryLocalGeometry:
    def test_basic(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path = _make_path(["a", "b", "c", "a"])
        result = mod.PathGeometry.compute_local_geometry(path, _simple_embeddings())
        assert isinstance(result, mod.LocalGeometry)


class TestPathGeometryComprehensive:
    def test_comprehensive_compare(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        path_a = _make_path(["a", "b", "c"])
        path_b = _make_path(["a", "c", "b"])
        result = mod.PathGeometry.comprehensive_compare(path_a, path_b, _simple_embeddings())
        assert isinstance(result, mod.ComprehensiveComparison)
        assert isinstance(result.levenshtein, mod.PathComparison)
        assert isinstance(result.frechet, mod.FrechetResult)
        assert isinstance(result.dtw, mod.DTWResult)
        assert -1.0 <= result.signature_similarity <= 1.0


class TestGetPathGeometry:
    def test_without_backend(self, any_backend, monkeypatch) -> None:
        monkeypatch.setattr(mod, "get_default_backend", lambda: any_backend)
        pg = mod.get_path_geometry()
        assert pg is not None

    def test_with_backend(self, any_backend) -> None:
        pg = mod.get_path_geometry(backend=any_backend)
        assert isinstance(pg, mod.BackendPathGeometry)
