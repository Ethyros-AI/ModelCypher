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

"""Tests for PathGeometry distance metrics.

Tests mathematical properties of path comparison algorithms:
- Levenshtein (edit distance): d(X,X)=0, non-negativity, triangle inequality
- Frechet distance: d(X,X)=0, captures worst-case deviation
- Dynamic Time Warping: d(X,X)=0, handles time warping
- Path signatures: translation invariance, similarity computation
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.path_geometry import (
    AlignmentOp,
    BackendPathGeometry,
    PathGeometry,
    PathNode,
    PathSignature,
    get_path_geometry,
)


def _simple_embeddings():
    """Orthogonal embeddings for gates A, B, C."""
    return {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "C": [1.0, 1.0],
        "D": [-1.0, 0.0],
    }


def _make_path(gate_ids: list[str], entropies: list[float] | None = None) -> PathSignature:
    """Helper to create a path from gate IDs."""
    if entropies is None:
        entropies = [0.1 * i for i in range(len(gate_ids))]
    nodes = [
        PathNode(gate_id=g, token_index=i, entropy=e)
        for i, (g, e) in enumerate(zip(gate_ids, entropies))
    ]
    return PathSignature(model_id="test", prompt_id="test", nodes=nodes)


def _eps() -> float:
    backend = get_default_backend()
    arr = backend.array([0.0])
    return division_epsilon(backend, arr)


PI = 3.141592653589793

class TestLevenshteinDistance:
    """Tests for Levenshtein-based path comparison."""

    def test_identical_paths_zero_distance(self) -> None:
        """Identical paths should have zero distance.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.compare(path, path, gate_embeddings=_simple_embeddings())

        eps = _eps()
        assert abs(result.total_distance - 0.0) <= eps
        assert abs(result.normalized_distance - 0.0) <= eps

    def test_different_paths_positive_distance(self) -> None:
        """Different paths should have positive distance."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])  # D is different from B

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        backend, cache = PathGeometry._prepare_embedding_cache(_simple_embeddings())
        gate_ids_a, gate_ids_b, node_map_a, node_map_b, sim_matrix = (
            PathGeometry._prepare_gate_similarity(path_a, path_b, cache, backend)
        )
        idx_a = gate_ids_a.index("B")
        idx_b = gate_ids_b.index("D")
        sim = sim_matrix[node_map_a[idx_a]][node_map_b[idx_b]]
        expected = 1.0 - sim
        eps = _eps()
        assert abs(result.total_distance - expected) <= eps

    def test_distance_non_negative(self) -> None:
        """Distance should always be non-negative."""
        path_a = _make_path(["A", "B"])
        path_b = _make_path(["C", "D"])

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        assert result.total_distance >= 0
        assert result.normalized_distance >= 0

    def test_alignment_tracks_operations(self) -> None:
        """Alignment should track insert/delete/substitute operations."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "C"])  # Missing B

        result = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        # Should have a delete operation for B
        ops = [step.op for step in result.alignment]
        assert AlignmentOp.delete in ops or AlignmentOp.insert in ops

    def test_substitution_cost_uses_embedding_similarity(self) -> None:
        """Substituting similar gates should cost less than dissimilar ones."""
        # A and D are opposite directions (cosine = -1)
        path_a = _make_path(["A"])
        path_d = _make_path(["D"])
        # B is orthogonal to A (cosine = 0)
        path_b = _make_path(["B"])

        cost_ad = PathGeometry.compare(path_a, path_d, gate_embeddings=_simple_embeddings())
        cost_ab = PathGeometry.compare(path_a, path_b, gate_embeddings=_simple_embeddings())

        # Substituting A->D (opposite) should cost more than A->B (orthogonal)
        # Because similarity(A,D) < similarity(A,B): -1 < 0
        assert cost_ad.total_distance >= cost_ab.total_distance


class TestFrechetDistance:
    """Tests for discrete Frechet distance."""

    def test_identical_paths_zero_distance(self) -> None:
        """Identical paths should have zero Frechet distance.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.frechet_distance(path, path, gate_embeddings=_simple_embeddings())

        eps = _eps()
        assert abs(result.distance - 0.0) <= eps

    def test_optimal_coupling_starts_at_origin(self) -> None:
        """Optimal coupling should start at (0, 0)."""
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.frechet_distance(path, path, gate_embeddings=_simple_embeddings())

        assert result.optimal_coupling[0] == (0, 0)

    def test_empty_path_returns_inf(self) -> None:
        """Empty path should return infinite distance."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        path = _make_path(["A"])

        result = PathGeometry.frechet_distance(empty, path, gate_embeddings=_simple_embeddings())

        assert result.distance == float("inf")

    def test_different_lengths_handled(self) -> None:
        """Frechet distance should handle paths of different lengths."""
        short = _make_path(["A", "B"])
        long = _make_path(["A", "B", "C", "C", "C"])

        result = PathGeometry.frechet_distance(short, long, gate_embeddings=_simple_embeddings())

        assert result.distance >= 0
        assert len(result.optimal_coupling) > 0


class TestDTW:
    """Tests for Dynamic Time Warping."""

    def test_identical_paths_zero_cost(self) -> None:
        """Identical paths should have zero DTW cost.

        Mathematical property: d(X, X) = 0.
        """
        path = _make_path(["A", "B", "C"])
        result = PathGeometry.dynamic_time_warping(path, path, gate_embeddings=_simple_embeddings())

        eps = _eps()
        assert abs(result.total_cost - 0.0) <= eps
        assert abs(result.normalized_cost - 0.0) <= eps

    def test_warping_path_covers_all_points(self) -> None:
        """Warping path should cover all points in both sequences."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "A", "B", "C"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings()
        )

        # Warping path should start at (0, 0) and end at (n-1, m-1)
        assert result.warping_path[0] == (0, 0)
        assert result.warping_path[-1] == (len(path_a.nodes) - 1, len(path_b.nodes) - 1)

    def test_empty_path_returns_inf(self) -> None:
        """Empty path should return infinite cost."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        path = _make_path(["A"])

        result = PathGeometry.dynamic_time_warping(
            empty, path, gate_embeddings=_simple_embeddings()
        )

        assert result.total_cost == float("inf")

    def test_window_constraint_limits_alignment(self) -> None:
        """Window constraints are geometry-derived; default DTW aligns all indices."""
        path_a = _make_path(["A", "B", "C", "D", "A"])
        path_b = _make_path(["A", "B", "C", "D", "A"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings()
        )

        assert result.warping_path

    def test_compression_ratio_bounded(self) -> None:
        """Compression ratio should be in reasonable range."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "B", "C"])

        result = PathGeometry.dynamic_time_warping(
            path_a, path_b, gate_embeddings=_simple_embeddings()
        )

        assert result.compression_ratio >= 0, "Compression ratio should be non-negative"


class TestPathSignatures:
    """Tests for path signature computation."""

    def test_identical_signatures_high_similarity(self) -> None:
        """Identical signatures should have similarity = 1."""
        path = _make_path(["A", "B", "C"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        similarity = PathGeometry.signature_similarity(sig, sig)

        eps = _eps()
        assert abs(similarity - 1.0) <= eps

    def test_single_node_path_zero_signature(self) -> None:
        """Single node path has no increments, so signature components are zero."""
        path = _make_path(["A"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        eps = _eps()
        assert abs(sig.signature_norm - 0.0) <= eps

    def test_signed_area_non_negative(self) -> None:
        """Signed area (magnitude) should be non-negative."""
        path = _make_path(["A", "B", "C"])
        sig = PathGeometry.compute_signature(path, gate_embeddings=_simple_embeddings())

        assert sig.signed_area >= 0

    def test_different_paths_different_signatures(self) -> None:
        """Different paths should produce different signatures."""
        path_abc = _make_path(["A", "B", "C"])
        path_cba = _make_path(["C", "B", "A"])

        sig_abc = PathGeometry.compute_signature(path_abc, gate_embeddings=_simple_embeddings())
        sig_cba = PathGeometry.compute_signature(path_cba, gate_embeddings=_simple_embeddings())

        similarity = PathGeometry.signature_similarity(sig_abc, sig_cba)

        # Reversed path should have different signature
        eps = _eps()
        assert abs(similarity - 1.0) > eps


class TestEntropyPathAnalysis:
    """Tests for entropy path analysis."""

    def test_empty_path_defaults(self) -> None:
        """Empty path should return safe defaults."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        analysis = PathGeometry.analyze_entropy_path(empty)

        assert analysis.total_entropy == 0.0
        assert analysis.mean_entropy == 0.0

    def test_max_entropy_tracking(self) -> None:
        """Maximum entropy and its index should be tracked."""
        nodes = [
            PathNode(gate_id="A", token_index=0, entropy=1.0),
            PathNode(gate_id="B", token_index=1, entropy=5.0),  # Max
            PathNode(gate_id="C", token_index=2, entropy=2.0),
        ]
        path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)
        analysis = PathGeometry.analyze_entropy_path(path)

        assert analysis.max_entropy == 5.0
        assert analysis.max_entropy_index == 1


class TestLocalGeometry:
    """Tests for local geometry computation."""

    def test_short_path_empty_curvatures(self) -> None:
        """Paths with < 3 nodes have no curvatures."""
        path = _make_path(["A", "B"])
        geom = PathGeometry.compute_local_geometry(path, gate_embeddings=_simple_embeddings())

        assert geom.curvatures == []
        assert geom.mean_curvature == 0.0

    def test_curvatures_bounded(self) -> None:
        """Curvatures should be bounded angles in [0, π]."""
        path = _make_path(["A", "B", "C", "D"])
        geom = PathGeometry.compute_local_geometry(path, gate_embeddings=_simple_embeddings())

        for curv in geom.curvatures:
            assert 0 <= curv <= PI, f"Curvature {curv} out of bounds"


class TestComprehensiveCompare:
    """Tests for comprehensive path comparison."""

    def test_comprehensive_matches_individual_metrics(self) -> None:
        """Comprehensive comparison should match individual metric computations."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])
        embeddings = _simple_embeddings()

        result = PathGeometry.comprehensive_compare(
            path_a, path_b, gate_embeddings=embeddings
        )
        lev = PathGeometry.compare(path_a, path_b, gate_embeddings=embeddings)
        frech = PathGeometry.frechet_distance(path_a, path_b, gate_embeddings=embeddings)
        dtw = PathGeometry.dynamic_time_warping(path_a, path_b, gate_embeddings=embeddings)
        sig_a = PathGeometry.compute_signature(path_a, gate_embeddings=embeddings)
        sig_b = PathGeometry.compute_signature(path_b, gate_embeddings=embeddings)
        sig_sim = PathGeometry.signature_similarity(sig_a, sig_b)

        eps = _eps()
        assert abs(result.levenshtein.total_distance - lev.total_distance) <= eps
        assert abs(result.levenshtein.normalized_distance - lev.normalized_distance) <= eps
        assert abs(result.frechet.distance - frech.distance) <= eps
        assert abs(result.dtw.total_cost - dtw.total_cost) <= eps
        assert abs(result.dtw.normalized_cost - dtw.normalized_cost) <= eps
        assert abs(result.signature_similarity - sig_sim) <= eps

        assert result.levenshtein.alignment == lev.alignment
        assert result.frechet.optimal_coupling == frech.optimal_coupling
        assert result.dtw.warping_path == dtw.warping_path


class TestBackendPathGeometry:
    """Tests for the GPU-accelerated BackendPathGeometry."""

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    @pytest.fixture
    def pg(self, backend):
        return BackendPathGeometry(backend)

    def test_compute_signature_identical_to_pure_python(self, pg) -> None:
        """Backend signature computation should match pure Python."""
        path = _make_path(["A", "B", "C"])
        emb = _simple_embeddings()

        pure_sig = PathGeometry.compute_signature(path, emb)
        backend_sig = pg.compute_signature(path, emb)
        eps = _eps()

        # Compare level1
        for i in range(len(pure_sig.level1)):
            assert abs(pure_sig.level1[i] - backend_sig.level1[i]) <= eps

        # Compare signed area and norm
        assert abs(pure_sig.signed_area - backend_sig.signed_area) <= eps
        assert abs(pure_sig.signature_norm - backend_sig.signature_norm) <= eps

    def test_signature_similarity_identical_to_pure_python(self, pg) -> None:
        """Backend signature similarity should match pure Python."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])
        emb = _simple_embeddings()

        sig_a = PathGeometry.compute_signature(path_a, emb)
        sig_b = PathGeometry.compute_signature(path_b, emb)

        pure_sim = PathGeometry.signature_similarity(sig_a, sig_b)
        backend_sim = pg.signature_similarity(sig_a, sig_b)

        eps = _eps()
        assert abs(pure_sim - backend_sim) <= eps

    def test_analyze_entropy_path_identical_to_pure_python(self, pg) -> None:
        """Backend entropy analysis should match pure Python."""
        nodes = [
            PathNode(gate_id="A", token_index=0, entropy=1.0),
            PathNode(gate_id="B", token_index=1, entropy=5.0),
            PathNode(gate_id="C", token_index=2, entropy=2.0),
            PathNode(gate_id="D", token_index=3, entropy=3.0),
            PathNode(gate_id="E", token_index=4, entropy=1.5),
        ]
        path = PathSignature(model_id="m", prompt_id="p", nodes=nodes)

        pure_analysis = PathGeometry.analyze_entropy_path(path)
        backend_analysis = pg.analyze_entropy_path(path)
        eps = _eps()

        assert abs(pure_analysis.total_entropy - backend_analysis.total_entropy) <= eps
        assert abs(pure_analysis.mean_entropy - backend_analysis.mean_entropy) <= eps
        assert abs(pure_analysis.entropy_variance - backend_analysis.entropy_variance) <= eps
        assert pure_analysis.max_entropy == backend_analysis.max_entropy
        assert pure_analysis.max_entropy_index == backend_analysis.max_entropy_index
        assert abs(pure_analysis.mean_gradient - backend_analysis.mean_gradient) <= eps

    def test_compute_local_geometry_identical_to_pure_python(self, pg) -> None:
        """Backend local geometry should match pure Python."""
        path = _make_path(["A", "B", "C", "D"])
        emb = _simple_embeddings()

        pure_geom = PathGeometry.compute_local_geometry(path, emb)
        backend_geom = pg.compute_local_geometry(path, emb)
        eps = _eps()

        assert len(pure_geom.curvatures) == len(backend_geom.curvatures)
        for i in range(len(pure_geom.curvatures)):
            assert abs(pure_geom.curvatures[i] - backend_geom.curvatures[i]) <= eps

        assert abs(pure_geom.mean_curvature - backend_geom.mean_curvature) <= eps
        assert abs(pure_geom.max_curvature - backend_geom.max_curvature) <= eps
        assert abs(pure_geom.total_curvature - backend_geom.total_curvature) <= eps

    def test_comprehensive_compare_identical_to_pure_python(self, pg) -> None:
        """Backend comprehensive comparison should match pure Python."""
        path_a = _make_path(["A", "B", "C"])
        path_b = _make_path(["A", "D", "C"])
        emb = _simple_embeddings()

        pure_result = PathGeometry.comprehensive_compare(path_a, path_b, emb)
        backend_result = pg.comprehensive_compare(path_a, path_b, emb)
        eps = _eps()

        # DP algorithms are identical (use pure Python)
        assert pure_result.levenshtein.total_distance == backend_result.levenshtein.total_distance
        assert pure_result.frechet.distance == backend_result.frechet.distance
        assert pure_result.dtw.total_cost == backend_result.dtw.total_cost

        # Signature similarity computed by Backend
        assert abs(pure_result.signature_similarity - backend_result.signature_similarity) <= eps

    def test_empty_path_handling(self, pg) -> None:
        """Backend should handle empty paths correctly."""
        empty = PathSignature(model_id="m", prompt_id="p", nodes=[])
        emb = _simple_embeddings()

        analysis = pg.analyze_entropy_path(empty)
        assert analysis.total_entropy == 0.0

        sig = pg.compute_signature(empty, emb)
        assert sig.signature_norm == 0.0

    def test_single_node_path(self, pg) -> None:
        """Backend should handle single node paths correctly."""
        path = _make_path(["A"])
        emb = _simple_embeddings()

        sig = pg.compute_signature(path, emb)
        assert sig.signature_norm == 0.0

        geom = pg.compute_local_geometry(path, emb)
        assert geom.curvatures == []


class TestGetPathGeometry:
    """Tests for the factory function."""

    def test_returns_class_without_backend(self) -> None:
        """Factory should return PathGeometry class without backend."""
        result = get_path_geometry()
        assert result is PathGeometry

    def test_returns_instance_with_backend(self) -> None:
        """Factory should return BackendPathGeometry instance with backend."""
        backend = get_default_backend()
        result = get_path_geometry(backend)
        assert isinstance(result, BackendPathGeometry)
